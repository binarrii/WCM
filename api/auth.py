"""Password, WebAuthn, TOTP and role administration endpoints."""

import base64
import hashlib
import io
import json
import secrets
import time
import uuid
from typing import Literal
from urllib.parse import urlsplit

import pyotp
import qrcode
import qrcode.image.svg
from argon2 import PasswordHasher
from argon2.exceptions import VerificationError
from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import delete, func, insert, select, update
from webauthn import (
    base64url_to_bytes,
    generate_authentication_options,
    generate_registration_options,
    options_to_json,
    verify_authentication_response,
    verify_registration_response,
)
from webauthn.helpers import bytes_to_base64url
from webauthn.helpers.exceptions import WebAuthnException
from webauthn.helpers.structs import (
    AuthenticatorSelectionCriteria,
    PublicKeyCredentialDescriptor,
    ResidentKeyRequirement,
    UserVerificationRequirement,
)

from . import auth_store as store
from .avatar_images import MAX_AVATAR_BYTES, normalize_avatar

router = APIRouter(prefix="/auth", tags=["账户与权限"])
password_hasher = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=2)
DUMMY_HASH = password_hasher.hash(secrets.token_urlsafe(32))


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Signup(Payload):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.@-]*$")
    display_name: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=12, max_length=128)

    @field_validator("display_name")
    @classmethod
    def nonempty_name(cls, value):
        if not value.strip():
            raise ValueError("显示名称不能为空")
        return value.strip()


class Login(Payload):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class VerifyCode(Payload):
    challenge_id: str = Field(min_length=1, max_length=128)
    code: str = Field(min_length=6, max_length=64)


class SensitiveProof(Payload):
    password: str = Field(default="", max_length=128)
    code: str = Field(default="", max_length=64)


class VerificationOperation(Payload):
    operation: str = Field(
        max_length=256,
        pattern=r"^(POST /api/v1/auth/(password|2fa/(setup|disable|recovery-codes)|passkeys/register/options)|DELETE /api/v1/auth/passkeys/[A-Za-z0-9_-]{1,128}|PUT /api/v1/auth/(users/[A-Za-z0-9_-]{1,128}/(role|active)|roles/(user|admin)))$",
    )


class Reauthenticate(SensitiveProof, VerificationOperation):
    method: Literal["password", "totp"] = "password"


class ChangePassword(SensitiveProof):
    new_password: str = Field(min_length=12, max_length=128)


class PasskeyResponse(Payload):
    challenge_id: str = Field(min_length=1, max_length=128)
    credential: dict
    name: str = Field(default="我的 Passkey", min_length=1, max_length=80)


class RoleChange(Payload):
    role: Literal["admin", "user"]


class ActiveChange(Payload):
    active: bool


class PermissionChange(Payload):
    permissions: list[str] = Field(max_length=30)


def rate(request, action, account="", maximum=10):
    # Peer addresses are provided by the trusted reverse proxy / Uvicorn only.
    peer = request.client.host if request.client else "unknown"
    store.limit(f"ip:{action}:{peer}", maximum * 5)
    if account:
        store.limit(f"account:{action}:{account.lower()}", maximum)


def current(request, superadmin=False):
    identity = store.identity(request.cookies.get(store.COOKIE))
    if not identity:
        raise HTTPException(401, "请先登录")
    if superadmin and identity["user"]["role"] != "superadmin":
        raise HTTPException(403, "仅超级管理员可管理用户和角色权限")
    return identity


def locked_user(connection, identity):
    """Recheck authorization within the same transaction as sensitive mutations."""
    session = store.row(
        connection, store.sessions, store.sessions.c.id == identity["session"]["id"]
    )
    user = store.row(connection, store.users, store.users.c.id == identity["user"]["id"])
    if not session or session["expires_at"] <= time.time() or not user or not user["active"]:
        raise HTTPException(401, "登录已失效")
    return user


def issue_verification(connection, identity, operation):
    """Authorize one operation in this session, never a reusable time window."""
    token = store.challenge(
        connection,
        "operation-verification",
        identity["user"]["id"],
        identity["session"]["id"],
        {"operation": operation},
    )
    connection.execute(
        update(store.challenges)
        .where(store.challenges.c.id == store.digest(token))
        .values(expires_at=time.time() + 120)
    )
    return {"verification_token": token}


def consume_verification(connection, identity, request):
    """Consume atomically with the mutation, including across API replicas."""
    token = request.headers.get("X-WCM-Verification", "")
    if not token or len(token) > 128:
        raise HTTPException(403, "请为本次操作验证身份", headers={"X-WCM-Reauth": "required"})
    item = store.consume(connection, token, "operation-verification", identity["session"]["id"])
    if (
        item["user_id"] != identity["user"]["id"]
        or item["payload"]["operation"] != f"{request.method} {request.url.path}"
    ):
        raise HTTPException(403, "身份验证不适用于本次操作")


def log(connection, actor, action, target=None):
    connection.execute(
        insert(store.audit).values(
            id=str(uuid.uuid4()),
            actor_id=actor,
            action=action,
            target_id=target,
            created_at=time.time(),
        )
    )


def verify_password(user, password):
    try:
        valid = password_hasher.verify(user["password_hash"] if user else DUMMY_HASH, password)
    except VerificationError:
        valid = False
    return bool(valid and user and user["active"])


def verify_factor(connection, user, code, *, recovery=True):
    if not user["totp_secret"]:
        return True
    normalized = code.strip().replace(" ", "")
    if len(normalized) == 6 and normalized.isdigit():
        totp = pyotp.TOTP(store.cipher().decrypt(user["totp_secret"].encode()).decode())
        step = int(time.time() // 30)
        for candidate in (step, step - 1, step + 1):
            if candidate > (
                user["totp_last_step"] if user["totp_last_step"] is not None else -1
            ) and secrets.compare_digest(totp.at(candidate * 30), normalized):
                connection.execute(
                    update(store.users)
                    .where(store.users.c.id == user["id"])
                    .values(totp_last_step=candidate)
                )
                return True
    if recovery:
        hashes = json.loads(user["recovery_hashes"] or "[]")
        hashed = store.digest(normalized.upper())
        if hashed in hashes:
            hashes.remove(hashed)
            connection.execute(
                update(store.users)
                .where(store.users.c.id == user["id"])
                .values(recovery_hashes=json.dumps(hashes))
            )
            return True
    return False


def session_cookie_secure(request):
    # TLS may terminate before the WebUI proxy. The browser's allowlisted Origin
    # still identifies HTTPS without trusting arbitrary forwarded headers.
    origin = request.headers.get("origin", "")
    return (
        store.config.cookie_secure
        or request.url.scheme == "https"
        or (origin in store.config.origins and urlsplit(origin).scheme == "https")
    )


def new_session(connection, user, request, response, old_token=None):
    if old_token:
        connection.execute(
            delete(store.sessions).where(store.sessions.c.id == store.digest(old_token))
        )
    token = secrets.token_urlsafe(32)
    csrf = secrets.token_urlsafe(32)
    lifetime = store.config.session_hours * 3600
    connection.execute(
        insert(store.sessions).values(
            id=store.digest(token),
            user_id=user["id"],
            csrf=csrf,
            expires_at=time.time() + lifetime,
            verified_at=0,  # Legacy column; no time-based authorization is granted.
        )
    )
    response.set_cookie(
        store.COOKIE,
        token,
        max_age=lifetime,
        httponly=True,
        secure=session_cookie_secure(request),
        samesite="lax",
        path="/",
    )
    return {"user": store.public_user(connection, user), "csrf_token": csrf}


def finish_login(connection, user, request, response):
    log(connection, user["id"], "login")
    return new_session(connection, user, request, response, request.cookies.get(store.COOKIE))


def passkey_origin(request):
    origin = request.headers.get("origin", "")
    parsed = urlsplit(origin)
    host = parsed.hostname or ""
    if origin not in store.config.origins or not (
        parsed.scheme == "https" or (parsed.scheme == "http" and host == "localhost")
    ):
        raise HTTPException(400, "Passkey 需要配置 HTTPS 域名（本机 localhost 除外）")
    rp = store.config.rp_id
    if host != rp and not host.endswith("." + rp):
        raise HTTPException(400, "当前域名与 Passkey RP ID 配置不匹配")
    return origin


@router.get("/config")
def auth_config():
    with store.engine().connect() as connection:
        first = not connection.execute(select(store.users.c.id).limit(1)).first()
    return {"first_registration": first, "rp_id": store.config.rp_id, "min_password_length": 12}


@router.post("/register", status_code=201)
def register(payload: Signup, request: Request, response: Response):
    username = payload.username.lower()
    rate(request, "register", username, 5)
    hashed = password_hasher.hash(payload.password)
    with store.transaction() as connection:
        if store.row(connection, store.users, store.users.c.username == username):
            raise HTTPException(409, "该用户名已被使用")
        first = not connection.execute(select(store.users.c.id).limit(1)).first()
        user = {
            "id": str(uuid.uuid4()),
            "username": username,
            "display_name": payload.display_name,
            "password_hash": hashed,
            "role": "superadmin" if first else "user",
            "active": True,
            "totp_secret": None,
            "totp_last_step": -1,
            "recovery_hashes": "[]",
            "created_at": time.time(),
        }
        connection.execute(insert(store.users).values(**user))
        log(connection, user["id"], "register", user["role"])
        return finish_login(connection, user, request, response)


@router.post("/login")
def login(payload: Login, request: Request, response: Response):
    username = payload.username.strip().lower()
    rate(request, "login", username)
    with store.engine().connect() as connection:
        user = store.row(connection, store.users, store.users.c.username == username)
    if not verify_password(user, payload.password):
        raise HTTPException(401, "用户名或密码错误")
    with store.transaction() as connection:
        latest = store.row(connection, store.users, store.users.c.id == user["id"])
        if not latest["active"] or latest["password_hash"] != user["password_hash"]:
            raise HTTPException(401, "用户名或密码错误")
        if latest["totp_secret"]:
            token = store.challenge(connection, "mfa", latest["id"])
            return {"mfa_required": True, "challenge_id": token}
        return finish_login(connection, latest, request, response)


@router.post("/login/2fa")
def login_factor(payload: VerifyCode, request: Request, response: Response):
    rate(request, "mfa", store.digest(payload.challenge_id))
    with store.transaction() as connection:
        item = store.consume(connection, payload.challenge_id, "mfa")
        user = store.row(connection, store.users, store.users.c.id == item["user_id"])
        if (
            not user
            or not user["active"]
            or not user["totp_secret"]
            or not verify_factor(connection, user, payload.code)
        ):
            raise HTTPException(401, "验证码无效或已使用，请重新登录")
        return finish_login(connection, user, request, response)


@router.get("/me")
def me(request: Request):
    identity = current(request)
    return {"user": identity["user"], "csrf_token": identity["session"]["csrf"]}


@router.post("/logout")
def logout(request: Request, response: Response):
    identity = current(request)
    with store.transaction() as connection:
        connection.execute(
            delete(store.sessions).where(store.sessions.c.id == identity["session"]["id"])
        )
    response.delete_cookie(
        store.COOKIE, path="/", secure=session_cookie_secure(request), httponly=True, samesite="lax"
    )
    return {"ok": True}


@router.put("/avatar")
def upload_avatar(request: Request, file: UploadFile = File(...)):
    identity = current(request)
    rate(request, "avatar", identity["user"]["id"], maximum=20)
    content = normalize_avatar(file.file.read(MAX_AVATAR_BYTES + 1))
    version = hashlib.sha256(content).hexdigest()
    with store.transaction() as connection:
        user = locked_user(connection, identity)
        connection.execute(delete(store.avatars).where(store.avatars.c.user_id == user["id"]))
        connection.execute(
            insert(store.avatars).values(
                user_id=user["id"], content=content, version=version, updated_at=time.time()
            )
        )
        log(connection, user["id"], "avatar.updated")
        return {
            "user": store.public_user(connection, user),
            "csrf_token": identity["session"]["csrf"],
        }


@router.delete("/avatar")
def remove_avatar(request: Request):
    identity = current(request)
    with store.transaction() as connection:
        user = locked_user(connection, identity)
        connection.execute(delete(store.avatars).where(store.avatars.c.user_id == user["id"]))
        log(connection, user["id"], "avatar.removed")
        return {
            "user": store.public_user(connection, user),
            "csrf_token": identity["session"]["csrf"],
        }


@router.get("/avatars/{user_id}/{version}")
def read_avatar(user_id: str, version: str, request: Request):
    identity = current(request)
    if user_id != identity["user"]["id"] and identity["user"]["role"] != "superadmin":
        raise HTTPException(403, "无权查看该用户头像")
    with store.engine().connect() as connection:
        avatar = store.row(
            connection,
            store.avatars,
            (store.avatars.c.user_id == user_id) & (store.avatars.c.version == version),
        )
    if not avatar:
        raise HTTPException(404, "头像不存在")
    return Response(avatar["content"], media_type="image/jpeg")


@router.post("/reauthenticate")
def reauthenticate(payload: Reauthenticate, request: Request):
    identity = current(request)
    rate(request, "reauth", identity["user"]["id"])
    with store.transaction() as connection:
        user = locked_user(connection, identity)
        if payload.method == "totp":
            valid = bool(user["totp_secret"]) and verify_factor(connection, user, payload.code)
        else:
            # Keep the existing password + second factor path for older clients.
            valid = (
                bool(payload.password)
                and verify_password(user, payload.password)
                and verify_factor(connection, user, payload.code)
            )
        if not valid:
            raise HTTPException(400, "密码或验证码无效；已使用的验证码不能重复使用")
        log(connection, user["id"], f"reauth.{payload.method}")
        return issue_verification(connection, identity, payload.operation)


@router.post("/password")
def change_password(payload: ChangePassword, request: Request, response: Response):
    identity = current(request)
    rate(request, "password", identity["user"]["id"])
    new_hash = password_hasher.hash(payload.new_password)
    with store.transaction() as connection:
        user = locked_user(connection, identity)
        consume_verification(connection, identity, request)
        connection.execute(
            update(store.users).where(store.users.c.id == user["id"]).values(password_hash=new_hash)
        )
        connection.execute(delete(store.sessions).where(store.sessions.c.user_id == user["id"]))
        connection.execute(delete(store.challenges).where(store.challenges.c.user_id == user["id"]))
        log(connection, user["id"], "password.changed")
        return new_session(connection, user, request, response)


@router.get("/security")
def security(request: Request):
    identity = current(request)
    with store.engine().connect() as connection:
        user = store.row(connection, store.users, store.users.c.id == identity["user"]["id"])
        items = (
            connection.execute(
                select(
                    store.credentials.c.id, store.credentials.c.name, store.credentials.c.created_at
                ).where(store.credentials.c.user_id == user["id"])
            )
            .mappings()
            .all()
        )
        return {
            "passkeys": [dict(item) for item in items],
            "totp_enabled": bool(user["totp_secret"]),
            "recovery_codes_remaining": len(json.loads(user["recovery_hashes"] or "[]")),
        }


@router.post("/2fa/setup")
def setup_factor(request: Request):
    identity = current(request)
    rate(request, "setup-factor", identity["user"]["id"])
    with store.transaction() as connection:
        user = locked_user(connection, identity)
        consume_verification(connection, identity, request)
        if user["totp_secret"]:
            raise HTTPException(409, "已绑定验证器")
        secret = pyotp.random_base32()
        connection.execute(
            delete(store.challenges).where(
                (store.challenges.c.user_id == user["id"])
                & (store.challenges.c.kind == "totp-setup")
            )
        )
        token = store.challenge(
            connection,
            "totp-setup",
            user["id"],
            identity["session"]["id"],
            {
                "secret": store.cipher().encrypt(secret.encode()).decode(),
                "operation_verified": True,
            },
        )
        uri = pyotp.TOTP(secret).provisioning_uri(name=user["username"], issuer_name="WCM")
        buffer = io.BytesIO()
        qrcode.make(uri, image_factory=qrcode.image.svg.SvgPathImage).save(buffer)
        return {
            "challenge_id": token,
            "secret": secret,
            "uri": uri,
            "qr_code": "data:image/svg+xml;base64," + base64.b64encode(buffer.getvalue()).decode(),
        }


def recovery_codes(connection, user_id):
    codes = [secrets.token_hex(8).upper() for _ in range(10)]
    connection.execute(
        update(store.users)
        .where(store.users.c.id == user_id)
        .values(recovery_hashes=json.dumps([store.digest(code) for code in codes]))
    )
    return codes


@router.post("/2fa/confirm")
def confirm_factor(payload: VerifyCode, request: Request, response: Response):
    identity = current(request)
    rate(request, "confirm-factor", identity["user"]["id"])
    with store.transaction() as connection:
        user = locked_user(connection, identity)
        item = store.consume(
            connection, payload.challenge_id, "totp-setup", identity["session"]["id"]
        )
        if not item["payload"].get("operation_verified"):
            raise HTTPException(400, "绑定已失效，请重新验证身份并开始绑定")
        if user["totp_secret"]:
            raise HTTPException(409, "已绑定验证器")
        user["totp_secret"] = item["payload"]["secret"]
        user["totp_last_step"] = -1
        if not verify_factor(connection, user, payload.code, recovery=False):
            raise HTTPException(400, "验证码错误，请重新开始绑定")
        connection.execute(
            update(store.users)
            .where(store.users.c.id == user["id"])
            .values(totp_secret=user["totp_secret"])
        )
        codes = recovery_codes(connection, user["id"])
        connection.execute(delete(store.sessions).where(store.sessions.c.user_id == user["id"]))
        connection.execute(delete(store.challenges).where(store.challenges.c.user_id == user["id"]))
        log(connection, user["id"], "2fa.enabled")
        return new_session(connection, user, request, response) | {"recovery_codes": codes}


@router.post("/2fa/disable")
def disable_factor(payload: SensitiveProof, request: Request, response: Response):
    identity = current(request)
    rate(request, "disable-factor", identity["user"]["id"])
    with store.transaction() as connection:
        user = locked_user(connection, identity)
        if not user["totp_secret"]:
            raise HTTPException(409, "尚未绑定验证器")
        consume_verification(connection, identity, request)
        connection.execute(
            update(store.users)
            .where(store.users.c.id == user["id"])
            .values(totp_secret=None, totp_last_step=-1, recovery_hashes="[]")
        )
        connection.execute(delete(store.sessions).where(store.sessions.c.user_id == user["id"]))
        connection.execute(delete(store.challenges).where(store.challenges.c.user_id == user["id"]))
        user["totp_secret"] = None
        log(connection, user["id"], "2fa.disabled")
        return new_session(connection, user, request, response)


@router.post("/2fa/recovery-codes")
def regenerate_codes(payload: SensitiveProof, request: Request):
    identity = current(request)
    rate(request, "recovery-codes", identity["user"]["id"])
    with store.transaction() as connection:
        user = locked_user(connection, identity)
        if not user["totp_secret"]:
            raise HTTPException(409, "尚未绑定验证器")
        consume_verification(connection, identity, request)
        log(connection, user["id"], "recovery-codes.rotated")
        return {"recovery_codes": recovery_codes(connection, user["id"])}


@router.post("/passkeys/register/options")
def registration_options(request: Request):
    identity = current(request)
    origin = passkey_origin(request)
    rate(request, "passkey-register", identity["user"]["id"])
    with store.transaction() as connection:
        user = locked_user(connection, identity)
        consume_verification(connection, identity, request)
        existing = (
            connection.execute(
                select(store.credentials).where(store.credentials.c.user_id == user["id"])
            )
            .mappings()
            .all()
        )
        if len(existing) >= 10:
            raise HTTPException(400, "最多绑定 10 个 Passkey")
        options = generate_registration_options(
            rp_id=store.config.rp_id,
            rp_name="WCM",
            user_id=user["id"].encode(),
            user_name=user["username"],
            user_display_name=user["display_name"],
            exclude_credentials=[
                PublicKeyCredentialDescriptor(id=base64url_to_bytes(item["credential_id"]))
                for item in existing
            ],
            authenticator_selection=AuthenticatorSelectionCriteria(
                resident_key=ResidentKeyRequirement.REQUIRED,
                user_verification=UserVerificationRequirement.REQUIRED,
            ),
        )
        token = store.challenge(
            connection,
            "passkey-register",
            user["id"],
            identity["session"]["id"],
            {
                "challenge": bytes_to_base64url(options.challenge),
                "origin": origin,
                "operation_verified": True,
            },
        )
        return {"challenge_id": token, "options": json.loads(options_to_json(options))}


@router.post("/passkeys/register/verify")
def registration_verify(payload: PasskeyResponse, request: Request):
    identity = current(request)
    origin = passkey_origin(request)
    rate(request, "passkey-verify", identity["user"]["id"])
    with store.transaction() as connection:
        user = locked_user(connection, identity)
        item = store.consume(
            connection, payload.challenge_id, "passkey-register", identity["session"]["id"]
        )
        if not item["payload"].get("operation_verified"):
            raise HTTPException(400, "绑定已失效，请重新验证身份并开始绑定")
        if origin != item["payload"]["origin"]:
            raise HTTPException(400, "验证来源不匹配")
        try:
            verified = verify_registration_response(
                credential=payload.credential,
                expected_challenge=base64url_to_bytes(item["payload"]["challenge"]),
                expected_rp_id=store.config.rp_id,
                expected_origin=origin,
                require_user_verification=True,
            )
        except (WebAuthnException, ValueError, TypeError, KeyError):
            raise HTTPException(400, "Passkey 验证失败，请重新绑定")
        credential_id = bytes_to_base64url(verified.credential_id)
        key_id = store.digest(credential_id)
        if store.row(connection, store.credentials, store.credentials.c.id == key_id):
            raise HTTPException(409, "该 Passkey 已绑定")
        count = connection.execute(
            select(func.count())
            .select_from(store.credentials)
            .where(store.credentials.c.user_id == user["id"])
        ).scalar()
        if count >= 10:
            raise HTTPException(400, "最多绑定 10 个 Passkey")
        connection.execute(
            insert(store.credentials).values(
                id=key_id,
                user_id=user["id"],
                credential_id=credential_id,
                public_key=bytes_to_base64url(verified.credential_public_key),
                sign_count=verified.sign_count,
                name=payload.name,
                created_at=time.time(),
            )
        )
        log(connection, user["id"], "passkey.added", key_id)
        return {"ok": True}


@router.post("/passkeys/reauthenticate/options")
def reauthentication_options(payload: VerificationOperation, request: Request):
    identity = current(request)
    origin = passkey_origin(request)
    rate(request, "reauth", identity["user"]["id"])
    with store.transaction() as connection:
        user = locked_user(connection, identity)
        keys = (
            connection.execute(
                select(store.credentials.c.credential_id).where(
                    store.credentials.c.user_id == user["id"]
                )
            )
            .scalars()
            .all()
        )
        if not keys:
            raise HTTPException(400, "当前账户未绑定 Passkey")
        options = generate_authentication_options(
            rp_id=store.config.rp_id,
            allow_credentials=[
                PublicKeyCredentialDescriptor(id=base64url_to_bytes(key)) for key in keys
            ],
            user_verification=UserVerificationRequirement.REQUIRED,
        )
        token = store.challenge(
            connection,
            "passkey-reauth",
            user["id"],
            identity["session"]["id"],
            {
                "challenge": bytes_to_base64url(options.challenge),
                "origin": origin,
                "operation": payload.operation,
            },
        )
        return {"challenge_id": token, "options": json.loads(options_to_json(options))}


@router.post("/passkeys/reauthenticate/verify")
def reauthentication_verify(payload: PasskeyResponse, request: Request):
    identity = current(request)
    origin = passkey_origin(request)
    rate(request, "reauth", identity["user"]["id"])
    with store.transaction() as connection:
        user = locked_user(connection, identity)
        item = store.consume(
            connection, payload.challenge_id, "passkey-reauth", identity["session"]["id"]
        )
        if not item["payload"].get("operation"):
            raise HTTPException(400, "验证已失效，请重新开始本次操作")
        credential_id = payload.credential.get("id")
        if not isinstance(credential_id, str) or len(credential_id) > 4096:
            raise HTTPException(400, "Passkey 验证失败")
        key = store.row(
            connection, store.credentials, store.credentials.c.id == store.digest(credential_id)
        )
        if (
            not key
            or key["user_id"] != user["id"]
            or item["user_id"] != user["id"]
            or origin != item["payload"]["origin"]
        ):
            raise HTTPException(400, "请使用当前账户绑定的 Passkey")
        try:
            verified = verify_authentication_response(
                credential=payload.credential,
                expected_challenge=base64url_to_bytes(item["payload"]["challenge"]),
                expected_rp_id=store.config.rp_id,
                expected_origin=origin,
                credential_public_key=base64url_to_bytes(key["public_key"]),
                credential_current_sign_count=key["sign_count"],
                require_user_verification=True,
            )
            handle = payload.credential.get("response", {}).get("userHandle")
            # allowCredentials already binds the assertion to this account; a supplied handle must match.
            if handle is not None and base64url_to_bytes(handle) != user["id"].encode():
                raise ValueError("User handle mismatch")
        except (WebAuthnException, ValueError, TypeError, KeyError):
            raise HTTPException(400, "Passkey 验证失败，请重试")
        connection.execute(
            update(store.credentials)
            .where(store.credentials.c.id == key["id"])
            .values(sign_count=verified.new_sign_count)
        )
        log(connection, user["id"], "reauth.passkey")
        return issue_verification(connection, identity, item["payload"]["operation"])


@router.post("/passkeys/login/options")
def authentication_options(request: Request):
    origin = passkey_origin(request)
    rate(request, "passkey-login", maximum=30)
    options = generate_authentication_options(
        rp_id=store.config.rp_id, user_verification=UserVerificationRequirement.REQUIRED
    )
    with store.transaction() as connection:
        token = store.challenge(
            connection,
            "passkey-login",
            payload={"challenge": bytes_to_base64url(options.challenge), "origin": origin},
        )
    return {"challenge_id": token, "options": json.loads(options_to_json(options))}


@router.post("/passkeys/login/verify")
def authentication_verify(payload: PasskeyResponse, request: Request, response: Response):
    origin = passkey_origin(request)
    rate(request, "passkey-login-verify", maximum=30)
    with store.transaction() as connection:
        item = store.consume(connection, payload.challenge_id, "passkey-login")
        credential_id = payload.credential.get("id")
        if not isinstance(credential_id, str) or len(credential_id) > 4096:
            raise HTTPException(400, "Passkey 验证失败")
        key = store.row(
            connection, store.credentials, store.credentials.c.id == store.digest(credential_id)
        )
        user = (
            store.row(connection, store.users, store.users.c.id == key["user_id"]) if key else None
        )
        if not key or not user or not user["active"] or origin != item["payload"]["origin"]:
            raise HTTPException(400, "Passkey 验证失败")
        try:
            verified = verify_authentication_response(
                credential=payload.credential,
                expected_challenge=base64url_to_bytes(item["payload"]["challenge"]),
                expected_rp_id=store.config.rp_id,
                expected_origin=origin,
                credential_public_key=base64url_to_bytes(key["public_key"]),
                credential_current_sign_count=key["sign_count"],
                require_user_verification=True,
            )
            handle = payload.credential.get("response", {}).get("userHandle")
            if not handle or base64url_to_bytes(handle) != user["id"].encode():
                raise ValueError("User handle mismatch")
        except (WebAuthnException, ValueError, TypeError, KeyError):
            raise HTTPException(400, "Passkey 验证失败，请重新登录")
        connection.execute(
            update(store.credentials)
            .where(store.credentials.c.id == key["id"])
            .values(sign_count=verified.new_sign_count)
        )
        # A verified Passkey already proves possession and biometric/PIN verification.
        return finish_login(connection, user, request, response)


@router.delete("/passkeys/{key_id}")
def delete_passkey(key_id: str, request: Request):
    identity = current(request)
    with store.transaction() as connection:
        user = locked_user(connection, identity)
        consume_verification(connection, identity, request)
        result = connection.execute(
            delete(store.credentials).where(
                (store.credentials.c.id == key_id) & (store.credentials.c.user_id == user["id"])
            )
        )
        if not result.rowcount:
            raise HTTPException(404, "Passkey 不存在")
        connection.execute(
            delete(store.sessions).where(
                (store.sessions.c.user_id == user["id"])
                & (store.sessions.c.id != identity["session"]["id"])
            )
        )
        log(connection, user["id"], "passkey.removed", key_id)
    return {"ok": True}


@router.get("/users")
def list_users(request: Request, page: int = 1, query: str = ""):
    current(request, superadmin=True)
    page = max(1, page)
    with store.engine().connect() as connection:
        condition = store.users.c.username.contains(query[:64], autoescape=True)
        total = connection.execute(
            select(func.count()).select_from(store.users).where(condition)
        ).scalar()
        rows = connection.execute(
            select(store.users)
            .where(condition)
            .order_by(store.users.c.created_at)
            .offset((page - 1) * 30)
            .limit(30)
        ).mappings()
        return {
            "items": [store.public_user(connection, dict(user)) for user in rows],
            "total": total,
            "page": page,
        }


def admin_target(connection, identity, user_id):
    actor = locked_user(connection, identity)
    if actor["role"] != "superadmin":
        raise HTTPException(403, "仅超级管理员可操作")
    user = store.row(connection, store.users, store.users.c.id == user_id)
    if not user:
        raise HTTPException(404, "用户不存在")
    if user["role"] == "superadmin":
        raise HTTPException(400, "不能降级或停用超级管理员")
    return user


@router.put("/users/{user_id}/role")
def update_role(user_id: str, payload: RoleChange, request: Request):
    identity = current(request, superadmin=True)
    with store.transaction() as connection:
        admin_target(connection, identity, user_id)
        consume_verification(connection, identity, request)
        connection.execute(
            update(store.users).where(store.users.c.id == user_id).values(role=payload.role)
        )
        connection.execute(delete(store.sessions).where(store.sessions.c.user_id == user_id))
        connection.execute(delete(store.challenges).where(store.challenges.c.user_id == user_id))
        log(connection, identity["user"]["id"], "role." + payload.role, user_id)
    return {"ok": True}


@router.put("/users/{user_id}/active")
def update_active(user_id: str, payload: ActiveChange, request: Request):
    identity = current(request, superadmin=True)
    with store.transaction() as connection:
        admin_target(connection, identity, user_id)
        consume_verification(connection, identity, request)
        connection.execute(
            update(store.users).where(store.users.c.id == user_id).values(active=payload.active)
        )
        connection.execute(delete(store.sessions).where(store.sessions.c.user_id == user_id))
        connection.execute(delete(store.challenges).where(store.challenges.c.user_id == user_id))
        log(
            connection,
            identity["user"]["id"],
            "user.enabled" if payload.active else "user.disabled",
            user_id,
        )
    return {"ok": True}


@router.get("/roles")
def list_roles(request: Request):
    current(request, superadmin=True)
    with store.engine().connect() as connection:
        return {
            "permissions": store.PERMISSIONS,
            "admin_only": sorted(store.ADMIN_ONLY),
            "roles": {role: store.permissions(connection, role) for role in store.DEFAULTS},
        }


@router.put("/roles/{role}")
def update_permissions(role: str, payload: PermissionChange, request: Request):
    identity = current(request, superadmin=True)
    allowed = set(payload.permissions)
    if (
        role not in {"admin", "user"}
        or allowed - store.PERMISSIONS.keys()
        or (role == "user" and allowed & store.ADMIN_ONLY)
    ):
        raise HTTPException(400, "角色或权限无效，普通用户不能获授系统管理权限")
    if "people.write" in allowed:
        allowed.add("people.read")
    if allowed & {"review.run", "review.manage"}:
        allowed.add("review.read")
    with store.transaction() as connection:
        actor = locked_user(connection, identity)
        if actor["role"] != "superadmin":
            raise HTTPException(403, "仅超级管理员可操作")
        consume_verification(connection, identity, request)
        connection.execute(
            update(store.policies)
            .where(store.policies.c.role == role)
            .values(permissions=json.dumps(sorted(allowed)))
        )
        log(connection, actor["id"], "permissions.updated", role)
    return {"ok": True}
