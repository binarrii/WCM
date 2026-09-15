"""Exercise the real auth guard, database and cryptographic verifiers end to end."""

import hashlib
import json
import struct
import time
from concurrent.futures import ThreadPoolExecutor

import cbor2
import pyotp
import pytest
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import WebSocket
from fastapi.testclient import TestClient
from sqlalchemy import delete, select, update
from starlette.websockets import WebSocketDisconnect
from webauthn.helpers import bytes_to_base64url as b64

from api import auth_store as store
from api.main import create_app

PASSWORD = "a long unique test password 2026"
ORIGIN = "http://localhost:5173"


@pytest.fixture
def client(tmp_path, monkeypatch):
    store.engine.cache_clear()
    store.cipher.cache_clear()
    monkeypatch.setattr(store.config, "database_url", f"sqlite:///{tmp_path}/auth.sqlite3")
    monkeypatch.setattr(store.config, "secret_key", Fernet.generate_key().decode())
    monkeypatch.setattr(store.config, "origins", [ORIGIN])
    monkeypatch.setattr(store.config, "rp_id", "localhost")
    monkeypatch.setattr(store.settings, "cluster_enabled", False)
    store.initialize()
    app = create_app()

    @app.websocket("/api/v1/review_tasks/auth-probe")
    async def probe(websocket: WebSocket):
        await websocket.accept()
        while True:
            message = await websocket.receive_text()
            await websocket.send_text(message)

    result = TestClient(app, base_url=ORIGIN, headers={"Origin": ORIGIN, "X-WCM-Client": "web"})
    yield result
    result.close()
    store.engine().dispose()
    store.engine.cache_clear()
    store.cipher.cache_clear()


def sibling(client):
    return TestClient(
        client.app, base_url=ORIGIN, headers={"Origin": ORIGIN, "X-WCM-Client": "web"}
    )


def accept(client, response):
    assert response.status_code in {200, 201}, response.text
    payload = response.json()
    if "csrf_token" in payload:
        client.headers["X-CSRF-Token"] = payload["csrf_token"]
    return payload


def signup(client, name="root"):
    return accept(
        client,
        client.post(
            "/api/v1/auth/register",
            json={"username": name, "display_name": name, "password": PASSWORD},
        ),
    )


def password_login(client, name="root"):
    return accept(
        client, client.post("/api/v1/auth/login", json={"username": name, "password": PASSWORD})
    )


def setup_totp(client):
    pending = accept(client, client.post("/api/v1/auth/2fa/setup"))
    code = pyotp.TOTP(pending["secret"]).now()
    payload = accept(
        client,
        client.post(
            "/api/v1/auth/2fa/confirm", json={"challenge_id": pending["challenge_id"], "code": code}
        ),
    )
    return pending["secret"], payload["recovery_codes"]


def test_registration_roles_hashes_cookie_and_logout(client):
    assert client.get("/api/v1/auth/config").json()["first_registration"] is True
    response = client.post(
        "/api/v1/auth/register",
        json={"username": "Root", "display_name": "首位用户", "password": PASSWORD},
    )
    root = accept(client, response)
    assert root["user"]["role"] == "superadmin"
    assert (
        "HttpOnly" in response.headers["set-cookie"]
        and "SameSite=lax" in response.headers["set-cookie"]
    )
    assert "no-store" in response.headers["cache-control"]
    assert "password" not in response.text
    user_client = sibling(client)
    assert signup(user_client, "member")["user"]["role"] == "user"
    duplicate = user_client.post(
        "/api/v1/auth/register",
        json={"username": "ROOT", "display_name": "duplicate", "password": PASSWORD},
    )
    assert duplicate.status_code == 409
    with store.engine().connect() as connection:
        assert all(
            item.password_hash.startswith("$argon2id$")
            for item in connection.execute(select(store.users))
        )
        assert not any(
            item.id == client.cookies[store.COOKIE]
            for item in connection.execute(select(store.sessions))
        )
    token = client.cookies[store.COOKIE]
    assert client.post("/api/v1/auth/logout").status_code == 200
    client.cookies.set(store.COOKIE, token)
    assert client.get("/api/v1/auth/me").status_code == 401
    assert password_login(client)["user"]["id"] == root["user"]["id"]


def test_first_registration_is_atomic(client):
    def register(index):
        other = sibling(client)
        try:
            return signup(other, f"member{index}")["user"]["role"]
        finally:
            other.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        roles = list(pool.map(register, range(4)))
    assert roles.count("superadmin") == 1
    assert roles.count("user") == 3


@pytest.mark.parametrize(
    "path,method",
    [
        ("/api/v1/face_records", "GET"),
        ("/api/v1/parameters", "GET"),
        ("/api/v1/insightface/replication/sync", "POST"),
        ("/images/private.jpg", "GET"),
        ("/api/v1/review_tasks", "GET"),
        ("/openapi.json", "GET"),
    ],
)
def test_all_business_and_image_routes_require_login(client, path, method):
    assert client.request(method, path, json={}).status_code == 401


def test_registration_cannot_inject_role_or_weak_password(client):
    body = {"username": "attacker", "display_name": "A", "password": PASSWORD}
    assert (
        client.post("/api/v1/auth/register", json={**body, "role": "superadmin"}).status_code == 422
    )
    assert (
        client.post("/api/v1/auth/register", json={**body, "password": "short"}).status_code == 422
    )
    assert (
        client.post("/api/v1/auth/register", json={**body, "display_name": " "}).status_code == 422
    )
    assert (
        client.post("/api/v1/auth/register", json={**body, "password": "x" * 66000}).status_code
        == 413
    )


def test_readonly_user_can_subscribe_but_not_submit_on_shared_socket(client, monkeypatch):
    from api import routes

    signup(client)
    other = sibling(client)
    signup(other, "viewer")
    assert (
        client.put("/api/v1/auth/roles/user", json={"permissions": ["review.read"]}).status_code
        == 200
    )

    async def subscription(socket, payload):
        await socket.send_json({"type": "snapshot"})

    monkeypatch.setattr(routes, "stream_review_tasks", subscription)
    path = "ws://localhost:5173/api/v1/ws/analyze_media"
    with other.websocket_connect(path, headers={"Origin": ORIGIN}) as socket:
        socket.send_json({"type": "subscribe", "task_ids": []})
        assert socket.receive_json()["type"] == "snapshot"
    with other.websocket_connect(path, headers={"Origin": ORIGIN}) as socket:
        socket.send_json({"url": "https://example.com/video.mp4"})
        with pytest.raises(WebSocketDisconnect) as error:
            socket.receive_json()
        assert error.value.code == 4403


def test_role_permissions_and_revocation(client):
    root = signup(client)["user"]
    user_client = sibling(client)
    user = signup(user_client, "member")["user"]
    for path, method in [
        ("/api/v1/parameters", "GET"),
        ("/api/v1/insightface/replication", "GET"),
        ("/api/v1/face_records", "POST"),
        ("/api/v1/review_tasks/example", "DELETE"),
        ("/api/v1/auth/users", "GET"),
    ]:
        assert user_client.request(method, path, json={}).status_code == 403
    assert (
        user_client.put(f"/api/v1/auth/users/{user['id']}/role", json={"role": "admin"}).status_code
        == 403
    )
    assert (
        client.put(f"/api/v1/auth/users/{user['id']}/role", json={"role": "admin"}).status_code
        == 200
    )
    assert user_client.get("/api/v1/auth/me").status_code == 401
    assert password_login(user_client, "member")["user"]["role"] == "admin"
    # Ordinary admins cannot grant roles, edit role policies, or disable users.
    assert (
        user_client.put(
            f"/api/v1/auth/users/{root['id']}/active", json={"active": False}
        ).status_code
        == 403
    )
    assert user_client.put("/api/v1/auth/roles/user", json={"permissions": []}).status_code == 403
    assert (
        client.put(f"/api/v1/auth/users/{root['id']}/role", json={"role": "user"}).status_code
        == 400
    )
    assert (
        client.put(f"/api/v1/auth/users/{root['id']}/active", json={"active": False}).status_code
        == 400
    )
    assert (
        client.put("/api/v1/auth/roles/user", json={"permissions": ["system.manage"]}).status_code
        == 400
    )
    assert client.put("/api/v1/auth/roles/admin", json={"permissions": []}).status_code == 200
    assert user_client.get("/api/v1/parameters").status_code == 403
    assert (
        client.put(f"/api/v1/auth/users/{user['id']}/active", json={"active": False}).status_code
        == 200
    )
    assert user_client.get("/api/v1/auth/me").status_code == 401
    assert (
        user_client.post(
            "/api/v1/auth/login", json={"username": "member", "password": PASSWORD}
        ).status_code
        == 401
    )


def test_permission_dependencies_and_protected_system_permissions(client):
    signup(client)
    assert (
        client.put(
            "/api/v1/auth/roles/user", json={"permissions": ["review.run", "people.write"]}
        ).status_code
        == 200
    )
    roles = client.get("/api/v1/auth/roles").json()["roles"]
    assert set(roles["user"]) == {"review.run", "review.read", "people.write", "people.read"}
    assert client.put("/api/v1/auth/roles/superadmin", json={"permissions": []}).status_code == 400
    assert (
        client.put("/api/v1/auth/roles/admin", json={"permissions": ["users.manage"]}).status_code
        == 400
    )


def test_csrf_cors_and_stale_sessions(client):
    signup(client)
    assert client.post("/api/v1/auth/2fa/setup", headers={"X-CSRF-Token": "bad"}).status_code == 403
    assert (
        client.post("/api/v1/auth/2fa/setup", headers={"Origin": "https://evil.test"}).status_code
        == 403
    )
    assert (
        client.post(
            "/api/v1/auth/login",
            json={"username": "root", "password": PASSWORD},
            headers={"Origin": "https://evil.test"},
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/v1/auth/login",
            json={"username": "root", "password": PASSWORD},
            headers={"X-WCM-Client": ""},
        ).status_code
        == 403
    )
    with store.transaction() as connection:
        connection.execute(update(store.sessions).values(verified_at=time.time() - 301))
    assert client.post("/api/v1/auth/2fa/setup").status_code == 403
    accept(client, client.post("/api/v1/auth/reauthenticate", json={"password": PASSWORD}))
    assert client.post("/api/v1/auth/2fa/setup").status_code == 200
    with store.transaction() as connection:
        connection.execute(update(store.sessions).values(expires_at=time.time() - 1))
    assert client.get("/api/v1/auth/me").status_code == 401


def test_totp_pending_confirmation_recovery_and_replay(client):
    signup(client)
    secret, codes = setup_totp(client)
    with store.engine().connect() as connection:
        user = connection.execute(select(store.users)).mappings().one()
        assert secret not in user["totp_secret"]
        assert codes[0] not in user["recovery_hashes"]
    stranger = sibling(client)
    first = password_login(stranger)
    assert first["mfa_required"] is True
    assert stranger.get("/api/v1/auth/me").status_code == 401
    # Code used to enroll cannot be replayed immediately to log in.
    assert (
        stranger.post(
            "/api/v1/auth/login/2fa",
            json={"challenge_id": first["challenge_id"], "code": pyotp.TOTP(secret).now()},
        ).status_code
        == 401
    )
    next_login = password_login(stranger)
    accept(
        stranger,
        stranger.post(
            "/api/v1/auth/login/2fa",
            json={"challenge_id": next_login["challenge_id"], "code": codes[0]},
        ),
    )
    assert stranger.get("/api/v1/auth/security").json()["recovery_codes_remaining"] == 9
    assert (
        stranger.post(
            "/api/v1/auth/login/2fa",
            json={"challenge_id": next_login["challenge_id"], "code": codes[1]},
        ).status_code
        == 400
    )
    third = password_login(sibling(client))
    assert (
        stranger.post(
            "/api/v1/auth/login/2fa", json={"challenge_id": third["challenge_id"], "code": codes[0]}
        ).status_code
        == 401
    )
    # New TOTP step succeeds, but only once across sessions/instances.
    with store.transaction() as connection:
        connection.execute(update(store.users).values(totp_last_step=int(time.time() // 30) - 1))
    fourth = password_login(stranger)
    accept(
        stranger,
        stranger.post(
            "/api/v1/auth/login/2fa",
            json={"challenge_id": fourth["challenge_id"], "code": pyotp.TOTP(secret).now()},
        ),
    )
    accept(
        stranger,
        stranger.post("/api/v1/auth/2fa/disable", json={"password": PASSWORD, "code": codes[1]}),
    )
    assert not stranger.get("/api/v1/auth/me").json()["user"]["totp_enabled"]
    assert client.get("/api/v1/auth/me").status_code == 401


def test_setup_challenge_is_session_bound_and_cannot_overwrite_mfa(client):
    signup(client)
    pending = client.post("/api/v1/auth/2fa/setup").json()
    second = sibling(client)
    password_login(second)
    response = second.post(
        "/api/v1/auth/2fa/confirm",
        json={"challenge_id": pending["challenge_id"], "code": pyotp.TOTP(pending["secret"]).now()},
    )
    assert response.status_code == 400
    assert not second.get("/api/v1/auth/security").json()["totp_enabled"]
    setup_totp(second)
    assert client.get("/api/v1/auth/me").status_code == 401
    assert second.post("/api/v1/auth/2fa/setup").status_code == 409


def test_password_change_invalidates_sessions_and_pending_logins(client):
    signup(client)
    _, codes = setup_totp(client)
    second = sibling(client)
    pending = password_login(second)
    accept(
        client,
        client.post(
            "/api/v1/auth/password",
            json={"password": PASSWORD, "code": codes[0], "new_password": PASSWORD + "changed"},
        ),
    )
    assert (
        second.post(
            "/api/v1/auth/login/2fa",
            json={"challenge_id": pending["challenge_id"], "code": codes[1]},
        ).status_code
        == 400
    )
    assert (
        second.post(
            "/api/v1/auth/login", json={"username": "root", "password": PASSWORD}
        ).status_code
        == 401
    )


def test_login_rate_limit_shared_across_clients(client):
    signup(client)
    for _ in range(10):
        assert (
            sibling(client)
            .post("/api/v1/auth/login", json={"username": "root", "password": "incorrect"})
            .status_code
            == 401
        )
    assert (
        sibling(client)
        .post("/api/v1/auth/login", json={"username": "root", "password": PASSWORD})
        .status_code
        == 429
    )


def test_websocket_auth_origin_and_live_revocation(client):
    for headers in ({"Origin": ORIGIN}, {"Origin": "https://evil.test"}):
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                "ws://localhost:5173/api/v1/review_tasks/auth-probe", headers=headers
            ):
                pass
    signup(client)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            "ws://localhost:5173/api/v1/review_tasks/auth-probe",
            headers={"Origin": "https://evil.test"},
        ):
            pass
    with client.websocket_connect(
        "ws://localhost:5173/api/v1/review_tasks/auth-probe", headers={"Origin": ORIGIN}
    ) as socket:
        socket.send_text("ping")
        assert socket.receive_text() == "ping"
        client.post("/api/v1/auth/logout")
        socket.send_text("cannot continue")
        with pytest.raises(WebSocketDisconnect):
            socket.receive_text()


class Authenticator:
    """A real ES256 test authenticator (no mocked WebAuthn verifier)."""

    def __init__(self):
        self.key = ec.generate_private_key(ec.SECP256R1())
        self.id = b"test-credential-" + self.key.public_key().public_numbers().x.to_bytes(32, "big")

    def client_data(self, options, kind, origin=ORIGIN):
        return json.dumps(
            {
                "type": kind,
                "challenge": options["challenge"],
                "origin": origin,
                "crossOrigin": False,
            }
        ).encode()

    def register(self, options, origin=ORIGIN, uv=True):
        public = self.key.public_key().public_numbers()
        cose = {
            1: 2,
            3: -7,
            -1: 1,
            -2: public.x.to_bytes(32, "big"),
            -3: public.y.to_bytes(32, "big"),
        }
        auth_data = (
            hashlib.sha256(b"localhost").digest()
            + bytes([0x45 if uv else 0x41])
            + struct.pack(">I", 0)
            + b"\x00" * 16
            + struct.pack(">H", len(self.id))
            + self.id
            + cbor2.dumps(cose)
        )
        return {
            "id": b64(self.id),
            "rawId": b64(self.id),
            "type": "public-key",
            "response": {
                "clientDataJSON": b64(self.client_data(options, "webauthn.create", origin)),
                "attestationObject": b64(
                    cbor2.dumps({"fmt": "none", "attStmt": {}, "authData": auth_data})
                ),
                "transports": ["internal"],
            },
        }

    def authenticate(self, options, user_id, count=1, origin=ORIGIN, uv=True, rp="localhost"):
        client_data = self.client_data(options, "webauthn.get", origin)
        auth_data = (
            hashlib.sha256(rp.encode()).digest()
            + bytes([5 if uv else 1])
            + struct.pack(">I", count)
        )
        signature = self.key.sign(
            auth_data + hashlib.sha256(client_data).digest(), ec.ECDSA(hashes.SHA256())
        )
        return {
            "id": b64(self.id),
            "rawId": b64(self.id),
            "type": "public-key",
            "response": {
                "clientDataJSON": b64(client_data),
                "authenticatorData": b64(auth_data),
                "signature": b64(signature),
                "userHandle": b64(user_id.encode()),
            },
        }


def bind(client, authenticator):
    pending = accept(client, client.post("/api/v1/auth/passkeys/register/options"))
    return accept(
        client,
        client.post(
            "/api/v1/auth/passkeys/register/verify",
            json={
                "challenge_id": pending["challenge_id"],
                "credential": authenticator.register(pending["options"]),
                "name": "Test device",
            },
        ),
    )


def test_real_passkey_registration_login_counter_and_removal(client):
    user = signup(client)["user"]
    authenticator = Authenticator()
    bind(client, authenticator)
    second = sibling(client)
    pending = second.post("/api/v1/auth/passkeys/login/options").json()
    assertion = {
        "challenge_id": pending["challenge_id"],
        "credential": authenticator.authenticate(pending["options"], user["id"]),
    }
    assert (
        accept(second, second.post("/api/v1/auth/passkeys/login/verify", json=assertion))["user"][
            "id"
        ]
        == user["id"]
    )
    assert second.post("/api/v1/auth/passkeys/login/verify", json=assertion).status_code == 400
    pending = second.post("/api/v1/auth/passkeys/login/options").json()
    # Reusing a non-zero authenticator counter fails with a fresh challenge too.
    assert (
        second.post(
            "/api/v1/auth/passkeys/login/verify",
            json={
                "challenge_id": pending["challenge_id"],
                "credential": authenticator.authenticate(pending["options"], user["id"]),
            },
        ).status_code
        == 400
    )
    key_id = client.get("/api/v1/auth/security").json()["passkeys"][0]["id"]
    assert client.delete(f"/api/v1/auth/passkeys/{key_id}").status_code == 200
    assert second.get("/api/v1/auth/me").status_code == 401


@pytest.mark.parametrize("invalid", ["origin", "rp", "uv", "signature", "handle", "challenge"])
def test_real_passkey_rejects_invalid_assertions(client, invalid):
    user = signup(client)["user"]
    authenticator = Authenticator()
    bind(client, authenticator)
    pending = client.post("/api/v1/auth/passkeys/login/options").json()
    options = dict(pending["options"])
    if invalid == "challenge":
        options["challenge"] = b64(b"a wrong challenge")
    credential = authenticator.authenticate(
        options,
        "wrong-user" if invalid == "handle" else user["id"],
        origin="https://evil.test" if invalid == "origin" else ORIGIN,
        rp="evil.test" if invalid == "rp" else "localhost",
        uv=invalid != "uv",
    )
    if invalid == "signature":
        credential["response"]["signature"] = b64(b"invalid")
    assert (
        client.post(
            "/api/v1/auth/passkeys/login/verify",
            json={"challenge_id": pending["challenge_id"], "credential": credential},
        ).status_code
        == 400
    )


def test_passkey_requires_secure_configured_origin_and_verified_registration(client):
    signup(client)
    assert (
        client.post(
            "/api/v1/auth/passkeys/login/options", headers={"Origin": "http://10.252.25.251:8000"}
        ).status_code
        == 403
    )
    pending = client.post("/api/v1/auth/passkeys/register/options").json()
    bad = Authenticator().register(pending["options"], uv=False)
    assert (
        client.post(
            "/api/v1/auth/passkeys/register/verify",
            json={"challenge_id": pending["challenge_id"], "credential": bad},
        ).status_code
        == 400
    )


def test_user_cannot_remove_another_users_passkey(client):
    signup(client)
    bind(client, Authenticator())
    key_id = client.get("/api/v1/auth/security").json()["passkeys"][0]["id"]
    other = sibling(client)
    signup(other, "member")
    assert other.delete(f"/api/v1/auth/passkeys/{key_id}").status_code == 404
    assert len(client.get("/api/v1/auth/security").json()["passkeys"]) == 1


def expire_verification(client):
    with store.transaction() as connection:
        connection.execute(
            update(store.sessions)
            .where(store.sessions.c.id == store.digest(client.cookies.get(store.COOKIE)))
            .values(verified_at=time.time() - 301)
        )


def test_totp_reauthentication_requires_enrollment_and_prevents_replay(client):
    signup(client)
    assert (
        client.post(
            "/api/v1/auth/reauthenticate", json={"method": "totp", "code": "000000"}
        ).status_code
        == 400
    )
    secret, codes = setup_totp(client)
    expire_verification(client)
    assert (
        client.post("/api/v1/auth/reauthenticate", json={"password": PASSWORD}).status_code == 400
    )
    code = pyotp.TOTP(secret).at((int(time.time() // 30) + 1) * 30)
    old_cookie = client.cookies.get(store.COOKIE)
    accept(
        client, client.post("/api/v1/auth/reauthenticate", json={"method": "totp", "code": code})
    )
    assert client.cookies.get(store.COOKIE) != old_cookie
    assert store.identity(old_cookie) is None
    assert client.get("/api/v1/auth/security").json()["recently_verified"] is True
    assert (
        client.post(
            "/api/v1/auth/reauthenticate", json={"method": "totp", "code": code}
        ).status_code
        == 400
    )
    accept(
        client,
        client.post("/api/v1/auth/reauthenticate", json={"method": "totp", "code": codes[0]}),
    )
    assert (
        client.post(
            "/api/v1/auth/reauthenticate", json={"method": "totp", "code": codes[0]}
        ).status_code
        == 400
    )


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/password", {"new_password": "another secure password 2026"}),
        ("/2fa/disable", {}),
        ("/2fa/recovery-codes", {}),
    ],
)
def test_sensitive_actions_use_recent_verification_and_reject_stale_session(client, path, payload):
    signup(client)
    _, codes = setup_totp(client)
    expire_verification(client)
    blocked = client.post("/api/v1/auth" + path, json=payload)
    assert blocked.status_code == 403
    assert blocked.headers["X-WCM-Reauth"] == "required"
    assert client.get("/api/v1/auth/security").json()["totp_enabled"] is True
    accept(
        client,
        client.post("/api/v1/auth/reauthenticate", json={"method": "totp", "code": codes[0]}),
    )
    accept(client, client.post("/api/v1/auth" + path, json=payload))


def test_reauthentication_rejects_empty_password_and_is_rate_limited(client):
    signup(client)
    for _ in range(10):
        assert (
            client.post("/api/v1/auth/reauthenticate", json={"method": "password"}).status_code
            == 400
        )
    assert (
        client.post("/api/v1/auth/reauthenticate", json={"password": PASSWORD}).status_code == 429
    )


@pytest.mark.parametrize("without_handle", [False, True])
def test_real_passkey_reauthentication_scopes_account_and_rotates_session(client, without_handle):
    user = signup(client)["user"]
    authenticator = Authenticator()
    bind(client, authenticator)
    expire_verification(client)
    old_cookie = client.cookies.get(store.COOKIE)
    pending = accept(client, client.post("/api/v1/auth/passkeys/reauthenticate/options"))
    assert pending["options"]["userVerification"] == "required"
    assert [item["id"] for item in pending["options"]["allowCredentials"]] == [
        b64(authenticator.id)
    ]
    credential = authenticator.authenticate(pending["options"], user["id"])
    if without_handle:
        credential["response"].pop("userHandle")
    assertion = {"challenge_id": pending["challenge_id"], "credential": credential}
    assert (
        accept(client, client.post("/api/v1/auth/passkeys/reauthenticate/verify", json=assertion))[
            "user"
        ]["id"]
        == user["id"]
    )
    assert store.identity(old_cookie) is None
    assert client.get("/api/v1/auth/security").json()["recently_verified"] is True
    assert (
        client.post("/api/v1/auth/passkeys/reauthenticate/verify", json=assertion).status_code
        == 400
    )
    pending = accept(client, client.post("/api/v1/auth/passkeys/reauthenticate/options"))
    assert (
        client.post(
            "/api/v1/auth/passkeys/reauthenticate/verify",
            json={
                "challenge_id": pending["challenge_id"],
                "credential": authenticator.authenticate(pending["options"], user["id"]),
            },
        ).status_code
        == 400
    )  # Reused nonzero authenticator counter.


@pytest.mark.parametrize(
    "invalid", ["origin", "rp", "uv", "signature", "handle", "challenge", "expired"]
)
def test_passkey_reauthentication_rejects_invalid_proofs(client, invalid):
    user = signup(client)["user"]
    authenticator = Authenticator()
    bind(client, authenticator)
    expire_verification(client)
    pending = accept(client, client.post("/api/v1/auth/passkeys/reauthenticate/options"))
    options = dict(pending["options"])
    if invalid == "challenge":
        options["challenge"] = b64(b"wrong challenge")
    credential = authenticator.authenticate(
        options,
        "another-user" if invalid == "handle" else user["id"],
        origin="https://evil.test" if invalid == "origin" else ORIGIN,
        uv=invalid != "uv",
        rp="evil.test" if invalid == "rp" else "localhost",
    )
    if invalid == "signature":
        credential["response"]["signature"] = b64(b"invalid signature")
    if invalid == "expired":
        with store.transaction() as connection:
            connection.execute(update(store.challenges).values(expires_at=time.time() - 1))
    assert (
        client.post(
            "/api/v1/auth/passkeys/reauthenticate/verify",
            json={"challenge_id": pending["challenge_id"], "credential": credential},
        ).status_code
        == 400
    )
    assert client.get("/api/v1/auth/security").json()["recently_verified"] is False


def test_passkey_reauthentication_rejects_other_accounts_sessions_and_login_challenges(client):
    user = signup(client)["user"]
    authenticator = Authenticator()
    bind(client, authenticator)
    other = sibling(client)
    other_user = signup(other, "member")["user"]
    other_authenticator = Authenticator()
    bind(other, other_authenticator)
    pending = accept(client, client.post("/api/v1/auth/passkeys/reauthenticate/options"))
    assertion = {
        "challenge_id": pending["challenge_id"],
        "credential": authenticator.authenticate(pending["options"], user["id"]),
    }
    assert (
        other.post("/api/v1/auth/passkeys/reauthenticate/verify", json=assertion).status_code == 400
    )
    same_user = sibling(client)
    password_login(same_user)
    assert (
        same_user.post("/api/v1/auth/passkeys/reauthenticate/verify", json=assertion).status_code
        == 400
    )
    assert (
        client.post(
            "/api/v1/auth/passkeys/reauthenticate/verify",
            json={
                "challenge_id": pending["challenge_id"],
                "credential": other_authenticator.authenticate(
                    pending["options"], other_user["id"]
                ),
            },
        ).status_code
        == 400
    )
    pending_login = client.post("/api/v1/auth/passkeys/login/options").json()
    assert (
        client.post(
            "/api/v1/auth/passkeys/reauthenticate/verify",
            json={
                "challenge_id": pending_login["challenge_id"],
                "credential": authenticator.authenticate(pending_login["options"], user["id"]),
            },
        ).status_code
        == 400
    )


def test_passkey_reauthentication_requires_login_csrf_and_bound_credential(client):
    assert client.post("/api/v1/auth/passkeys/reauthenticate/options").status_code == 401
    signup(client)
    assert client.post("/api/v1/auth/passkeys/reauthenticate/options").status_code == 400
    assert (
        client.post(
            "/api/v1/auth/passkeys/reauthenticate/options", headers={"X-CSRF-Token": "bad"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/v1/auth/reauthenticate",
            json={"password": PASSWORD},
            headers={"X-CSRF-Token": "bad"},
        ).status_code
        == 403
    )
