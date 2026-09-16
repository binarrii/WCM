"""Exercise the real auth guard, database and cryptographic verifiers end to end."""

import hashlib
import io
import json
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import cbor2
import pyotp
import pytest
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import Request, WebSocket
from fastapi.testclient import TestClient
from PIL import Image
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
    monkeypatch.setattr(store.config, "cookie_secure", False)
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


def verification(client, method, path, proof=None):
    data = accept(
        client,
        client.post(
            "/api/v1/auth/reauthenticate",
            json={"operation": f"{method} {path}", **(proof or {"password": PASSWORD})},
        ),
    )
    return {"X-WCM-Verification": data["verification_token"]}


def authorized(client, method, path, *, proof=None, **kwargs):
    headers = verification(client, method, path, proof)
    headers.update(kwargs.pop("headers", {}))
    return client.request(method, path, headers=headers, **kwargs)


def setup_totp(client):
    pending = accept(client, authorized(client, "POST", "/api/v1/auth/2fa/setup"))
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


def avatar_upload(client, color="red", **kwargs):
    content = io.BytesIO()
    Image.new("RGB", (300, 180), color).save(content, format="PNG")
    return client.put(
        "/api/v1/auth/avatar",
        files={"file": ("avatar.png", content.getvalue(), "image/png")},
        **kwargs,
    )


def avatar_url(user):
    return f"/api/v1/auth/avatars/{user['id']}/{user['avatar_version']}"


def test_avatar_lifecycle_is_shared_between_sessions_and_visible_to_superadmin(client):
    root = signup(client)
    assert root["user"]["avatar_version"] is None
    member = sibling(client)
    other_session = sibling(client)
    try:
        original = signup(member, "member")
        cookie = member.cookies[store.COOKIE]
        saved = accept(member, avatar_upload(member))
        assert member.cookies[store.COOKIE] == cookie
        assert saved["user"]["id"] == original["user"]["id"]
        assert len(saved["user"]["avatar_version"]) == 64
        image = member.get(avatar_url(saved["user"]))
        assert image.status_code == 200
        assert image.headers["content-type"] == "image/jpeg"
        assert "no-store" in image.headers["cache-control"]
        assert image.headers["x-content-type-options"] == "nosniff"
        with Image.open(io.BytesIO(image.content)) as normalized:
            assert normalized.size == (256, 256)
        assert client.get(avatar_url(saved["user"])).content == image.content
        listed = client.get("/api/v1/auth/users").json()["items"]
        assert (
            next(user for user in listed if user["id"] == saved["user"]["id"])["avatar_version"]
            == saved["user"]["avatar_version"]
        )
        assert (
            password_login(other_session, "member")["user"]["avatar_version"]
            == saved["user"]["avatar_version"]
        )
        replaced = accept(member, avatar_upload(member, "blue"))
        assert replaced["user"]["avatar_version"] != saved["user"]["avatar_version"]
        assert (
            other_session.get("/api/v1/auth/me").json()["user"]["avatar_version"]
            == replaced["user"]["avatar_version"]
        )
        assert other_session.get(avatar_url(replaced["user"])).status_code == 200
        assert member.get(avatar_url(saved["user"])).status_code == 404
        store.initialize()
        assert other_session.get(avatar_url(replaced["user"])).status_code == 200
        removed = accept(member, member.delete("/api/v1/auth/avatar"))
        assert removed["user"]["avatar_version"] is None
        assert other_session.get(avatar_url(replaced["user"])).status_code == 404
        assert other_session.get("/api/v1/auth/me").json()["user"]["avatar_version"] is None
    finally:
        member.close()
        other_session.close()


def test_avatar_requires_session_csrf_and_cannot_target_another_user(client):
    assert avatar_upload(client).status_code == 401
    assert client.delete("/api/v1/auth/avatar").status_code == 401
    root = signup(client)
    saved_root = accept(client, avatar_upload(client))
    assert avatar_upload(client, headers={"X-CSRF-Token": "invalid"}).status_code == 403
    assert (
        client.delete("/api/v1/auth/avatar", headers={"X-CSRF-Token": "invalid"}).status_code == 403
    )
    other = sibling(client)
    try:
        assert other.get(avatar_url(saved_root["user"])).status_code == 401
        member = signup(other, "member")
        assert other.get(avatar_url(saved_root["user"])).status_code == 403
        saved = accept(other, avatar_upload(other, "blue", data={"user_id": root["user"]["id"]}))
        assert saved["user"]["id"] == member["user"]["id"]
        assert (
            client.get("/api/v1/auth/me").json()["user"]["avatar_version"]
            == saved_root["user"]["avatar_version"]
        )
        assert (
            other.delete(
                f"/api/v1/auth/avatars/{root['user']['id']}/{saved_root['user']['avatar_version']}"
            ).status_code
            == 405
        )
    finally:
        other.close()


@pytest.mark.parametrize(
    "content,status",
    [(b"<svg/>", 422), (b"x" * (5 * 1024 * 1024 + 1), 413), (b"x" * (6 * 1024 * 1024), 413)],
)
def test_invalid_avatar_upload_preserves_current_image(client, content, status):
    signup(client)
    saved = accept(client, avatar_upload(client))
    response = client.put("/api/v1/auth/avatar", files={"file": ("bad.png", content, "image/png")})
    assert response.status_code == status
    assert (
        client.get("/api/v1/auth/me").json()["user"]["avatar_version"]
        == saved["user"]["avatar_version"]
    )
    assert client.get(avatar_url(saved["user"])).status_code == 200


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
        authorized(
            client, "PUT", "/api/v1/auth/roles/user", json={"permissions": ["review.read"]}
        ).status_code
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
        authorized(
            client, "PUT", f"/api/v1/auth/users/{user['id']}/role", json={"role": "admin"}
        ).status_code
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
        authorized(
            client, "PUT", f"/api/v1/auth/users/{root['id']}/role", json={"role": "user"}
        ).status_code
        == 400
    )
    assert (
        authorized(
            client, "PUT", f"/api/v1/auth/users/{root['id']}/active", json={"active": False}
        ).status_code
        == 400
    )
    assert (
        authorized(
            client, "PUT", "/api/v1/auth/roles/user", json={"permissions": ["system.manage"]}
        ).status_code
        == 400
    )
    assert (
        authorized(client, "PUT", "/api/v1/auth/roles/admin", json={"permissions": []}).status_code
        == 200
    )
    assert user_client.get("/api/v1/parameters").status_code == 403
    assert (
        authorized(
            client, "PUT", f"/api/v1/auth/users/{user['id']}/active", json={"active": False}
        ).status_code
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
        authorized(
            client,
            "PUT",
            "/api/v1/auth/roles/user",
            json={"permissions": ["review.run", "people.write"]},
        ).status_code
        == 200
    )
    roles = client.get("/api/v1/auth/roles").json()["roles"]
    assert set(roles["user"]) == {"review.run", "review.read", "people.write", "people.read"}
    assert client.put("/api/v1/auth/roles/superadmin", json={"permissions": []}).status_code == 400
    assert (
        authorized(
            client, "PUT", "/api/v1/auth/roles/admin", json={"permissions": ["users.manage"]}
        ).status_code
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
    assert client.post("/api/v1/auth/2fa/setup").status_code == 403
    assert authorized(client, "POST", "/api/v1/auth/2fa/setup").status_code == 200
    with store.transaction() as connection:
        connection.execute(update(store.sessions).values(expires_at=time.time() - 1))
    assert client.get("/api/v1/auth/me").status_code == 401


LAN_ORIGIN = "http://10.252.25.251:8000"
HTTPS_ORIGIN = "https://wcmcore.ai-t.wtvdev.com"


@pytest.mark.parametrize("origin", [LAN_ORIGIN, HTTPS_ORIGIN])
def test_dual_origins_cors_sessions_csrf_and_websocket(client, monkeypatch, origin):
    store.config.origins[:] = [LAN_ORIGIN, HTTPS_ORIGIN]
    monkeypatch.setattr(store.config, "rp_id", "wcmcore.ai-t.wtvdev.com")

    @client.app.middleware("http")
    async def proxy_terminates_tls(request, call_next):
        # The API sees HTTP even when the browser used HTTPS at the outer proxy.
        request.scope["scheme"] = "http"
        return await call_next(request)

    secure = origin == HTTPS_ORIGIN
    with closing(
        TestClient(client.app, base_url=origin, headers={"Origin": origin, "X-WCM-Client": "web"})
    ) as browser:
        preflight = browser.options(
            "/api/v1/auth/login",
            headers={
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,x-wcm-client,x-csrf-token,x-wcm-verification",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == origin
        assert preflight.headers["access-control-allow-credentials"] == "true"
        assert "Origin" in preflight.headers["vary"]
        user = signup(browser)["user"]
        assert next(iter(browser.cookies.jar)).secure is secure
        assert browser.get("/api/v1/auth/me").json()["user"]["id"] == user["id"]
        assert (
            browser.post("/api/v1/auth/logout", headers={"X-CSRF-Token": "bad"}).status_code == 403
        )
        socket_url = origin.replace("https:", "wss:").replace("http:", "ws:")
        with browser.websocket_connect(socket_url + "/api/v1/review_tasks/auth-probe") as socket:
            socket.send_text("ping")
            assert socket.receive_text() == "ping"
        passkey = browser.post("/api/v1/auth/passkeys/login/options")
        assert passkey.status_code == (200 if secure else 400)
        if secure:
            assert passkey.json()["options"]["rpId"] == "wcmcore.ai-t.wtvdev.com"
        rotated = authorized(
            browser, "POST", "/api/v1/auth/password", json={"new_password": PASSWORD + " new"}
        )
        accept(browser, rotated)
        assert ("Secure" in rotated.headers["set-cookie"]) is secure
        assert browser.get("/api/v1/auth/me").status_code == 200
        logged_out = browser.post("/api/v1/auth/logout")
        assert logged_out.status_code == 200
        assert ("Secure" in logged_out.headers["set-cookie"]) is secure
        assert browser.get("/api/v1/auth/me").status_code == 401
        login = browser.post(
            "/api/v1/auth/login", json={"username": "root", "password": PASSWORD + " new"}
        )
        accept(browser, login)
        assert ("Secure" in login.headers["set-cookie"]) is secure
        assert browser.get("/api/v1/auth/me").status_code == 200


@pytest.mark.parametrize("origin", ["https://evil.test", HTTPS_ORIGIN + ".evil.test", "null"])
def test_dual_origins_reject_unlisted_cors_writes_and_websockets(client, origin):
    store.config.origins[:] = [LAN_ORIGIN, HTTPS_ORIGIN]
    headers = {"Origin": origin}
    response = client.options(
        "/api/v1/auth/login", headers={**headers, "Access-Control-Request-Method": "POST"}
    )
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers
    assert client.post("/api/v1/auth/login", headers=headers).status_code == 403
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/v1/review_tasks/auth-probe", headers=headers):
            pass


@pytest.mark.parametrize(
    "scheme,origin,forced,expected",
    [
        ("http", LAN_ORIGIN, False, False),
        ("http", HTTPS_ORIGIN, False, True),
        ("https", "", False, True),
        ("http", "https://evil.test", False, False),
        ("http", "", False, False),
        ("http", LAN_ORIGIN, True, True),
    ],
)
def test_cookie_security_with_tls_termination(monkeypatch, scheme, origin, forced, expected):
    from api.auth import session_cookie_secure

    monkeypatch.setattr(store.config, "origins", [LAN_ORIGIN, HTTPS_ORIGIN])
    monkeypatch.setattr(store.config, "cookie_secure", forced)
    request = Request(
        {
            "type": "http",
            "scheme": scheme,
            "path": "/api/v1/auth/login",
            "headers": [(b"origin", origin.encode()), (b"x-forwarded-proto", b"https")],
            "server": ("api", 8000),
        }
    )
    assert session_cookie_secure(request) is expected


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
        authorized(
            stranger,
            "POST",
            "/api/v1/auth/2fa/disable",
            proof={"method": "totp", "code": codes[1]},
            json={},
        ),
    )
    assert not stranger.get("/api/v1/auth/me").json()["user"]["totp_enabled"]
    assert client.get("/api/v1/auth/me").status_code == 401


def test_setup_challenge_is_session_bound_and_cannot_overwrite_mfa(client):
    signup(client)
    pending = authorized(client, "POST", "/api/v1/auth/2fa/setup").json()
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
    assert second.post("/api/v1/auth/2fa/setup").status_code == 403


def test_password_change_invalidates_sessions_and_pending_logins(client):
    signup(client)
    _, codes = setup_totp(client)
    second = sibling(client)
    pending = password_login(second)
    accept(
        client,
        authorized(
            client,
            "POST",
            "/api/v1/auth/password",
            proof={"method": "totp", "code": codes[0]},
            json={"new_password": PASSWORD + "changed"},
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
    pending = accept(client, authorized(client, "POST", "/api/v1/auth/passkeys/register/options"))
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
    assert authorized(client, "DELETE", f"/api/v1/auth/passkeys/{key_id}").status_code == 200
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
    pending = authorized(client, "POST", "/api/v1/auth/passkeys/register/options").json()
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
    assert authorized(other, "DELETE", f"/api/v1/auth/passkeys/{key_id}").status_code == 404
    assert len(client.get("/api/v1/auth/security").json()["passkeys"]) == 1


def test_login_and_legacy_verified_time_never_authorize_mutations(client):
    signup(client)
    with store.transaction() as connection:
        connection.execute(update(store.sessions).values(verified_at=time.time() + 300))
    assert "recently_verified" not in client.get("/api/v1/auth/security").json()
    for path, body in [
        ("/2fa/setup", {}),
        ("/password", {"new_password": PASSWORD + "new"}),
        ("/passkeys/register/options", {}),
    ]:
        response = client.post("/api/v1/auth" + path, json=body)
        assert response.status_code == 403
        assert response.headers["X-WCM-Reauth"] == "required"
    assert client.put("/api/v1/auth/roles/user", json={"permissions": []}).status_code == 403


def test_verification_is_scoped_to_one_request_and_does_not_refresh_session(client):
    signup(client)
    path = "/api/v1/auth/roles/user"
    headers = verification(client, "PUT", path)
    assert client.put(path, json={"permissions": []}).status_code == 403
    assert client.put(path, json={"permissions": []}, headers=headers).status_code == 200
    assert (
        client.put(path, json={"permissions": ["people.read"]}, headers=headers).status_code == 400
    )
    assert client.put(path, json={"permissions": ["people.read"]}).status_code == 403
    assert authorized(client, "PUT", path, json={"permissions": ["people.read"]}).status_code == 200


@pytest.mark.parametrize("other_target", ["/api/v1/auth/roles/admin", "/api/v1/auth/2fa/setup"])
def test_grant_cannot_authorize_another_operation_or_target(client, other_target):
    signup(client)
    headers = verification(client, "PUT", "/api/v1/auth/roles/user")
    method = "PUT" if "/roles/" in other_target else "POST"
    assert (
        client.request(
            method,
            other_target,
            json={"permissions": []} if method == "PUT" else {},
            headers=headers,
        ).status_code
        == 403
    )
    assert (
        client.put("/api/v1/auth/roles/user", json={"permissions": []}, headers=headers).status_code
        == 400
    )


@pytest.mark.parametrize("other_account", [False, True])
def test_grant_cannot_cross_sessions_or_accounts(client, other_account):
    signup(client)
    path = "/api/v1/auth/2fa/setup"
    headers = verification(client, "POST", path)
    other = sibling(client)
    if other_account:
        signup(other, "member")
    else:
        password_login(other)
    assert other.post(path, headers=headers).status_code == 400
    assert client.post(path, headers=headers).status_code == 200


def test_one_grant_is_consumed_atomically_by_concurrent_requests(client):
    signup(client)
    path = "/api/v1/auth/roles/user"
    headers = verification(client, "PUT", path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(
            pool.map(
                lambda _: client.put(path, json={"permissions": []}, headers=headers), range(2)
            )
        )
    assert sorted(r.status_code for r in responses) == [200, 400]


def test_expired_grant_and_changed_session_are_rejected(client):
    signup(client)
    path = "/api/v1/auth/2fa/setup"
    headers = verification(client, "POST", path)
    with store.transaction() as connection:
        connection.execute(update(store.challenges).values(expires_at=time.time() - 1))
    assert client.post(path, headers=headers).status_code == 400
    headers = verification(client, "POST", path)
    password_login(client)
    assert client.post(path, headers=headers).status_code == 400


def test_totp_reauthentication_requires_enrollment_and_prevents_replay(client):
    signup(client)
    operation = "POST /api/v1/auth/2fa/recovery-codes"
    assert (
        client.post(
            "/api/v1/auth/reauthenticate",
            json={"operation": operation, "method": "totp", "code": "000000"},
        ).status_code
        == 400
    )
    secret, codes = setup_totp(client)
    assert (
        client.post(
            "/api/v1/auth/reauthenticate", json={"operation": operation, "password": PASSWORD}
        ).status_code
        == 400
    )
    code = pyotp.TOTP(secret).at((int(time.time() // 30) + 1) * 30)
    proof = {"operation": operation, "method": "totp", "code": code}
    accept(client, client.post("/api/v1/auth/reauthenticate", json=proof))
    assert client.post("/api/v1/auth/reauthenticate", json=proof).status_code == 400
    proof["code"] = codes[0]
    accept(client, client.post("/api/v1/auth/reauthenticate", json=proof))
    assert client.post("/api/v1/auth/reauthenticate", json=proof).status_code == 400


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/password", {"new_password": "another secure password 2026"}),
        ("/2fa/disable", {}),
        ("/2fa/recovery-codes", {}),
    ],
)
def test_sensitive_actions_always_need_a_new_proof(client, path, payload):
    signup(client)
    _, codes = setup_totp(client)
    path = "/api/v1/auth" + path
    assert client.post(path, json=payload).status_code == 403
    accept(
        client,
        authorized(client, "POST", path, proof={"method": "totp", "code": codes[0]}, json=payload),
    )
    assert client.post("/api/v1/auth/password", json={"new_password": PASSWORD}).status_code == 403


def test_verification_is_required_for_each_admin_write(client):
    signup(client)
    other = sibling(client)
    user = signup(other, "member")["user"]
    for suffix, payload in [("role", {"role": "admin"}), ("active", {"active": False})]:
        path = f"/api/v1/auth/users/{user['id']}/{suffix}"
        assert client.put(path, json=payload).status_code == 403
        assert authorized(client, "PUT", path, json=payload).status_code == 200
        assert client.put(path, json=payload).status_code == 403


def test_reauthentication_rejects_empty_password_and_is_rate_limited(client):
    signup(client)
    for _ in range(10):
        assert (
            client.post(
                "/api/v1/auth/reauthenticate",
                json={"operation": "POST /api/v1/auth/password", "method": "password"},
            ).status_code
            == 400
        )
    assert (
        client.post(
            "/api/v1/auth/reauthenticate",
            json={"operation": "POST /api/v1/auth/password", "password": PASSWORD},
        ).status_code
        == 429
    )


@pytest.mark.parametrize("without_handle", [False, True])
def test_real_passkey_produces_a_single_operation_grant(client, without_handle):
    user = signup(client)["user"]
    authenticator = Authenticator()
    bind(client, authenticator)
    operation = "POST /api/v1/auth/password"
    pending = accept(
        client,
        client.post("/api/v1/auth/passkeys/reauthenticate/options", json={"operation": operation}),
    )
    assert pending["options"]["userVerification"] == "required"
    assert [item["id"] for item in pending["options"]["allowCredentials"]] == [
        b64(authenticator.id)
    ]
    credential = authenticator.authenticate(pending["options"], user["id"])
    if without_handle:
        credential["response"].pop("userHandle")
    assertion = {"challenge_id": pending["challenge_id"], "credential": credential}
    grant = accept(
        client, client.post("/api/v1/auth/passkeys/reauthenticate/verify", json=assertion)
    )
    assert (
        client.post("/api/v1/auth/passkeys/reauthenticate/verify", json=assertion).status_code
        == 400
    )
    assert (
        client.post("/api/v1/auth/password", json={"new_password": PASSWORD + "new"}).status_code
        == 403
    )
    accept(
        client,
        client.post(
            "/api/v1/auth/password",
            json={"new_password": PASSWORD + "new"},
            headers={"X-WCM-Verification": grant["verification_token"]},
        ),
    )
    assert client.post("/api/v1/auth/password", json={"new_password": PASSWORD}).status_code == 403


@pytest.mark.parametrize(
    "invalid", ["origin", "rp", "uv", "signature", "handle", "challenge", "expired"]
)
def test_passkey_reauthentication_rejects_invalid_proofs(client, invalid):
    user = signup(client)["user"]
    authenticator = Authenticator()
    bind(client, authenticator)
    pending = accept(
        client,
        client.post(
            "/api/v1/auth/passkeys/reauthenticate/options",
            json={"operation": "POST /api/v1/auth/password"},
        ),
    )
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
    assert (
        client.post("/api/v1/auth/password", json={"new_password": PASSWORD + "new"}).status_code
        == 403
    )


def test_passkey_proof_cannot_cross_session_user_or_purpose(client):
    user = signup(client)["user"]
    authenticator = Authenticator()
    bind(client, authenticator)
    other = sibling(client)
    other_user = signup(other, "member")["user"]
    other_authenticator = Authenticator()
    bind(other, other_authenticator)
    pending = accept(
        client,
        client.post(
            "/api/v1/auth/passkeys/reauthenticate/options",
            json={"operation": "POST /api/v1/auth/password"},
        ),
    )
    assertion = {
        "challenge_id": pending["challenge_id"],
        "credential": authenticator.authenticate(pending["options"], user["id"]),
    }
    assert (
        other.post("/api/v1/auth/passkeys/reauthenticate/verify", json=assertion).status_code == 400
    )
    same = sibling(client)
    password_login(same)
    assert (
        same.post("/api/v1/auth/passkeys/reauthenticate/verify", json=assertion).status_code == 400
    )
    assertion["credential"] = other_authenticator.authenticate(pending["options"], other_user["id"])
    assert (
        client.post("/api/v1/auth/passkeys/reauthenticate/verify", json=assertion).status_code
        == 400
    )
    login = client.post("/api/v1/auth/passkeys/login/options").json()
    assert (
        client.post(
            "/api/v1/auth/passkeys/reauthenticate/verify",
            json={
                "challenge_id": login["challenge_id"],
                "credential": authenticator.authenticate(login["options"], user["id"]),
            },
        ).status_code
        == 400
    )


def test_old_enrollment_challenges_without_operation_proof_are_rejected(client):
    user = signup(client)["user"]
    identity = store.identity(client.cookies.get(store.COOKIE))
    secret = pyotp.random_base32()
    with store.transaction() as connection:
        pending = store.challenge(
            connection,
            "totp-setup",
            user["id"],
            identity["session"]["id"],
            {"secret": store.cipher().encrypt(secret.encode()).decode()},
        )
    assert (
        client.post(
            "/api/v1/auth/2fa/confirm",
            json={"challenge_id": pending, "code": pyotp.TOTP(secret).now()},
        ).status_code
        == 400
    )
    assert client.get("/api/v1/auth/security").json()["totp_enabled"] is False


def test_verification_endpoints_require_session_csrf_and_valid_scope(client):
    operation = {"operation": "POST /api/v1/auth/password"}
    assert (
        client.post("/api/v1/auth/passkeys/reauthenticate/options", json=operation).status_code
        == 401
    )
    assert (
        client.post(
            "/api/v1/auth/reauthenticate", json={**operation, "password": PASSWORD}
        ).status_code
        == 401
    )
    signup(client)
    for path, payload in [
        ("/passkeys/reauthenticate/options", operation),
        ("/reauthenticate", {**operation, "password": PASSWORD}),
    ]:
        assert (
            client.post(
                "/api/v1/auth" + path, json=payload, headers={"X-CSRF-Token": "bad"}
            ).status_code
            == 403
        )
    assert (
        client.post("/api/v1/auth/passkeys/reauthenticate/options", json=operation).status_code
        == 400
    )
    assert (
        client.post("/api/v1/auth/reauthenticate", json={"password": PASSWORD}).status_code == 422
    )
    assert (
        client.post(
            "/api/v1/auth/reauthenticate",
            json={"operation": "DELETE /api/v1/auth/users/all", "password": PASSWORD},
        ).status_code
        == 422
    )
