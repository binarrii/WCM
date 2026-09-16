"""Verify real MySQL authentication across two independent API processes.

Run in a disposable container with an empty wcm_auth_verify_* database and the
normal WCM_AUTH_SECRET_KEY. Never uses the production user database.
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pyotp
from sqlalchemy import func, select

from api import auth_store as store

BASES = ["http://127.0.0.1:18141", "http://127.0.0.1:18142"]
HEADERS = {"Origin": "http://localhost:5173", "X-WCM-Client": "web"}
PASSWORD = "Isolated cluster verification 2026!"


def require(response, status=200):
    assert response.status_code == status, (response.status_code, response.text)
    return response.json()


def accept(client, response, status=200):
    data = require(response, status)
    if "csrf_token" in data:
        client.headers["X-CSRF-Token"] = data["csrf_token"]
    return data


def verification(client, method, path, proof=None, base=None):
    data = require(client.post((base or "") + "/api/v1/auth/reauthenticate", json={
        "operation": f"{method} {path}", **(proof or {"password": PASSWORD})
    }))
    return {"X-WCM-Verification": data["verification_token"]}


def authorized(client, method, path, **kwargs):
    return client.request(method, path, headers=verification(client, method, path), **kwargs)


def main():
    db = store.engine()
    assert db.dialect.name == "mysql" and db.url.database.startswith("wcm_auth_verify_"), "Requires an isolated MySQL verification database"
    store.initialize()
    with db.connect() as connection:
        assert connection.execute(select(func.count()).select_from(store.users)).scalar() == 0, "Requires an empty verification database"
    env = dict(os.environ, WCM_AUTH_ORIGINS=json.dumps([HEADERS["Origin"]]), WCM_AUTH_RP_ID="localhost", WCM_AUTH_COOKIE_SECURE="false")
    processes = [subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(port), "--lifespan", "off"], env=env, stdout=subprocess.DEVNULL) for port in (18141, 18142)]
    clients = []
    try:
        for base in BASES:
            deadline = time.monotonic() + 30
            while True:
                try:
                    require(httpx.get(base + "/api/v1/auth/config"))
                    break
                except httpx.TransportError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(.2)

        def register(index):
            client = httpx.Client(base_url=BASES[index % 2], headers=HEADERS, timeout=15)
            clients.append(client)
            data = accept(client, client.post("/api/v1/auth/register", json={"username": f"verify{index}", "display_name": f"Verification {index}", "password": PASSWORD}), 201)
            return client, data["user"]

        with ThreadPoolExecutor(max_workers=4) as pool:
            accounts = list(pool.map(register, range(4)))
        assert sum(user["role"] == "superadmin" for _, user in accounts) == 1
        root, root_user = next(item for item in accounts if item[1]["role"] == "superadmin")
        member, member_user = next(item for item in accounts if item[1]["role"] == "user")
        for base in BASES:
            assert require(root.get(base + "/api/v1/auth/me"))["user"]["id"] == root_user["id"]
            assert member.get(base + "/api/v1/parameters").status_code == 403
        require(authorized(root, "PUT", f"/api/v1/auth/users/{member_user['id']}/role", json={"role": "admin"}))
        for base in BASES:
            assert member.get(base + "/api/v1/auth/me").status_code == 401
        assert accept(member, member.post("/api/v1/auth/login", json={"username": member_user["username"], "password": PASSWORD}))["user"]["role"] == "admin"
        assert member.get("/api/v1/auth/users").status_code == 403
        require(authorized(root, "PUT", "/api/v1/auth/roles/admin", json={"permissions": ["review.read"]}))
        assert member.get(BASES[1] + "/api/v1/parameters").status_code == 403

        pending = require(authorized(root, "POST", "/api/v1/auth/2fa/setup"))
        enabled = accept(root, root.post("/api/v1/auth/2fa/confirm", json={"challenge_id": pending["challenge_id"], "code": pyotp.TOTP(pending["secret"]).now()}))
        stranger = httpx.Client(base_url=BASES[0], headers=HEADERS, timeout=15)
        clients.append(stranger)
        login = require(stranger.post("/api/v1/auth/login", json={"username": root_user["username"], "password": PASSWORD}))
        assert login["mfa_required"]
        assert stranger.get(BASES[1] + "/api/v1/auth/me").status_code == 401
        proof = {"challenge_id": login["challenge_id"], "code": enabled["recovery_codes"][0]}
        assert accept(stranger, stranger.post(BASES[1] + "/api/v1/auth/login/2fa", json=proof))["user"]["id"] == root_user["id"]
        assert stranger.post(BASES[0] + "/api/v1/auth/login/2fa", json=proof).status_code == 400
        require(stranger.post(BASES[0] + "/api/v1/auth/logout"))
        assert stranger.get(BASES[1] + "/api/v1/auth/me").status_code == 401
        path = "/api/v1/auth/roles/admin"
        for base in BASES:
            assert root.put(base + path, json={"permissions": ["review.read"]}).status_code == 403
        grant = verification(root, "PUT", path, {"method": "totp", "code": enabled["recovery_codes"][1]}, BASES[0])
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda base: root.put(base + path, json={"permissions": ["review.read"]}, headers=grant).status_code, BASES))
        assert sorted(results) == [200, 400], results
        assert root.put(BASES[1] + path, json={"permissions": []}).status_code == 403
        assert root.post(BASES[1] + "/api/v1/auth/reauthenticate", json={"operation": "PUT " + path, "method": "totp", "code": enabled["recovery_codes"][1]}).status_code == 400
        grant = verification(root, "PUT", path, {"method": "totp", "code": enabled["recovery_codes"][2]}, BASES[1])
        require(root.put(BASES[0] + path, json={"permissions": ["review.read"]}, headers=grant))
        password_path = "/api/v1/auth/password"
        grant = verification(root, "POST", password_path, {"method": "totp", "code": enabled["recovery_codes"][3]}, BASES[1])
        accept(root, root.post(BASES[0] + password_path, json={"new_password": PASSWORD + " changed"}, headers=grant))
        assert root.post(BASES[1] + password_path, json={"new_password": PASSWORD}).status_code == 403
        assert require(stranger.post(BASES[1] + "/api/v1/auth/login", json={"username": root_user["username"], "password": PASSWORD + " changed"}))["mfa_required"]
        print(json.dumps({"result": "passed", "database": db.url.database, "api_processes": 2,
                          "concurrent_registrations": 4, "superadmins": 1,
                          "checks": ["shared-cookie-sessions", "cross-process-role-revocation", "role-permissions", "encrypted-TOTP", "cross-process-MFA-challenge", "recovery-replay-rejected", "cross-process-logout", "single-use-grant-race", "every-write-requires-proof", "cross-process-operation-grant", "password-login-keeps-MFA"]}))
    finally:
        for client in clients:
            client.close()
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
