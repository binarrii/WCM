"""Default-deny HTTP and WebSocket authorization, including image delivery."""

import asyncio
import contextlib
import json
import secrets

from fastapi import HTTPException, WebSocketDisconnect
from sqlalchemy.exc import SQLAlchemyError
from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse

from . import auth_store as store

PUBLIC = {
    ("GET", "/api/v1/health"),
    ("GET", "/api/v1/auth/config"),
    ("POST", "/api/v1/auth/register"),
    ("POST", "/api/v1/auth/login"),
    ("POST", "/api/v1/auth/login/2fa"),
    ("POST", "/api/v1/auth/passkeys/login/options"),
    ("POST", "/api/v1/auth/passkeys/login/verify"),
}


def required_permission(path, method):
    path = path.rstrip("/")
    if path.startswith("/images/"):
        return {"people.read", "review.read"}
    if path.startswith("/api/v1/auth/"):
        return set()
    if path.startswith("/api/v1/parameters"):
        return {"parameters.manage"}
    if path.startswith("/api/v1/insightface/"):
        return {"system.manage"}
    if path.startswith("/api/v1/face_records"):
        return {"people.read" if method in {"GET", "HEAD"} else "people.write"}
    if path.startswith("/api/v1/review_tasks"):
        if method == "DELETE" or path.endswith("/cancel"):
            return {"review.manage"}
        if method == "POST" and path == "/api/v1/review_tasks":
            return {"review.run"}
        return {"review.read"}
    if path in {"/api/v1/search", "/api/v1/ws/search", "/api/v1/detect"}:
        return {"people.read"}
    if path == "/api/v1/ws/analyze_media" and method == "WS":
        return {"review.read"}
    if path in {
        f"/api/v1/{prefix}{name}"
        for prefix in ("", "ws/")
        for name in ("analyze_media", "detect_sensitive", "detect_nsfw")
    }:
        return {"review.run"}
    return {"system.manage"}


def protected(path):
    return path.startswith(("/api/", "/images/")) or path in {
        "/docs",
        "/redoc",
        "/openapi.json",
        "/docs/oauth2-redirect",
    }


def authorize(scope):
    connection = HTTPConnection(scope)
    path = scope["path"].rstrip("/")
    method = scope.get("method", "WS")
    websocket = scope["type"] == "websocket"
    origin = connection.headers.get("origin")
    unsafe = websocket or method not in {"GET", "HEAD", "OPTIONS"}
    if unsafe and origin and origin not in store.config.origins:
        raise HTTPException(403, "请求来源不受信任")
    if websocket and not origin:
        raise HTTPException(403, "WebSocket 必须提供受信任的 Origin")
    if (method, path) in PUBLIC:
        if unsafe and connection.headers.get("x-wcm-client") != "web":
            raise HTTPException(403, "缺少客户端请求标识")
        return None
    identity = store.identity(connection.cookies.get(store.COOKIE))
    if not identity:
        raise HTTPException(401, "请先登录")
    if unsafe and not websocket:
        csrf = connection.headers.get("x-csrf-token", "")
        if not secrets.compare_digest(csrf, identity["session"]["csrf"]):
            raise HTTPException(403, "安全验证已失效，请刷新页面")
    required = required_permission(path, method)
    if required and not required.intersection(identity["user"]["permissions"]):
        raise HTTPException(403, "当前角色没有此操作权限")
    return identity


class AuthGuard:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in {"http", "websocket"} or not protected(scope["path"]):
            return await self.app(scope, receive, send)
        # CORS handles preflight outside this middleware; no business route uses OPTIONS.
        if scope.get("method") == "OPTIONS":
            return await self.app(scope, receive, send)
        try:
            identity = await asyncio.to_thread(authorize, scope)
        except (HTTPException, SQLAlchemyError) as error:
            code = error.status_code if isinstance(error, HTTPException) else 503
            detail = error.detail if isinstance(error, HTTPException) else "账户服务暂时不可用"
            if scope["type"] == "websocket":
                return await send(
                    {"type": "websocket.close", "code": 4401 if code == 401 else 4403}
                )
            return await JSONResponse(
                {"detail": detail}, status_code=code, headers={"Cache-Control": "no-store"}
            )(scope, receive, send)
        scope.setdefault("state", {})["identity"] = identity
        if scope["type"] == "http":
            if scope["path"].startswith("/api/v1/auth/") and scope.get("method") not in {
                "GET",
                "HEAD",
                "OPTIONS",
            }:
                # Bound credentials even for chunked requests without Content-Length.
                body = bytearray()
                while True:
                    part = await receive()
                    if part["type"] == "http.disconnect":
                        return
                    body.extend(part.get("body", b""))
                    if len(body) > 65536:
                        return await JSONResponse({"detail": "认证请求过大"}, status_code=413)(
                            scope, receive, send
                        )
                    if not part.get("more_body", False):
                        break
                original_receive = receive
                delivered = False

                async def body_receive():
                    nonlocal delivered
                    if not delivered:
                        delivered = True
                        return {"type": "http.request", "body": bytes(body), "more_body": False}
                    return await original_receive()

                receive = body_receive

            async def private_send(message):
                if message["type"] == "http.response.start":
                    headers = [
                        (key, value)
                        for key, value in message.get("headers", [])
                        if key.lower() != b"cache-control"
                    ]
                    message = {
                        **message,
                        "headers": headers
                        + [
                            (b"cache-control", b"private, no-store"),
                            (b"x-content-type-options", b"nosniff"),
                        ],
                    }
                await send(message)

            return await self.app(scope, receive, private_send)

        # Re-check long-lived sockets before every data exchange. Logging out,
        # disabling an account or changing permissions also revokes open sockets.
        closed = False

        async def check():
            nonlocal closed
            if closed:
                raise WebSocketDisconnect(4401)
            try:
                await asyncio.to_thread(authorize, scope)
            except (HTTPException, SQLAlchemyError):
                closed = True
                await send({"type": "websocket.close", "code": 4401})
                raise WebSocketDisconnect(4401)

        async def guarded_receive():
            nonlocal closed
            message = await receive()
            if message["type"] == "websocket.receive":
                await check()
                if scope["path"].rstrip("/") == "/api/v1/ws/analyze_media":
                    try:
                        payload = json.loads(message.get("text") or "{}")
                    except ValueError:
                        payload = None
                    if not isinstance(payload, dict) or payload.get("type") != "subscribe":
                        latest = await asyncio.to_thread(authorize, scope)
                        if "review.run" not in latest["user"]["permissions"]:
                            closed = True
                            await send({"type": "websocket.close", "code": 4403})
                            raise WebSocketDisconnect(4403)
            return message

        async def guarded_send(message):
            if message["type"] == "websocket.send":
                await check()
            if not closed:
                await send(message)

        with contextlib.suppress(WebSocketDisconnect):
            await self.app(scope, guarded_receive, guarded_send)
