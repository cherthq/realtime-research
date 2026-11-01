"""Restricted public HTTP gateway. Run behind HTTPS, with LiveKit Cloud media.

Provider credentials and the private app access code remain on the server.
The underlying application must use AVATAR_CATALOG=demo and stay on loopback.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
from contextlib import asynccontextmanager
import json
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse, Response

from .scripts.demo_host import DEFAULT, load
from .call_queue import CallQueue

PUBLIC_IDS = {f"call{i:02}" for i in range(1, 11)}
WEBSITE_ORIGINS = {"https://dunlichan.com", "https://www.dunlichan.com"}
MAX_BODY = 600_000


def allowed(method: str, path: str) -> bool:
    if method == "GET":
        return bool(
            path
            in {
                "/",
                "/chert-logo.png",
                "/api/access",
                "/api/health",
                "/api/models",
                "/api/avatars",
                "/api/voices",
                "/api/scenarios",
                "/api/queue",
            }
            or re.fullmatch(r"/assets/[A-Za-z0-9_.-]+", path)
            or re.fullmatch(r"/api/avatars/call(?:0[1-9]|10)/image", path)
            or re.fullmatch(r"/api/voices/[A-Za-z0-9_-]{1,128}/preview", path)
            or re.fullmatch(r"/api/sessions/[a-f0-9]{24}(?:/transcripts)?", path)
        )
    if method == "POST":
        return path in {"/api/sessions", "/api/queue"} or bool(
            re.fullmatch(r"/api/sessions/[a-f0-9]{24}/(?:interrupt|playback)", path)
        )
    return method == "DELETE" and (
        path == "/api/queue" or bool(re.fullmatch(r"/api/sessions/[a-f0-9]{24}", path))
    )


class StartLimiter:
    """Bound paid starts globally, even when visitors rotate IP addresses."""

    def __init__(self):
        self.attempts = deque(maxlen=30)

    def admit(self, now: float) -> bool:
        while self.attempts and now - self.attempts[0] >= 3600:
            self.attempts.popleft()
        if len(self.attempts) >= 30 or sum(now - t < 60 for t in self.attempts) >= 3:
            return False
        self.attempts.append(now)
        return True


def create_gateway(origin: str, upstream: str, token: str, *, transport=None):
    parsed = urlsplit(origin)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Supply an HTTPS origin without a trailing slash")
    target = urlsplit(upstream)
    if target.scheme != "http" or target.hostname != "127.0.0.1" or not token:
        raise ValueError("Use an authenticated loopback application")
    client = httpx.AsyncClient(
        base_url=upstream, timeout=950, transport=transport, follow_redirects=False
    )
    limiter = StartLimiter()
    starts: set[asyncio.Task] = set()

    async def health():
        response = await client.get(
            "/api/health", headers={"Authorization": f"Bearer {token}"}, timeout=5
        )
        response.raise_for_status()
        return response.json()

    queue = CallQueue(health)

    async def start_call(key, body):
        try:
            return await client.post(
                "/api/sessions",
                content=body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )
        finally:
            await queue.finish(key)

    @asynccontextmanager
    async def lifespan(app):
        yield
        if starts:
            await asyncio.gather(*starts, return_exceptions=True)
        await client.aclose()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.queue = queue

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.headers.get("host") != parsed.netloc:
            return JSONResponse({"detail": "Unknown host"}, status_code=400)
        supplied_origin = request.headers.get("origin")
        readiness = (
            request.method == "GET"
            and request.url.path == "/ready"
            and supplied_origin in WEBSITE_ORIGINS
        )
        if supplied_origin and supplied_origin != origin and not readiness:
            return JSONResponse(
                {"detail": "Cross-origin access denied"}, status_code=403
            )
        if (
            request.url.path.startswith("/api/")
            and request.headers.get("sec-fetch-site") == "cross-site"
        ):
            return JSONResponse({"detail": "Cross-site access denied"}, status_code=403)
        if request.method in {"POST", "DELETE"} and supplied_origin != origin:
            return JSONResponse(
                {"detail": "Same-origin request required"}, status_code=403
            )
        response = await call_next(request)
        if readiness:
            response.headers["Access-Control-Allow-Origin"] = supplied_origin
            response.headers["Vary"] = "Origin"
        response.headers.update(
            {
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "Content-Security-Policy": "frame-ancestors 'self' https://dunlichan.com https://www.dunlichan.com",
            }
        )
        return response

    @app.api_route("/{path:path}", methods=["GET", "POST", "DELETE"])
    async def proxy(path: str, request: Request):
        path = "/" + path
        if path == "/ready" and request.method == "GET":
            try:
                result = await client.get(
                    "/api/health",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=5,
                )
                result.raise_for_status()
                health = result.json()
                return JSONResponse(
                    {
                        "ready": bool(health.get("avatarforever_ready")),
                        "busy": bool(health.get("active_session")),
                    }
                )
            except (httpx.HTTPError, ValueError):
                return JSONResponse({"ready": False, "busy": False}, status_code=503)
        if not allowed(request.method, path):
            return JSONResponse(
                {"detail": "Not available in the public demo"}, status_code=404
            )
        queue_key = request.headers.get("x-queue-key", "")
        if path == "/api/queue":
            if request.method == "DELETE":
                return JSONResponse(await queue.leave(queue_key))
            return JSONResponse(
                await queue.visit(queue_key, join=request.method == "POST")
            )
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_BODY:
                return JSONResponse({"detail": "Request too large"}, status_code=413)
        if path == "/api/sessions" and request.method == "POST":
            try:
                payload = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                payload = None
            if (
                not isinstance(payload, dict)
                or payload.get("avatar_id") not in PUBLIC_IDS
                or payload.get("model_id") != "avatarforever"
                or payload.get("mode") != "voice_diagnostic"
            ):
                return JSONResponse(
                    {"detail": "Select a public avatar for a live call"},
                    status_code=422,
                )
            await queue.claim(queue_key)
            if not limiter.admit(time.monotonic()):
                await queue.finish(queue_key)
                return JSONResponse(
                    {"detail": "Please wait before starting another call"},
                    status_code=429,
                    headers={"Retry-After": "60"},
                )
        headers = {"Authorization": f"Bearer {token}"}
        for key in ("content-type", "x-session-key"):
            if key in request.headers:
                headers[key] = request.headers[key]
        # Access is granted by this gateway; browser never receives the app secret.
        target_path = "/api/health" if path == "/api/access" else path
        try:
            if path == "/api/sessions" and request.method == "POST":
                task = asyncio.create_task(start_call(queue_key, bytes(body)))
                starts.add(task)

                # A closed browser must not release admission while the backend
                # is still starting a call. SessionManager reaps abandoned calls.
                def finished(task):
                    starts.discard(task)
                    if not task.cancelled():
                        task.exception()  # Observe errors even if the browser left.

                task.add_done_callback(finished)
                result = await asyncio.shield(task)
            else:
                result = await client.request(
                    request.method,
                    target_path,
                    params=request.query_params,
                    content=bytes(body),
                    headers=headers,
                )
        except httpx.HTTPError:
            return JSONResponse(
                {"detail": "The demo is temporarily offline"}, status_code=503
            )
        data = result.content
        if result.status_code == 200 and path == "/api/access":
            data = json.dumps(
                {"required": False, "authenticated": True, "queue_required": True}
            ).encode()
        elif result.status_code == 200 and path == "/api/avatars":
            avatars = result.json()
            if {a["id"] for a in avatars} != PUBLIC_IDS:
                return JSONResponse(
                    {"detail": "Public catalog is not ready"}, status_code=503
                )
        elif result.status_code == 200 and path == "/api/health":
            health = result.json()
            health["control_transport"] = "poll"
            data = json.dumps(health).encode()
        elif result.status_code == 200 and path == "/api/models":
            data = json.dumps(
                [m for m in result.json() if m["id"] == "avatarforever"]
            ).encode()
        elif result.status_code == 200 and path == "/api/scenarios":
            data = b"[]"
        return Response(
            data,
            status_code=result.status_code,
            headers={
                "Content-Type": result.headers.get(
                    "content-type", "application/octet-stream"
                )
            },
        )

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    parser.add_argument("--port", type=int, default=18820)
    args = parser.parse_args()
    profile = load(args.directory)
    from .config import Settings

    if Settings().avatar_catalog != "demo":
        raise ValueError("Public hosting requires AVATAR_CATALOG=demo")
    app = create_gateway(
        args.origin, f"http://127.0.0.1:{profile['api_port']}", profile["access_code"]
    )
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
