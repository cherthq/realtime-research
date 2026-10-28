from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, SecretStr
from typing import Literal
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .catalog import Catalog
from .access import COOKIE, TTL, DemoAccess
from .config import ROOT, Settings, settings
from .gpu_pool import GPUWorkerPool
from .worker import DemoWorker, delete_room, room_token
from .voices import VoicePreparationError, VoiceService
from .voices import digest, save_json


@dataclass
class Session:
    id: str
    key: str
    avatar_id: str
    worker: DemoWorker
    created: float = field(default_factory=time.monotonic)
    ended: bool = False
    end_reason: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    final_metrics: dict | None = None
    mode: str = "diagnostic"
    voice_selection: dict | None = None
    model_id: str = "mock"
    close_task: asyncio.Task | None = field(default=None, repr=False)

    def public(self):
        return {
            "id": self.id,
            "avatar_id": self.avatar_id,
            "model_id": self.model_id,
            "mode": self.mode,
            "voice_selection": self.voice_selection,
            "ended": self.ended,
            "end_reason": self.end_reason,
            "metrics": self.final_metrics or self.worker.snapshot(),
        }


class SessionManager:
    def __init__(self, config: Settings, catalog: Catalog):
        self.config, self.catalog = config, catalog
        self.sessions: dict[str, Session] = {}
        self.active: str | None = None
        self.lock = asyncio.Lock()
        self.gpu_pool = GPUWorkerPool(config)

    async def create(
        self,
        avatar_id: str,
        voice_selection: dict | None = None,
        model_id: str = "mock",
    ):
        async with self.lock:
            if self.active is not None:
                raise HTTPException(409, "The demo is busy. End the active call first.")
            sid = secrets.token_hex(12)
            worker = DemoWorker(
                sid,
                avatar_id,
                self.config,
                self.catalog,
                voice_selection=voice_selection,
                model_id=model_id,
                gpu_pool=self.gpu_pool if model_id == "avatarforever" else None,
            )
            session = Session(
                sid,
                secrets.token_urlsafe(32),
                avatar_id,
                worker,
                mode="voice_diagnostic" if voice_selection else "diagnostic",
                voice_selection=voice_selection,
                model_id=model_id,
            )
            self.sessions[sid] = session
            self.active = sid
            for old_id in list(self.sessions)[:-20]:
                if self.sessions[old_id].ended:
                    del self.sessions[old_id]
        try:
            if model_id == "mock":
                await self.gpu_pool.close()
            timeout = (
                self.config.gpu_load_timeout_seconds + 45
                if model_id == "avatarforever"
                else 45
            )
            await asyncio.wait_for(session.worker.start(), timeout=timeout)
        except BaseException:
            await self.close(session, "worker_start_failed")
            raise
        return session

    def get(self, sid: str, key: str):
        session = self.sessions.get(sid)
        if not session or not hmac.compare_digest(session.key, key):
            raise HTTPException(404, "Session not found")
        return session

    async def close(self, session: Session, reason: str = "ended_by_user"):
        async with session.lock:
            if session.close_task is None:
                session.ended, session.end_reason = True, reason
                session.final_metrics = session.worker.snapshot()
                session.final_metrics["state"] = "ended"
                session.worker.state_changed.set()
                session.close_task = asyncio.create_task(
                    self._close(session), name=f"close-session-{session.id}"
                )
        # Keep the single-session lease until teardown and the report finish,
        # even if an HTTP request disconnects or another caller ends the call.
        await asyncio.shield(session.close_task)

    async def _close(self, session: Session):
        try:
            try:
                # Worker IPC/resource operations have their own deadlines. A
                # second outer timeout could release the lease while they drain.
                await session.worker.close()
            finally:
                await asyncio.wait_for(
                    delete_room(self.config, session.worker.room_name), timeout=5
                )
        except Exception as exc:
            session.final_metrics["cleanup_error"] = type(exc).__name__
            raise
        finally:
            report = {
                "schema_version": 1,
                "session_id": session.id,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "renderer": session.model_id,
                "mode": session.mode,
                "avatar_id": session.avatar_id,
                "end_reason": session.end_reason,
                "voice_selection": session.voice_selection,
                "metrics": session.final_metrics,
                "geometry": {
                    "width": 640,
                    "height": 480,
                    "renderer_is_model_inference": session.model_id != "mock",
                },
                "recorded_media": False,
                "qualification": "streaming_experimental"
                if session.model_id != "mock"
                else "diagnostic_only",
            }
            try:
                image_path = self.catalog.image(session.avatar_id)
                if image_path.is_file():
                    report["reference_sha256"] = digest(image_path.read_bytes())
                await asyncio.to_thread(
                    save_json,
                    ROOT / "runs/realtime/sessions" / session.id / "report.json",
                    report,
                )
            finally:
                async with self.lock:
                    if self.active == session.id:
                        self.active = None

    async def reap(self):
        while True:
            await asyncio.sleep(1)
            if self.active:
                session = self.sessions[self.active]
                worker = session.worker
                if worker.status == "preparing":
                    continue  # startup has its own bounded deadline
                age = time.monotonic() - session.created
                abandoned = (
                    not worker.connected_user
                    and time.monotonic() - worker.last_presence
                    > self.config.reconnect_grace_seconds
                )
                reason = (
                    "session_limit"
                    if age > self.config.session_limit_seconds
                    else "disconnected"
                    if abandoned
                    else "worker_error"
                    if worker.error
                    else None
                )
                if reason:
                    with contextlib.suppress(Exception):
                        await self.close(session, reason)


class StartRequest(BaseModel):
    avatar_id: str = "ref01"
    model_id: Literal["mock", "avatarforever"] = "mock"
    mode: Literal["diagnostic", "voice_diagnostic"] = "diagnostic"
    voice_profile_id: str | None = None


class PlayRequest(BaseModel):
    scenario_id: str


class CastRequest(BaseModel):
    preference: str = ""
    recast: bool = False


class AccessRequest(BaseModel):
    token: SecretStr = Field(max_length=256)


class PlaybackRequest(BaseModel):
    frame_id: int | None = Field(default=None, ge=0, le=4294967295)
    audio_playing: bool = False
    cancel_epoch: int | None = Field(default=None, ge=0, le=65535)
    mute_ms: float | None = Field(default=None, ge=0, le=60000, allow_inf_nan=False)


def create_app(config: Settings = settings) -> FastAPI:
    catalog = Catalog(config)
    voices = VoiceService(config, catalog)
    manager = SessionManager(config, catalog)
    access = DemoAccess(config.demo_access_token.get_secret_value())

    @asynccontextmanager
    async def lifespan(app):
        cleanup = asyncio.create_task(manager.reap())
        yield
        cleanup.cancel()
        await asyncio.gather(cleanup, return_exceptions=True)
        for session in list(manager.sessions.values()):
            if not session.ended or (
                session.close_task and not session.close_task.done()
            ):
                with contextlib.suppress(Exception):
                    await manager.close(session, "server_shutdown")
        await voices.close()
        await manager.gpu_pool.close()

    app = FastAPI(title="Chert realtime demo", lifespan=lifespan)
    app.state.manager = manager
    app.state.voices = voices
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"],
    )

    @app.middleware("http")
    async def local_boundary(request: Request, call_next):
        # The initial build is local-only. No permissive CORS or public dev secrets.
        origin = request.headers.get("origin")
        if origin:
            parsed = urlparse(origin)
            if (parsed.scheme, parsed.netloc) != (
                request.url.scheme,
                request.url.netloc,
            ):
                return JSONResponse(
                    {"detail": "Cross-origin access denied"}, status_code=403
                )
        if request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "Cross-site access denied"}, status_code=403)
        if request.url.path.startswith("/api/") and request.url.path != "/api/access":
            if not access.authenticated(
                request.cookies.get(COOKIE, ""),
                request.headers.get("authorization", ""),
            ):
                return JSONResponse(
                    {"detail": "Access token required"}, status_code=401
                )
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/api/access")
    async def access_status(request: Request):
        return {
            "required": bool(access.secret),
            "authenticated": access.authenticated(request.cookies.get(COOKIE, "")),
        }

    @app.post("/api/access")
    async def unlock(body: AccessRequest, request: Request):
        if not access.secret:
            return {"required": False, "authenticated": True}
        ticket = access.login(body.token.get_secret_value())
        if ticket is None:
            raise HTTPException(401, "Incorrect access code")
        response = JSONResponse({"required": True, "authenticated": True})
        response.set_cookie(
            COOKIE,
            ticket,
            max_age=TTL,
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
            path="/api",
        )
        return response

    @app.delete("/api/access")
    async def lock(request: Request):
        access.logout(request.cookies.get(COOKIE, ""))
        response = JSONResponse(
            {"required": bool(access.secret), "authenticated": not bool(access.secret)}
        )
        response.delete_cookie(COOKIE, path="/api")
        return response

    @app.get("/api/health")
    async def health():
        url = config.livekit_url.replace("wss://", "https://").replace(
            "ws://", "http://"
        )
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                livekit = (await client.get(url)).status_code == 200
        except httpx.HTTPError:
            livekit = False
        gpu = manager.gpu_pool.snapshot()
        return {
            "status": "ok",
            "livekit": livekit,
            "openai_configured": bool(config.openai_api_key.get_secret_value()),
            "avatarforever_ready": gpu["loaded"],
            "avatarforever": gpu,
            "mode": "experimental_avatar"
            if config.avatarforever_enabled
            else "local_diagnostic",
            "active_session": manager.active is not None,
            "control_transport": config.control_transport,
        }

    @app.get("/api/models")
    async def models():
        from .streaming import capabilities

        return [
            {
                "id": "mock",
                "name": "Local diagnostic renderer",
                "available": True,
                "qualification": "diagnostic_only",
                "capabilities": capabilities("mock"),
                "description": "Real WebRTC with a synthetic diagnostic overlay; no avatar model inference.",
            },
            {
                "id": "avatarforever",
                "name": "AvatarForever · 22B (experimental)",
                "available": config.avatarforever_enabled,
                "runtime": manager.gpu_pool.snapshot(),
                "qualification": "streaming_experimental",
                "capabilities": capabilities("avatarforever"),
                "description": "Bounded streaming adapter. Requires pinned weights and a prepared GPU; qualification results are configuration-specific.",
            },
        ]

    @app.get("/api/avatars")
    async def avatars():
        result = catalog.avatars()
        for avatar in result:
            try:
                saved = voices.saved(avatar["id"])
                if saved:
                    avatar["casting_status"] = saved["status"]
                    avatar["voice_profile_id"] = saved["voice_profile_id"]
            except (ValueError, OSError):
                avatar["casting_status"] = "preparation_error"
        return result

    @app.get("/api/voices")
    async def voice_profiles():
        try:
            return voices.profiles()
        except (ValueError, OSError):
            raise HTTPException(
                409, "Voice catalog is incomplete or changed. Prepare it again."
            ) from None

    @app.get("/api/voices/{voice_id}/preview")
    async def voice_preview(voice_id: str):
        try:
            return FileResponse(voices.preview(voice_id), media_type="audio/wav")
        except (ValueError, OSError):
            raise HTTPException(404, "Voice preview unavailable") from None

    @app.post("/api/avatars/{avatar_id}/cast-voice")
    async def cast_voice(avatar_id: str, body: CastRequest):
        if avatar_id not in catalog.portraits:
            raise HTTPException(404, "Unknown avatar")
        if len(body.preference) > 1000:
            raise HTTPException(422, "Voice preference must be at most 1000 characters")
        if manager.active:
            raise HTTPException(409, "End the call before changing a voice.")
        try:
            return await voices.cast(
                avatar_id, preference=body.preference, recast=body.recast
            )
        except VoicePreparationError as exc:
            raise HTTPException(409, str(exc)) from None
        except Exception:
            raise HTTPException(
                502, "Voice matching failed. Retry or select a prepared voice."
            ) from None

    @app.get("/api/avatars/{avatar_id}/image")
    async def image(avatar_id: str):
        if avatar_id not in catalog.portraits:
            raise HTTPException(404)
        path = catalog.image(avatar_id)
        if not path.is_file():
            raise HTTPException(404, "Portrait asset not prepared")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/scenarios")
    async def scenarios():
        return catalog.cases()

    @app.post("/api/sessions", status_code=201)
    async def start(body: StartRequest):
        if manager.active is not None:
            raise HTTPException(
                409, "The demo is busy. Wait for the active call to finish."
            )
        if body.avatar_id not in catalog.portraits:
            raise HTTPException(422, "Unknown avatar")
        if body.model_id == "avatarforever" and not config.avatarforever_enabled:
            raise HTTPException(
                409,
                "Enable AvatarForever only on the prepared GPU host. No fallback was started.",
            )
        selected_voice = None
        if body.mode == "voice_diagnostic":
            if not config.openai_api_key.get_secret_value():
                raise HTTPException(
                    409, "Configure OPENAI_API_KEY before starting a live voice call."
                )
            try:
                selected_voice = await voices.resolve_for_call(
                    body.avatar_id, body.voice_profile_id
                )
            except VoicePreparationError as exc:
                raise HTTPException(409, str(exc)) from None
            except Exception:
                raise HTTPException(
                    502, "Voice preparation failed. Retry or select a prepared voice."
                ) from None
        try:
            session = await manager.create(
                body.avatar_id, selected_voice, body.model_id
            )
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                503,
                "Could not start the call. Check local LiveKit and, for voice mode, OpenAI access.",
            ) from None
        return {
            **session.public(),
            "session_key": session.key,
            "connection": {
                "url": config.livekit_browser_url or config.livekit_url,
                "token": room_token(config, session.worker.room_name, "browser"),
            },
        }

    async def own(sid: str, x_session_key: str = Header(default="")):
        return manager.get(sid, x_session_key)

    @app.get("/api/sessions/{sid}")
    async def get_session(session: Session = Depends(own)):
        return session.public()

    @app.delete("/api/sessions/{sid}")
    async def end(session: Session = Depends(own)):
        await manager.close(session)
        return session.public()

    @app.post("/api/sessions/{sid}/play")
    async def play(body: PlayRequest, session: Session = Depends(own)):
        async with session.lock:
            if session.ended:
                raise HTTPException(409, "Session ended")
            try:
                await session.worker.play(body.scenario_id)
            except (KeyError, FileNotFoundError):
                raise HTTPException(404, "Scripted audio is unavailable") from None
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from None
        return session.public()

    @app.post("/api/sessions/{sid}/interrupt")
    async def interrupt(session: Session = Depends(own)):
        async with session.lock:
            if session.ended:
                raise HTTPException(409, "Session ended")
            await session.worker.interrupt()
        return session.public()

    @app.post("/api/sessions/{sid}/playback")
    async def playback(body: PlaybackRequest, session: Session = Depends(own)):
        if session.ended:
            raise HTTPException(409, "Session ended")
        # Do not take session.lock: interruption may be waiting for this receipt.
        accepted = session.worker.playback.feedback(
            body.frame_id, body.audio_playing, body.cancel_epoch, body.mute_ms
        )
        return {"accepted": accepted}

    @app.get("/api/sessions/{sid}/events")
    async def events(request: Request, session: Session = Depends(own)):
        async def stream():
            while not await request.is_disconnected():
                session.worker.state_changed.clear()
                yield f"data: {json.dumps(session.public())}\n\n"
                if session.ended:
                    return
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(session.worker.state_changed.wait(), 0.5)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    dist = ROOT / "realtime/web/dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")
    else:

        @app.get("/")
        async def missing_build():
            return JSONResponse(
                {
                    "detail": "Web build missing. Run npm install and npm run build in realtime/web."
                },
                status_code=503,
            )

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("realtime.app:app", host="127.0.0.1", port=8080)
