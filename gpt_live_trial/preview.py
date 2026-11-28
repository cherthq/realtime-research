"""Loopback-only A/B preview. Both ports use the baseline's single GPU lease."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, asynccontextmanager
import socket
import hmac
from unittest.mock import patch

import uvicorn
from fastapi import APIRouter
from uvicorn.importer import import_from_string

from realtime import app as serving
from realtime import worker
from realtime.voices import VoiceService

from .agent import LiveAgent, MODEL, VOICE, INSTRUCTIONS
from avatar_config_runtime.runtime import install as install_configuration
from realtime.config import ROOT


class TrialVoices(VoiceService):
    def profiles(self):
        return [
            dict(
                id=VOICE,
                voice=VOICE,
                provider=MODEL,
                instructions="",
                preview_available=False,
                review=dict(
                    description="GPT-Live 1.1 · Marin", presentation="OpenAI voice"
                ),
            )
        ]

    def saved(self, avatar_id):
        return dict(
            status="matched", voice_profile_id=VOICE, description="GPT-Live 1.1 · Marin"
        )

    async def resolve_for_call(self, avatar_id, override=None):
        if override and override != VOICE:
            raise ValueError("This comparison uses Marin")
        return dict(
            profile=self.profiles()[0], mode="configured_voice", preparation_ms=0
        )

    async def cast(self, *args, **kwargs):
        raise ValueError("The trial uses the same voice on every portrait")


def make_pair(baseline, trial_port):
    manager = baseline.state.manager
    trial_config = manager.config.model_copy(update=dict(tts_provider="openai"))
    # Reuse the existing gallery, admission lock, session registry and GPU pool.
    # Only patch constructors while building this isolated process's second app.
    with (
        patch.object(serving, "SessionManager", return_value=manager),
        patch.object(serving, "Catalog", return_value=manager.catalog),
        patch.object(serving, "VoiceService", TrialVoices),
    ):
        trial = serving.create_app(trial_config)
    regular_web = next((r for r in baseline.routes if r.name == "web"), None)
    trial_web = next((r for r in trial.routes if r.name == "web"), None)
    if regular_web and trial_web:
        trial_web.app = regular_web.app

    routes = APIRouter()
    original_health = next(
        r.endpoint for r in baseline.routes if r.path == "/api/health"
    )

    @routes.get("/api/health")
    async def health():
        configured = bool(trial_config.openai_api_key.get_secret_value())
        base = await original_health()
        return {
            **base,
            "call_startup_state": "warming" if not base.get("avatarforever_ready") else "busy" if base.get("active_session") else "ready",
            "conversation_provider": MODEL,
            "conversation_model": MODEL,
            "voice_configured": configured,
            "tts_configured": configured,
            "tts_provider": MODEL,
            "camera_vision": trial_config.camera_vision,
            "web_search_enabled": trial_config.web_search_enabled,
            "document_retrieval": True,
            "call_recording_enabled": bool(configuration and configuration.logs),
        }

    @routes.get("/api/preview/provider")
    async def provider():
        return {
            "assistant_name": "Chert",
            "model": MODEL,
            "voice": VOICE,
            "avatar_pool": "shared",
            "camera": trial_config.camera_vision,
            "search": trial_config.web_search_enabled,
            "retrieval": "public_chert_docs",
        }

    trial.router.routes[:0] = routes.routes

    configuration = install_configuration(
        trial, manager, INSTRUCTIONS, ROOT / "runs/avatar-config/runtime.json"
    )

    original_lifespan = baseline.router.lifespan_context

    @asynccontextmanager
    async def lifecycle(app):
        async with original_lifespan(app):
            baseline_worker = serving.DemoWorker

            class RoutedWorker(baseline_worker):
                def __init__(self, *args, voice_selection=None, **kwargs):
                    self.is_live = bool(
                        voice_selection
                        and voice_selection["profile"].get("provider") == MODEL
                    )
                    if self.is_live:
                        if configuration:
                            args = (*args[:3], configuration.worker_catalog(args[3]), *args[4:])
                        super().__init__(*args, voice_selection=None, **kwargs)
                        self.settings = trial_config
                        self.buffer.max_pending_seconds = (
                            worker.LIVE_VOICE_PENDING_SECONDS
                        )
                        self.voice = LiveAgent(trial_config, self)
                        self.parallel_startup = True
                        if configuration:
                            configuration.configure_voice(self.voice)
                            configuration.configure_worker(self)
                    else:
                        super().__init__(
                            *args, voice_selection=voice_selection, **kwargs
                        )

                def snapshot(self):
                    data = super().snapshot()
                    if self.is_live:
                        data["voice_mode"] = MODEL
                    return data

            with patch.object(serving, "DemoWorker", RoutedWorker):
                try:
                    yield
                finally:
                    if configuration:
                        await configuration.close()
                    if receipts := getattr(trial.state, "start_receipts", None):
                        await receipts.close()
                    await trial.state.voices.close()

    baseline.router.lifespan_context = lifecycle

    async def route(scope, receive, send):
        is_trial = scope["type"] != "lifespan" and scope["server"][1] == trial_port
        token = configuration.trial_request.set(is_trial) if configuration else None
        source_token = None
        if configuration and is_trial:
            headers = dict(scope.get('headers', []))
            source_token = configuration.call_source.set('facetime' if headers.get(b'x-call-source') == b'facetime' else 'web')
            # The existing private FaceTime companion requests its greeting only
            # after the handset joins. Keep that deployed companion compatible.
            path = scope.get('path', '')
            if path.startswith('/api/sessions/') and path.endswith('/greet'):
                session = manager.sessions.get(path.split('/')[3])
                if session and hmac.compare_digest(headers.get(b'x-session-key',b''), session.key.encode()) and (recorder := getattr(session.worker, 'call_recorder', None)):
                    recorder.source = 'facetime'
        try:
            target = trial if is_trial else baseline
            if configuration and is_trial and scope.get("path") == "/api/avatar-config":
                target = configuration.api
            await target(scope, receive, send)
        finally:
            if configuration:
                configuration.trial_request.reset(token)
                if source_token is not None:
                    configuration.call_source.reset(source_token)

    return route


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-factory", default="realtime.app:create_app")
    parser.add_argument("--base-port", type=int, default=18821)
    parser.add_argument("--trial-port", type=int, default=18837)
    args = parser.parse_args()
    app = make_pair(import_from_string(args.base_factory)(), args.trial_port)
    with ExitStack() as stack:
        sockets = []
        for port in (args.base_port, args.trial_port):
            sock = stack.enter_context(socket.socket())
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
            sockets.append(sock)
        uvicorn.Server(uvicorn.Config(app, proxy_headers=False)).run(sockets=sockets)


if __name__ == "__main__":
    main()
