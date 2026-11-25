"""Loopback-only A/B preview. Both ports use the baseline's single GPU lease."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, asynccontextmanager
import socket
from unittest.mock import patch

import uvicorn
from fastapi import APIRouter
from uvicorn.importer import import_from_string

from realtime import app as serving
from realtime import worker
from realtime.voices import VoiceService

from .agent import LiveAgent, MODEL, VOICE


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
    trial_config = manager.config.model_copy(
        update=dict(
            tts_provider="openai", camera_vision=False, web_search_enabled=False
        )
    )
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
        return {
            **await original_health(),
            "conversation_provider": MODEL,
            "conversation_model": MODEL,
            "voice_configured": configured,
            "tts_configured": configured,
            "tts_provider": MODEL,
            "camera_vision": False,
            "web_search_enabled": False,
        }

    @routes.get("/api/preview/provider")
    async def provider():
        return {
            "model": MODEL,
            "voice": VOICE,
            "avatar_pool": "shared",
            "camera": False,
            "search": False,
        }

    trial.router.routes[:0] = routes.routes

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
                        super().__init__(*args, voice_selection=None, **kwargs)
                        self.settings = trial_config
                        self.buffer.max_pending_seconds = (
                            worker.LIVE_VOICE_PENDING_SECONDS
                        )
                        self.voice = LiveAgent(trial_config, self)
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
                    if receipts := getattr(trial.state, "start_receipts", None):
                        await receipts.close()
                    await trial.state.voices.close()

    baseline.router.lifespan_context = lifecycle

    async def route(scope, receive, send):
        is_trial = scope["type"] != "lifespan" and scope["server"][1] == trial_port
        await (trial if is_trial else baseline)(scope, receive, send)

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
