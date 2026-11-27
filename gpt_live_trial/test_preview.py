import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from fastapi import APIRouter

from realtime.app import create_app
from realtime.config import Settings
from gpt_live_trial.preview import make_pair


def test_main_and_trial_share_admission_and_only_trial_lists_marin():
    async def scenario():
        app = create_app(Settings(_env_file=None))
        custom = APIRouter()

        @custom.get("/api/health")
        async def custom_health():
            return {
                "ui_asset": "/assets/private-build.js",
                "conversation_model": "control",
            }

        app.router.routes[:0] = custom.routes
        pair = make_pair(app, 18837)
        transport = httpx.ASGITransport(app=pair)
        async with httpx.AsyncClient(transport=transport) as client:
            trial = "http://127.0.0.1:18837"
            control = "http://127.0.0.1:18821"
            health = (await client.get(trial + "/api/health")).json()
            assert health["ui_asset"] == "/assets/private-build.js"
            assert health["conversation_model"] == "gpt-live-1.1"
            result = await client.get(trial + "/api/voices")
            assert [row["id"] for row in result.json()] == ["marin"]
            provider = (await client.get(trial + "/api/preview/provider")).json()
            assert provider["assistant_name"] == "Chert"
            assert provider["model"] == "gpt-live-1.1"
            assert provider["retrieval"] == "public_chert_docs"
            assert provider["search"] == app.state.manager.config.web_search_enabled
            assert provider["camera"] == app.state.manager.config.camera_vision
            assert (
                await client.get(control + "/api/preview/provider")
            ).status_code == 404
            app.state.manager.active = "busy"
            for origin in (control, trial):
                r = await client.post(
                    origin + "/api/sessions",
                    json={"mode": "voice_diagnostic", "avatar_id": "call07"},
                )
                assert r.status_code == 409
        await app.state.voices.close()

    asyncio.run(scenario())


def test_facetime_start_recovery_uses_one_live_session(monkeypatch):
    async def scenario():
        app = create_app(Settings(_env_file=None))
        manager = app.state.manager
        session = SimpleNamespace(
            id="fixture", key="fixture-secret", ended=False,
            worker=SimpleNamespace(room_name="test-room"),
            public=lambda: {"id": "fixture"},
        )

        async def allocate(*args):
            manager.sessions[session.id] = session
            return session

        allocate_once = AsyncMock(side_effect=allocate)
        monkeypatch.setattr(manager, "create", allocate_once)
        monkeypatch.setattr("realtime.app.room_token", lambda *args: "fixture-token")
        transport = httpx.ASGITransport(app=make_pair(app, 18837))
        headers = {"x-start-key": "a" * 64}
        payload = {"avatar_id": "call07"}
        async with httpx.AsyncClient(
            transport=transport, base_url="http://127.0.0.1:18837"
        ) as client:
            async def post(path, data=payload):
                return await client.post(path, headers=headers, json=data)

            assert (await post("/api/sessions/recover")).status_code == 410
            first, repeated = await asyncio.gather(
                post("/api/sessions"), post("/api/sessions")
            )
            recovered = await post("/api/sessions/recover")
            assert first.status_code == repeated.status_code == recovered.status_code == 201
            assert recovered.json() == first.json() == repeated.json()
            assert (await post("/api/sessions/recover", {"avatar_id": "other"})).status_code == 409
            session.ended = True
            assert (await post("/api/sessions/recover")).status_code == 410
            allocate_once.assert_awaited_once()
        await app.state.voices.close()

    asyncio.run(scenario())
