import asyncio

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
