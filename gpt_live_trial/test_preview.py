import asyncio

import httpx

from realtime.app import create_app
from realtime.config import Settings
from gpt_live_trial.preview import make_pair


def test_main_and_trial_share_admission_and_only_trial_lists_marin():
    async def scenario():
        app = create_app(Settings(_env_file=None))
        pair = make_pair(app, 18837)
        transport = httpx.ASGITransport(app=pair)
        async with httpx.AsyncClient(transport=transport) as client:
            trial = "http://127.0.0.1:18837"
            control = "http://127.0.0.1:18821"
            result = await client.get(trial + "/api/voices")
            assert [row["id"] for row in result.json()] == ["marin"]
            assert (await client.get(trial + "/api/preview/provider")).json()[
                "model"
            ] == "gpt-live-1.1"
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
