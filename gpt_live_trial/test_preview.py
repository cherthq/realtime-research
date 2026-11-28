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


def test_opt_in_config_reaches_worker_without_public_catalog_changes(tmp_path, monkeypatch):
    import json
    from PIL import Image
    from realtime.contracts import ResponseBuffer
    from realtime.tests.test_core import FakeWorker
    from gpt_live_trial.agent import INSTRUCTIONS

    class Worker(FakeWorker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.buffer = ResponseBuffer()

    async def scenario():
        monkeypatch.setattr('gpt_live_trial.preview.ROOT', tmp_path)
        monkeypatch.setattr('realtime.app.ROOT', tmp_path)
        monkeypatch.setattr('realtime.app.DemoWorker', Worker)
        monkeypatch.setattr('realtime.app.delete_room', AsyncMock())
        monkeypatch.setattr('realtime.app.room_token', lambda *args: 'test-room-token')
        image = tmp_path / 'reference.png'
        Image.new('RGB', (64, 48), 'red').save(image)
        token = 'configuration-test-' + 'a' * 32
        config_path = tmp_path / 'runs/avatar-config/runtime.json'
        config_path.parent.mkdir(parents=True)
        config_path.write_text(json.dumps(dict(avatar_id='call07', admin_token=token)))
        app = create_app(Settings(_env_file=None, avatarforever_enabled=True,
                                  openai_api_key='test-only', demo_access_token='ordinary-preview'))
        manager = app.state.manager
        monkeypatch.setattr(manager.catalog, 'image', lambda _: image)
        pair = make_pair(app, 18837)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=pair), base_url='http://127.0.0.1:18837') as client:
                admin = {'Authorization': 'Bearer ' + token}
                assert (await client.get('/api/avatar-config')).status_code == 401
                data = (await client.get('/api/avatar-config', headers=admin)).json()
                assert data['current']['prompt'] == INSTRUCTIONS
                assert (await client.get('http://127.0.0.1:18821/api/avatar-config', headers=admin)).status_code != 200
                headers = {'x-start-key': 'a'*64, 'Authorization': 'Bearer ordinary-preview'}
                body = dict(avatar_id='call07', model_id='avatarforever', mode='voice_diagnostic')
                first = await client.post('/api/sessions', headers=headers, json=body)
                assert first.status_code == 201, first.text
                session = manager.sessions[first.json()['id']]
                assert session.worker.voice.instructions == INSTRUCTIONS
                assert session.worker.catalog.image('call07') != image
                assert manager.catalog.image('call07') == image
                updated = await client.post('/api/avatar-config', headers=admin, json=dict(prompt='New instructions', expectedRevision=1))
                assert updated.status_code == 200, updated.text
                assert session.worker.voice.instructions == INSTRUCTIONS
                recovered = await client.post('/api/sessions/recover', headers=headers, json=body)
                assert recovered.status_code == 201
                assert recovered.json()['id'] == session.id
                assert 'configuration_version' not in json.dumps(recovered.json())
                await manager.close(session)
                second = await client.post('/api/sessions', headers={**headers, 'x-start-key':'b'*64}, json=body)
                assert second.status_code == 201, second.text
                assert manager.sessions[second.json()['id']].worker.voice.instructions == 'New instructions'
    asyncio.run(scenario())
