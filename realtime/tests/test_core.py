import asyncio
import importlib
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from realtime.catalog import Catalog
from realtime.config import Settings
from realtime.contracts import FRAME_SAMPLES, AudioChunk, MediaChunk, ResponseBuffer
from realtime.mock import MockRenderer


def test_partial_audio_has_exact_valid_length_and_cancellation_invalidates_old_work():
    buffer = ResponseBuffer()
    samples = np.arange(FRAME_SAMPLES + 17, dtype=np.int16)
    buffer.replace("one", samples)
    first, final = buffer.next(), buffer.next()
    assert first.valid_samples == FRAME_SAMPLES
    assert final.valid_samples == 17 and final.sample_start == FRAME_SAMPLES
    assert np.array_equal(final.pcm[:17], samples[-17:])
    assert not np.any(final.pcm[17:])
    assert not buffer.active
    buffer.cancel()
    assert first.epoch != buffer.epoch
    assert buffer.next().valid_samples == 0
    buffer.replace("two", np.ones(42, dtype=np.int16))
    assert buffer.next().sample_start == 0


def test_bound_and_geometry_are_enforced():
    with pytest.raises(ValueError, match="twenty-second"):
        ResponseBuffer().replace("too-long", np.zeros(48_000 * 21, dtype=np.int16))
    with pytest.raises(ValueError, match="mono PCM16"):
        AudioChunk("x", 0, 0, np.zeros(10, dtype=np.float32), 10)
    chunk = AudioChunk("x", 0, 0, np.zeros(1920, dtype=np.int16), 10)
    with pytest.raises(ValueError, match="640x480"):
        MediaChunk(chunk, np.zeros((720, 416, 3), dtype=np.uint8))
    packet = MockRenderer(None).render(chunk)
    assert packet.rgb.shape == (480, 640, 3)


def test_sync_fixture_has_known_silence_and_tone_windows():
    samples = Catalog(Settings()).audio("sync_marker")
    assert len(samples) == 4 * 48_000
    assert not np.any(samples[:24_960])
    assert np.max(np.abs(samples[24_960:29_760])) > 6000
    assert not np.any(samples[29_760:96_000])
    continuous = Catalog(Settings()).audio("interrupt_marker")
    assert len(continuous) == 4 * 48_000
    assert np.all(np.max(np.abs(continuous.reshape(-1, 1920)), axis=1) > 6000)


class FakeWorker:
    def __init__(self, *args, **kwargs):
        self.status = "preparing"
        self.room_name = "test-room"
        self.connected_user = True
        self.last_presence = time.monotonic()
        self.error = None
        self.closed = False
        self.state_changed = asyncio.Event()

    async def start(self):
        self.status = "listening"

    async def close(self):
        self.closed = True
        self.status = "ended"

    async def play(self, name):
        self.status = "responding"

    async def interrupt(self):
        self.status = "listening"

    def snapshot(self):
        return {"state": self.status}


@pytest.fixture
def client(monkeypatch):
    module = importlib.import_module("realtime.app")
    monkeypatch.setattr(module, "DemoWorker", FakeWorker)

    async def delete(*_):
        return None

    monkeypatch.setattr(module, "delete_room", delete)
    with TestClient(module.create_app(Settings())) as client:
        yield client


def test_single_lease_authorization_and_idempotent_cleanup(client):
    created = client.post("/api/sessions", json={}).json()
    path = f"/api/sessions/{created['id']}"
    key = {"X-Session-Key": created["session_key"]}
    assert client.post("/api/sessions", json={}).status_code == 409
    assert client.get(path).status_code == 404
    assert client.delete(path, headers={"X-Session-Key": "wrong"}).status_code == 404
    assert (
        client.post(
            path + "/play", headers=key, json={"scenario_id": "s01_greeting"}
        ).status_code
        == 200
    )
    assert (
        client.post(path + "/interrupt", headers=key).json()["metrics"]["state"]
        == "listening"
    )
    assert client.delete(path, headers=key).json()["ended"]
    assert client.delete(path, headers=key).status_code == 200
    assert (
        client.post(
            path + "/play", headers=key, json={"scenario_id": "s01_greeting"}
        ).status_code
        == 409
    )
    assert client.post("/api/sessions", json={"avatar_id": "ref02"}).status_code == 201


def test_no_silent_model_fallback_and_no_cross_site_control(client):
    assert (
        client.post("/api/sessions", json={"model_id": "avatarforever"}).status_code
        == 409
    )
    assert (
        client.post("/api/sessions", json={"avatar_id": "../secret"}).status_code == 422
    )
    assert (
        client.post(
            "/api/sessions", json={}, headers={"Origin": "https://example.com"}
        ).status_code
        == 403
    )
    assert (
        client.get("/api/avatars", headers={"Host": "evil.example"}).status_code == 400
    )


def test_failed_start_releases_lease(client, monkeypatch):
    async def fail(self):
        raise RuntimeError("injected connection error")

    monkeypatch.setattr(FakeWorker, "start", fail)
    assert client.post("/api/sessions", json={}).status_code == 503
    assert client.app.state.manager.active is None
    assert all(
        session.worker.closed for session in client.app.state.manager.sessions.values()
    )


def test_live_call_prepares_voice_before_reserving_worker(client, monkeypatch):
    from pydantic import SecretStr
    from realtime.voices import VoicePreparationError

    voices = client.app.state.voices
    manager = client.app.state.manager
    voices.settings.openai_api_key = SecretStr("test-only-no-network")

    async def fail(*_):
        assert manager.active is None
        raise VoicePreparationError("Prepare the voice catalog first")

    monkeypatch.setattr(voices, "resolve_for_call", fail)
    assert (
        client.post("/api/sessions", json={"mode": "voice_diagnostic"}).status_code
        == 409
    )
    assert manager.active is None and not manager.sessions

    async def prepared(*_):
        assert manager.active is None
        return {"profile": {"id": "alloy"}, "mode": "image_cast", "preparation_ms": 12}

    monkeypatch.setattr(voices, "resolve_for_call", prepared)
    response = client.post("/api/sessions", json={"mode": "voice_diagnostic"})
    assert response.status_code == 201
    assert response.json()["voice_selection"]["preparation_ms"] == 12


def test_cancelled_end_request_keeps_lease_until_cleanup_finishes(
    monkeypatch, tmp_path
):
    module = importlib.import_module("realtime.app")

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        closes, deleted = [], []

        class SlowWorker(FakeWorker):
            async def close(self):
                closes.append(self)
                entered.set()
                await release.wait()
                await super().close()

        async def delete(*_):
            assert closes[0].closed
            deleted.append(True)

        monkeypatch.setattr(module, "DemoWorker", SlowWorker)
        monkeypatch.setattr(module, "delete_room", delete)
        monkeypatch.setattr(module, "ROOT", tmp_path)
        config = Settings()
        manager = module.SessionManager(config, Catalog(config))
        session = await manager.create("ref01")
        request = asyncio.create_task(manager.close(session))
        await asyncio.wait_for(entered.wait(), 1)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert manager.active == session.id
        assert not session.close_task.done()
        with pytest.raises(module.HTTPException) as busy:
            await manager.create("ref02")
        assert busy.value.status_code == 409
        second = asyncio.create_task(manager.close(session, "second_request"))
        await asyncio.sleep(0)
        assert not second.done()
        release.set()
        await asyncio.wait_for(second, 2)
        await manager.close(session)
        assert len(closes) == len(deleted) == 1
        assert manager.active is None
        assert session.end_reason == "ended_by_user"
        assert (
            tmp_path / "runs/realtime/sessions" / session.id / "report.json"
        ).is_file()

    asyncio.run(run())


def test_worker_teardown_survives_request_cancellation_and_queue_error():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from realtime.worker import DemoWorker

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        async def release_gpu(_):
            entered.set()
            await release.wait()

        # Exercise ownership without initializing native resources or a GPU.
        worker = DemoWorker.__new__(DemoWorker)
        worker.id, worker.closed, worker._closing = "cleanup-test", False, None
        worker.buffer = ResponseBuffer()
        worker.sync = SimpleNamespace(
            clear_queue=AsyncMock(side_effect=RuntimeError("queue failure")),
            aclose=AsyncMock(),
        )
        worker.gpu, worker.voice, worker.generate_task, worker.media_task = (
            None,
            None,
            None,
            None,
        )
        worker.control_task = None
        worker.gpu_pool = SimpleNamespace(release=AsyncMock(side_effect=release_gpu))
        worker.mic_tasks = set()
        worker.room = SimpleNamespace(disconnect=AsyncMock())
        worker.audio = SimpleNamespace(aclose=AsyncMock())
        worker.video = SimpleNamespace(aclose=AsyncMock())
        worker.renderer = object()
        first = asyncio.create_task(worker.close())
        await asyncio.wait_for(entered.wait(), 1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(worker.close())
        await asyncio.sleep(0)
        assert not second.done()
        assert worker.status == "closing"
        release.set()
        await asyncio.wait_for(second, 1)
        await worker.close()
        worker.gpu_pool.release.assert_awaited_once_with(worker.id)
        worker.sync.aclose.assert_awaited_once()
        worker.room.disconnect.assert_awaited_once()
        worker.audio.aclose.assert_awaited_once()
        worker.video.aclose.assert_awaited_once()
        assert worker.status == "ended" and worker.renderer is None

    asyncio.run(run())
