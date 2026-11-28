import asyncio
from builtins import ExceptionGroup
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from realtime.startup import StartupTrace, start_parallel


def test_connections_overlap_and_video_starts_before_voice_but_admission_waits():
    async def run():
        entered = {
            name: asyncio.Event()
            for name in ("renderer", "transport", "voice", "recorder")
        }
        release = {name: asyncio.Event() for name in entered}
        published = asyncio.Event()
        recorder = SimpleNamespace(avatar_audio_ready=asyncio.Event())

        async def stage(name):
            entered[name].set()
            await release[name].wait()

        async def connect():
            await stage("transport")
            recorder.avatar_audio_ready.set()

        worker = SimpleNamespace(
            startup=StartupTrace("test"),
            _prepare_renderer=lambda: stage("renderer"),
            _connect_media=connect,
            _start_media=published.set,
            voice=SimpleNamespace(start=lambda: stage("voice")),
            call_recorder=recorder,
        )
        recorder.start = lambda: stage("recorder")
        task = asyncio.create_task(start_parallel(worker))
        await asyncio.wait_for(asyncio.gather(*(e.wait() for e in entered.values())), 1)
        release["renderer"].set()
        release["transport"].set()
        await asyncio.wait_for(published.wait(), 1)
        assert not task.done()  # Never hand caller credentials out before recording.
        release["recorder"].set()
        release["voice"].set()
        await asyncio.wait_for(task, 1)
        events = worker.startup.events
        assert events["media_started"] <= events["voice_ready"]
        assert "recorder_ready" in events

    asyncio.run(run())


@pytest.mark.parametrize("cancel", [False, True])
def test_start_failure_and_hangup_drain_every_connection_task(cancel):
    async def run():
        entered = asyncio.Event()
        drained = []
        fail = asyncio.Event()

        async def renderer():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                drained.append("renderer")

        async def voice():
            try:
                await fail.wait()
                raise RuntimeError("connection refused")
            finally:
                drained.append("voice")

        async def media():
            try:
                await asyncio.Event().wait()
            finally:
                drained.append("media")

        worker = SimpleNamespace(
            startup=StartupTrace("test"),
            _prepare_renderer=renderer,
            _connect_media=media,
            _start_media=lambda: pytest.fail("Must not publish after failure"),
            voice=SimpleNamespace(start=voice),
        )
        task = asyncio.create_task(start_parallel(worker))
        await entered.wait()
        await asyncio.sleep(0)
        if cancel:
            task.cancel()
        else:
            fail.set()
        with pytest.raises(asyncio.CancelledError if cancel else ExceptionGroup):
            await task
        assert set(drained) == {"renderer", "voice", "media"}

    asyncio.run(run())


def test_recording_failure_does_not_block_call_and_is_visible(tmp_path):
    async def run():
        saved = []

        def update(*args):
            saved.append(args)

        recorder = SimpleNamespace(
            start=AsyncMock(side_effect=OSError("offline")),
            id="test",
            logs=SimpleNamespace(update=update),
        )
        worker = SimpleNamespace(
            startup=StartupTrace("test"),
            _prepare_renderer=AsyncMock(),
            _connect_media=AsyncMock(),
            _start_media=lambda: None,
            voice=SimpleNamespace(start=AsyncMock()),
            call_recorder=recorder,
        )
        await start_parallel(worker)
        assert "recorder_failed" in worker.startup.events
        assert saved[0][3] == "Recording connection could not start"
        worker.voice.start.assert_awaited_once()

    asyncio.run(run())


def test_trace_is_monotonic_offsets_and_first_event_only():
    now = [5.0]
    trace = StartupTrace("private-id", lambda: now[0])
    now[0] = 5.25
    trace.mark("first_video_captured")
    now[0] = 6
    trace.mark("first_video_captured")
    assert trace.snapshot()["events_ms"] == {"first_video_captured": 250.0}
    assert "private-id" not in str(trace.snapshot())


def test_cold_renderer_does_not_open_paid_voice_until_ready():
    async def run():
        began, ready = asyncio.Event(), asyncio.Event()

        async def load():
            began.set()
            await ready.wait()

        voice = AsyncMock()
        worker = SimpleNamespace(
            startup=StartupTrace("test"),
            _prepare_renderer=load,
            _connect_media=AsyncMock(),
            _start_media=lambda: None,
            voice=SimpleNamespace(start=voice),
            gpu_pool=SimpleNamespace(snapshot=lambda: {"loaded": False}),
        )
        task = asyncio.create_task(start_parallel(worker))
        await began.wait()
        voice.assert_not_awaited()
        worker._connect_media.assert_not_awaited()
        ready.set()
        await task
        voice.assert_awaited_once()

    asyncio.run(run())


def test_finished_start_does_not_leave_caller_task_owned_by_worker():
    from realtime.worker import DemoWorker

    async def run():
        worker = DemoWorker.__new__(DemoWorker)
        worker.startup = StartupTrace("test")
        worker.parallel_startup = True
        worker.voice = None
        worker._prepare_renderer = AsyncMock()
        worker._connect_media = AsyncMock()
        worker._start_media = lambda: None
        await worker.start()
        assert worker.startup_task is None
        assert "session_ready" in worker.startup.events

    asyncio.run(run())


def test_hangup_during_real_worker_start_drains_tasks_then_releases_lease():
    from realtime.worker import DemoWorker
    from realtime.contracts import ResponseBuffer
    from realtime.transcripts import TranscriptLedger

    async def run():
        worker = DemoWorker.__new__(DemoWorker)
        worker.id = "early-hangup"
        worker.startup = StartupTrace(worker.id)
        worker.parallel_startup = True
        worker.startup_task = None
        worker.closed = False
        worker._closing = None
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        async def preparing():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        worker._prepare_renderer = preparing
        worker._connect_media = AsyncMock(side_effect=lambda: None)
        worker._start_media = lambda: pytest.fail("No media before renderer readiness")
        worker.voice = SimpleNamespace(start=AsyncMock(), close=AsyncMock())
        worker.gpu = None

        async def release(_):
            assert cancelled.is_set()
            worker.gpu_pool.owner = None

        worker.gpu_pool = SimpleNamespace(
            owner=worker.id,
            snapshot=lambda: {"loaded": True},
            release=AsyncMock(side_effect=release),
        )
        worker.transcripts = TranscriptLedger()
        worker.buffer = ResponseBuffer()
        worker.camera = SimpleNamespace(close=AsyncMock())
        worker.sync = SimpleNamespace(clear_queue=AsyncMock(), aclose=AsyncMock())
        worker.room = SimpleNamespace(disconnect=AsyncMock())
        worker.audio = SimpleNamespace(aclose=AsyncMock())
        worker.video = SimpleNamespace(aclose=AsyncMock())
        worker.media_task = worker.generate_task = worker.control_task = (
            worker.caption_task
        ) = None
        worker.mic_tasks = set()
        worker.renderer = None
        starting = asyncio.create_task(worker.start())
        await entered.wait()
        await asyncio.wait_for(worker.close(), 2)
        assert starting.cancelled()
        assert worker.gpu_pool.owner is None
        worker.gpu_pool.release.assert_awaited_once_with(worker.id)
        worker.voice.close.assert_awaited_once()
        assert worker.status == "ended"

    asyncio.run(run())


def test_recorder_subscription_acknowledgment_is_required_before_admission():
    async def run():
        media = asyncio.Event()
        recorder = SimpleNamespace(
            start=AsyncMock(), avatar_audio_ready=asyncio.Event()
        )
        worker = SimpleNamespace(
            startup=StartupTrace("test"),
            _prepare_renderer=AsyncMock(),
            _connect_media=AsyncMock(),
            _start_media=media.set,
            voice=SimpleNamespace(start=AsyncMock()),
            call_recorder=recorder,
        )
        task = asyncio.create_task(start_parallel(worker))
        await media.wait()
        assert not task.done()
        recorder.avatar_audio_ready.set()
        await asyncio.wait_for(task, 1)
        assert "recorder_ready" in worker.startup.events

    asyncio.run(run())
