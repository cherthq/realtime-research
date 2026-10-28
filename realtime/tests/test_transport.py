"""Opt-in real local LiveKit integration; never requires cloud credentials."""

import asyncio
import json
import os
import time
import sys

import numpy as np
import pytest
from livekit import rtc

from realtime.catalog import Catalog
from realtime.config import Settings
from realtime.worker import DemoWorker, delete_room, room_token
from realtime.gpu_client import GPUProcess
from realtime.gpu_pool import GPUWorkerPool


async def eventually(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("Condition did not become true")
        await asyncio.sleep(0.02)


@pytest.mark.skipif(
    os.environ.get("RUN_LIVEKIT_TESTS") != "1",
    reason="Set RUN_LIVEKIT_TESTS=1 with local LiveKit running",
)
def test_real_webrtc_frames_audio_interrupt_and_disconnect():
    async def run():
        config = Settings()
        worker = DemoWorker(
            f"integration-{time.time_ns()}", "ref01", config, Catalog(config)
        )
        receiver = rtc.Room()
        video = asyncio.Queue()
        audio = asyncio.Queue()
        notices = []

        @receiver.on("data_received")
        def data_received(packet):
            if (
                packet.topic == "chert-control"
                and packet.participant.identity == "avatar"
            ):
                notices.append(json.loads(packet.data))

        @receiver.on("track_subscribed")
        def subscribed(track, *_):
            (video if track.kind == rtc.TrackKind.KIND_VIDEO else audio).put_nowait(
                track
            )

        video_stream = audio_stream = None
        try:
            await worker.start()
            await receiver.connect(
                config.livekit_url,
                room_token(config, worker.room_name, "browser"),
                options=rtc.RoomOptions(single_peer_connection=False),
            )
            video_stream = rtc.VideoStream(
                await asyncio.wait_for(video.get(), 10), capacity=2
            )
            audio_stream = rtc.AudioStream(
                await asyncio.wait_for(audio.get(), 10), capacity=2
            )
            for _ in range(4):
                event = await asyncio.wait_for(video_stream.__anext__(), 5)
                assert (event.frame.width, event.frame.height) == (640, 480)
                assert event.metadata is not None and event.metadata.HasField(
                    "frame_id"
                )
            await worker.play("sync_marker")
            received = await asyncio.wait_for(audio_stream.__anext__(), 5)
            assert received.frame.sample_rate == 48_000
            epoch = worker.buffer.epoch
            await worker.interrupt()
            assert worker.buffer.epoch > epoch and not worker.buffer.active
            await eventually(
                lambda: {"type": "cancel", "epoch": worker.buffer.epoch} in notices
            )
            assert worker.last_cancel_ms < 300
            assert worker.error is None and worker.frames >= 4
        finally:
            if video_stream:
                await video_stream.aclose()
            if audio_stream:
                await audio_stream.aclose()
            await receiver.disconnect()
            await worker.close()
            await delete_room(config, worker.room_name)
        assert worker.closed and worker.media_task.done()

    asyncio.run(run())


@pytest.mark.skipif(
    os.environ.get("RUN_LIVEKIT_TESTS") != "1", reason="Local LiveKit required"
)
def test_two_calls_keep_model_process_and_release_each_room(tmp_path):
    async def run():
        config = Settings()

        def factory(command, log, **kwargs):
            return GPUProcess(
                [sys.executable, "-m", "realtime.tests.fake_gpu"],
                tmp_path / "pool.log",
                **kwargs,
            )

        pool = GPUWorkerPool(config, factory=factory)
        pid = None
        try:
            for avatar in ("ref01", "ref02"):
                worker = DemoWorker(
                    f"reuse-{time.time_ns()}",
                    avatar,
                    config,
                    Catalog(config),
                    model_id="avatarforever",
                    gpu_pool=pool,
                )
                try:
                    await worker.start()
                    if pid is None:
                        pid = worker.gpu.process.pid
                    assert worker.gpu.process.pid == pid
                    worker.buffer.replace("turn", np.ones(1001, dtype=np.int16))
                    await eventually(lambda: worker.published_response_samples == 1001)
                finally:
                    await worker.close()
                    await delete_room(config, worker.room_name)
                assert pool.owner is None and not pool.worker.closed
                assert worker.gpu.session_id is None
        finally:
            await pool.close()

    asyncio.run(run())


@pytest.mark.skipif(
    os.environ.get("RUN_LIVEKIT_TESTS") != "1", reason="Local LiveKit required"
)
def test_chunk_publisher_over_webrtc_has_exact_coverage_cancel_and_stall(tmp_path):
    """Real transport and process boundary, with synthetic GPU output only."""

    async def run():
        config = Settings()
        worker = DemoWorker(
            f"chunk-{time.time_ns()}",
            "ref01",
            config,
            Catalog(config),
            model_id="avatarforever",
        )
        worker.gpu = GPUProcess(
            [sys.executable, "-m", "realtime.tests.fake_gpu"], tmp_path / "gpu.log"
        )
        receiver = rtc.Room()
        tracks = asyncio.Queue()

        @receiver.on("track_subscribed")
        def subscribed(track, *_):
            tracks.put_nowait(track)

        streams, readers, dimensions, pcm = [], [], [], []

        async def read_video(stream):
            async for event in stream:
                dimensions.append((event.frame.width, event.frame.height))

        async def read_audio(stream):
            async for event in stream:
                pcm.append(np.frombuffer(event.frame.data, dtype=np.int16).copy())

        try:
            await worker.start()
            pid = worker.gpu.process.pid
            await receiver.connect(
                config.livekit_url,
                room_token(config, worker.room_name, "browser"),
                options=rtc.RoomOptions(single_peer_connection=False),
            )
            for _ in range(2):
                track = await asyncio.wait_for(tracks.get(), 10)
                if track.kind == rtc.TrackKind.KIND_VIDEO:
                    stream = rtc.VideoStream(track, capacity=2)
                    readers.append(asyncio.create_task(read_video(stream)))
                else:
                    stream = rtc.AudioStream(track, capacity=2)
                    readers.append(asyncio.create_task(read_audio(stream)))
                streams.append(stream)
            # Idle media must keep both RTP tracks alive without claiming any
            # generated speech. A one-second held-video heartbeat with no audio
            # previously left receiver playout discontinuous between replies.
            await eventually(lambda: len(pcm) >= 30 and len(dimensions) >= 15)
            assert (
                worker.generated_frames == 0 and worker.published_response_samples == 0
            )
            assert max(np.max(np.abs(part.astype(int))) for part in pcm) < 10
            pcm.clear()
            # Partial final chunk must publish exactly 49,001 valid PCM samples.
            samples = (
                np.sin(np.arange(49_001) * 2 * np.pi * 440 / 48_000) * 9000
            ).astype(np.int16)
            worker.buffer.replace("coverage", samples)
            await eventually(lambda: worker.published_response_samples == len(samples))
            await eventually(lambda: bool(pcm) and bool(dimensions))
            assert set(dimensions) == {(640, 480)}
            assert max(np.max(np.abs(part.astype(int))) for part in pcm) > 1000
            assert worker.frames == 26 and worker.generated_frames == 26
            # This fixture repeats one constant frame. Published FPS must not
            # make it look like 26 distinct generated images.
            assert worker.snapshot()["distinct_generated_frames"] == 1
            assert worker.snapshot()["exact_duplicate_frames"] == 25
            await eventually(lambda: not worker.response_active)
            await asyncio.sleep(0.2)
            assert worker.stalls == 0
            before_idle = len(pcm)
            await eventually(lambda: len(pcm) >= before_idle + 75)
            assert worker.frames == 26 and worker.generated_frames == 26
            # A GPU call is still busy when interruption is handled by the CPU.
            await worker.interrupt(count=False)
            worker.buffer.replace("slow", np.ones(48_000, dtype=np.int16))
            await eventually(lambda: worker.generating_epoch == worker.buffer.epoch)
            mark = time.monotonic()
            await worker.interrupt()
            assert time.monotonic() - mark < 0.3
            assert not worker.response_active
            await eventually(lambda: worker.generating_epoch is None)
            assert worker.gpu.process.pid == pid
            # A mid-response input/render stall must surface, not hold forever.
            epoch = worker.buffer.begin("stall")
            worker.buffer.append("stall", epoch, np.ones(48_000, dtype=np.int16))
            await eventually(
                lambda: worker.playing_epoch == epoch or worker.error is not None
            )
            assert worker.error is None, worker.snapshot()
            assert worker.response_active
            await eventually(lambda: worker.stalls == 1)
            assert "stalled" in worker.error and worker.status == "error"
        finally:
            for task in readers:
                task.cancel()
            await asyncio.gather(*readers, return_exceptions=True)
            for stream in streams:
                await stream.aclose()
            await receiver.disconnect()
            await worker.close()
            await delete_room(config, worker.room_name)
        assert worker.gpu.process.returncode is not None

    asyncio.run(run())
