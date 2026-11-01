import asyncio
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from realtime.config import Settings
from realtime.contracts import FRAME_SAMPLES, HEIGHT, WIDTH, ResponseBuffer
from realtime.gpu_client import GPUProcess
from realtime.gpu_pool import GPUWorkerPool
from realtime.idle import IdleClip, IdlePlayer, select_loop_end
from realtime.streaming import take_batch


def test_loop_selection_avoids_an_obviously_different_pose_and_keeps_source_immutable():
    frames = np.zeros((25, HEIGHT, WIDTH, 3), np.uint8)
    frames[4:14] = 10
    frames[20:] = 200
    original = frames.copy()
    end, seam = select_loop_end(frames)
    assert 15 <= end <= 20 and seam == 0
    clip = IdleClip(frames, 0.1)
    player = IdlePlayer(clip)
    last = frames[0]
    outputs = []
    for _ in range(80):
        last = player.next(last)
        outputs.append(last[0, 0, 0])
    assert len(set(outputs)) > 1
    assert np.array_equal(frames, original) and not frames.flags.writeable
    assert max(outputs) <= 10  # the mismatched tail is never spliced into the loop


def test_prepared_idle_reuses_only_matching_character_and_resets_speech_history(
    tmp_path, monkeypatch
):
    import realtime.idle as module

    monkeypatch.setattr(module, "IDLE_CHUNKS", 2)
    monkeypatch.setattr(module, "IDLE_FRAMES", 57)

    async def run():
        image = tmp_path / "image.bin"
        image.write_bytes(b"first character")

        def factory(_command, _log, **kwargs):
            return GPUProcess(
                [sys.executable, "-m", "realtime.tests.fake_gpu"],
                tmp_path / "gpu.log",
                **kwargs,
            )

        pool = GPUWorkerPool(
            Settings(
                avatar_catalog="research",
                avatar_idle_motion=True,
                avatar_continuous_motion=False,
            ),
            factory=factory,
        )
        try:
            gpu = await pool.acquire("one", image)
            first = gpu.idle_clip
            assert len(first.frames) == 57
            buffer = ResponseBuffer()
            buffer.replace("speech", np.ones(FRAME_SAMPLES, np.int16))
            result = await gpu.generate(take_batch(buffer))
            assert result.audio.sample_start == 0  # no silent preparation in the reply
            await pool.release("one")
            await pool.acquire("two", image)
            assert gpu.idle_clip is first
            await pool.release("two")
            image.write_bytes(b"changed character at the same path")
            await pool.acquire("three", image)
            assert gpu.idle_clip is not first
            await pool.release("three")
        finally:
            if pool.owner:
                await pool.release(pool.owner)
            await pool.close()
        assert gpu.idle_clip is None and gpu.closed

    asyncio.run(run())


def test_idle_preparation_failure_retires_half_prepared_worker(tmp_path, monkeypatch):
    async def run():
        gpu = GPUProcess(
            [sys.executable, "-m", "realtime.tests.fake_gpu"], tmp_path / "gpu.log"
        )
        image = tmp_path / "image.bin"
        image.write_bytes(b"reference")
        try:
            await gpu.start()
            await gpu.open_session("test", image)

            async def fail(*args):
                raise asyncio.CancelledError()

            monkeypatch.setattr(gpu, "generate", fail)
            with pytest.raises(asyncio.CancelledError):
                await gpu.prepare_idle(image)
            assert gpu.closed and gpu.idle_clip is None
        finally:
            await gpu.close()

    asyncio.run(run())


def test_idle_publisher_stays_silent_and_cannot_advance_speech_playback():
    from livekit import rtc
    from realtime.catalog import Catalog
    from realtime.contracts import AudioChunk
    from realtime.streaming import GeneratedBatch
    from realtime.worker import DemoWorker

    async def run():
        settings = Settings(avatar_catalog="research", avatar_idle_motion=True)
        worker = DemoWorker(
            "idle-publisher-test",
            "ref01",
            settings,
            Catalog(settings),
            model_id="avatarforever",
        )
        original_sync = worker.sync
        captures, audio = [], []

        class Sync:
            async def push(self, frame, timestamp):
                if isinstance(frame, rtc.VideoFrame):
                    captures.append(frame.playback_position)
                else:
                    audio.append(np.frombuffer(frame.data, np.int16).copy())

            async def clear_queue(self):
                pass

            async def aclose(self):
                pass

        worker.sync = Sync()
        worker.gpu = SimpleNamespace(
            idle_clip=IdleClip(np.zeros((17, HEIGHT, WIDTH, 3), np.uint8), 0.1)
        )

        async def until(predicate):
            async with asyncio.timeout(3):
                while not predicate():
                    await asyncio.sleep(0.01)

        task = asyncio.create_task(worker._publish_gpu())
        try:
            await until(lambda: worker.idle_motion_frames >= 3)
            assert all(not position[2] for position in captures)
            assert all(not np.any(part) for part in audio)
            assert worker.published_response_samples == 0
            assert (
                worker.first_chunk_published_at is None and not worker.response_active
            )
            chunk = AudioChunk(
                "reply",
                worker.buffer.epoch,
                0,
                np.ones(FRAME_SAMPLES * 2, np.int16),
                FRAME_SAMPLES * 2,
            )
            await worker.batches.put(
                GeneratedBatch(chunk, np.zeros((2, HEIGHT, WIDTH, 3), np.uint8), 0.1)
            )
            await until(
                lambda: worker.speech_frames == 2 and worker.idle_motion_frames >= 6
            )
            assert worker.published_response_samples == FRAME_SAMPLES * 2
            assert len([position for position in captures if position[2]]) == 2
            assert sum(np.count_nonzero(part) for part in audio) == FRAME_SAMPLES * 2
            assert worker.first_chunk_published_at is not None
            assert not worker.response_active and worker.error is None
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            worker.gpu = None
            worker.sync = original_sync
            await worker.close()

    asyncio.run(run())
