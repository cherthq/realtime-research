"""Local scheduling/telemetry regression tests; no provider or GPU requests."""

import asyncio
from contextlib import asynccontextmanager
import math
import threading

import numpy as np

from realtime.catalog import Catalog
from realtime.config import Settings
from realtime.contracts import FRAME_SAMPLES, HEIGHT, WIDTH
from realtime.streaming import GeneratedBatch, LOOKAHEAD_SAMPLES
from realtime.worker import DemoWorker


class FakeGPU:
    async def generate(self, chunk):
        count = math.ceil(chunk.valid_samples / FRAME_SAMPLES)
        return GeneratedBatch(
            chunk, np.zeros((count, HEIGHT, WIDTH, 3), np.uint8), 0.01
        )


@asynccontextmanager
async def running_worker():
    settings = Settings(
        avatar_catalog="research",
    )
    worker = DemoWorker("latency-test", "ref01", settings, Catalog(settings))
    worker.gpu = FakeGPU()
    worker.generate_task = asyncio.create_task(worker._generate_gpu())
    try:
        yield worker
    finally:
        worker.generate_task.cancel()
        await asyncio.gather(worker.generate_task, return_exceptions=True)
        worker.gpu = None
        await worker.close()


async def pop(worker):
    batch = await asyncio.wait_for(worker.batches.get(), 2)
    # The production publisher emits this notification when consuming a batch.
    worker.generation_ready.set()
    return batch


def test_audio_notifications_keep_native_coverage_and_wake_on_final_partial():
    async def run():
        async with running_worker() as worker:
            epoch = worker.buffer.begin("reply")
            await asyncio.sleep(0)  # renderer now waiting on incomplete audio
            worker.buffer.append(
                "reply",
                epoch,
                np.ones(25 * FRAME_SAMPLES + LOOKAHEAD_SAMPLES - 1, np.int16),
            )
            await asyncio.sleep(0)
            assert worker.batches.empty() and worker.buffer.position == 0
            worker.buffer.append("reply", epoch, np.ones(1, np.int16))
            first = await pop(worker)
            assert first.audio.valid_samples == 25 * FRAME_SAMPLES
            worker.buffer.append("reply", epoch, np.ones(123, np.int16))
            assert worker.batches.empty()
            worker.buffer.finish("reply", epoch)
            last = await pop(worker)
            assert last.audio.valid_samples == LOOKAHEAD_SAMPLES + 123
            assert last.audio.sample_start == first.audio.valid_samples
            assert not last.audio.pcm[LOOKAHEAD_SAMPLES + 123 :].any()
            assert not worker.buffer.active and worker.error is None

    asyncio.run(run())


def test_full_queue_backpressure_resumes_and_cancel_discards_old_work():
    async def run():
        async with running_worker() as worker:
            worker.buffer.replace(
                "old", np.ones((25 + 32 * 3) * FRAME_SAMPLES, np.int16)
            )
            # Fill both queue slots without running the publisher.
            async with asyncio.timeout(2):
                while not worker.batches.full():
                    await asyncio.sleep(0.001)
            assert worker.buffer.position == (25 + 32) * FRAME_SAMPLES
            assert (await pop(worker)).audio.sample_start == 0
            async with asyncio.timeout(2):
                while not worker.batches.full():
                    await asyncio.sleep(0.001)
            assert worker.buffer.position == (25 + 32 * 2) * FRAME_SAMPLES
            # Exercise the real cancellation path while the producer is blocked.
            gpu, worker.gpu = worker.gpu, None
            await worker.cancel_media(count=False)
            worker.gpu = gpu
            assert worker.batches.empty()
            worker.buffer.replace("new", np.ones(101, np.int16))
            fresh = await pop(worker)
            assert fresh.audio.response_id == "new" and fresh.audio.sample_start == 0
            assert fresh.audio.epoch == worker.buffer.epoch
            assert worker.error is None

    asyncio.run(run())


def test_slow_metrics_do_not_hold_first_batch_or_block_event_loop():
    async def run():
        entered, release = threading.Event(), threading.Event()

        def slow_hash(frames):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("The publisher could not run during hashing")
            return {b"test-hash"}

        async with running_worker() as worker:
            worker._hash_frames = slow_hash
            worker.buffer.replace("reply", np.ones(101, np.int16))
            try:
                batch = await pop(worker)
                async with asyncio.timeout(1):
                    while not entered.is_set():
                        await asyncio.sleep(0.001)
                assert batch.audio.valid_samples == 101
                assert worker.generated_frames == 0  # telemetry still pending
                assert not release.is_set() and worker.error is None
            finally:
                release.set()
            async with asyncio.timeout(2):
                while worker.generated_frames == 0:
                    await asyncio.sleep(0.001)
            assert worker.frame_hashes == {b"test-hash"}

    asyncio.run(run())
