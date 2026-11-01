import asyncio
import threading
import sys
from types import SimpleNamespace

import numpy as np

from realtime.continuous import ContinuousTimeline, ContinuousBatch, ContinuousRunner
from realtime.contracts import FRAME_SAMPLES, HEIGHT, WIDTH, ResponseBuffer
from realtime.gpu_client import GPUProcess
from realtime.streaming import GeneratedBatch
from realtime.worker import DemoWorker


def test_speech_transition_has_no_forced_frozen_preroll_or_silence_between_batches(
    tmp_path,
):
    from PIL import Image

    async def run():
        image = tmp_path / "reference.png"
        Image.new("RGB", (WIDTH, HEIGHT)).save(image)
        buffer = ResponseBuffer()
        buffer.replace("speech", np.full(57 * FRAME_SAMPLES, 1000, np.int16))
        worker = SimpleNamespace(
            closed=False,
            error=None,
            avatar_id="ref",
            buffer=buffer,
            catalog=SimpleNamespace(image=lambda _: image),
            batches=asyncio.Queue(2),
            generation_ready=asyncio.Event(),
            audio=SimpleNamespace(queued_duration=0),
            generating_epoch=None,
            frames=0,
            speech_frames=0,
            held_frames=0,
            idle_motion_frames=0,
            published_response_samples=0,
            first_chunk_published_at=None,
            first_media_ms=None,
            request_time=None,
        )
        runner = ContinuousRunner(worker)
        for _ in range(2):
            audio, start, speech = runner.timeline.take(buffer)
            frames = np.zeros(
                (len(audio.pcm) // FRAME_SAMPLES, HEIGHT, WIDTH, 3), np.uint8
            )
            worker.batches.put_nowait(
                ContinuousBatch(GeneratedBatch(audio, frames, 0.01), start, speech)
            )
            runner.pending_speech[buffer.epoch] += 1
        packets = []

        async def push(frame, pcm):
            packets.append(int(pcm[0]))
            if len(packets) == 57:
                worker.closed = True

        worker.sync = SimpleNamespace(push_pair=push)
        await asyncio.wait_for(runner.publish(), 2)
        assert worker.error is None
        assert packets == [1000] * 57
        assert runner.speech_preroll_frames == 0 and runner.speech_underflows == 0
        assert runner.published == 57 * FRAME_SAMPLES

    asyncio.run(run())


def test_visual_clock_survives_new_replies_and_preserves_response_offsets():
    timeline, buffer = ContinuousTimeline(), ResponseBuffer()
    idle, start, count = timeline.take(buffer)
    assert idle.sample_start == 0 and len(idle.pcm) == 25 * FRAME_SAMPLES
    assert count == 0 and not np.any(idle.pcm)
    epoch = buffer.begin("first")
    buffer.append("first", epoch, np.ones(40 * FRAME_SAMPLES, np.int16))
    first, start, count = timeline.take(buffer)
    assert first.sample_start == 25 * FRAME_SAMPLES and start == 0
    assert count == 32 * FRAME_SAMPLES
    buffer.cancel()
    idle, start, count = timeline.take(buffer)
    assert idle.sample_start == 57 * FRAME_SAMPLES and count == 0
    epoch = buffer.begin("second")
    buffer.append("second", epoch, np.full(1001, 2, np.int16))
    # Unfinished fractional frames must wait, not shift response frame IDs.
    partial, start, count = timeline.take(buffer)
    assert count == 0 and len(buffer.samples) == 1001
    buffer.finish("second", epoch)
    final, start, count = timeline.take(buffer)
    assert final.sample_start == 121 * FRAME_SAMPLES and start == 0 and count == 1001
    assert np.all(final.pcm[:1001] == 2) and not np.any(final.pcm[1001:])


def test_cancelled_video_has_no_speech_coverage():
    timeline, buffer = ContinuousTimeline(), ResponseBuffer()
    epoch = buffer.begin("speech")
    buffer.append("speech", epoch, np.ones(32 * FRAME_SAMPLES, np.int16))
    audio, start, speech = timeline.take(buffer)
    frames = np.zeros((25, HEIGHT, WIDTH, 3), np.uint8)
    item = ContinuousBatch(GeneratedBatch(audio, frames, 0.1), start, speech)
    assert item.speech_at(0, epoch)
    buffer.cancel()
    assert not item.speech_at(0, buffer.epoch)
    assert len(item.batch.frames) == 25


def test_queued_speech_keeps_response_active_but_cancelled_queue_does_not():
    buffer = ResponseBuffer()
    epoch = buffer.begin("speech")
    buffer.finish("speech", epoch)
    worker = SimpleNamespace(
        buffer=buffer,
        generating_epoch=None,
        publishing_epoch=None,
        continuous_motion=True,
        continuous_runner=ContinuousRunner(None),
        playing_epoch=None,
        response_audio_until=0,
    )
    worker.continuous_runner.pending_speech[epoch] = 1
    assert DemoWorker.response_active.fget(worker)
    buffer.cancel()
    assert not DemoWorker.response_active.fget(worker)


def test_unfinished_partial_speech_is_not_padded_with_an_invented_pause():
    timeline, buffer = ContinuousTimeline(), ResponseBuffer()
    epoch = buffer.begin("speech")
    part = np.ones(12 * FRAME_SAMPLES, np.int16)
    buffer.append("speech", epoch, part)
    idle, _, count = timeline.take(buffer)
    assert count == 0 and not np.any(idle.pcm)
    assert np.array_equal(buffer.samples, part) and buffer.position == 0
    buffer.append("speech", epoch, np.ones(27 * FRAME_SAMPLES, np.int16))
    speech, _, count = timeline.take(buffer)
    assert count == 32 * FRAME_SAMPLES and np.all(speech.pcm == 1)


def test_initial_speech_reserve_preserves_pcm_and_bypasses_completed_short_replies():
    timeline, buffer = ContinuousTimeline(), ResponseBuffer()
    epoch = buffer.begin("speech")
    buffer.append("speech", epoch, np.ones(50 * FRAME_SAMPLES, np.int16))
    minimum = 71 * FRAME_SAMPLES  # two 32-frame chunks plus seven lookahead frames
    idle, _, count = timeline.take(buffer, minimum_speech_samples=minimum)
    assert count == buffer.position == 0 and not idle.pcm.any()
    assert len(buffer.samples) == 50 * FRAME_SAMPLES
    buffer.append("speech", epoch, np.ones(22 * FRAME_SAMPLES, np.int16))
    speech, start, count = timeline.take(buffer, minimum_speech_samples=minimum)
    assert start == 0 and count == 32 * FRAME_SAMPLES
    assert np.all(speech.pcm == 1) and len(buffer.samples) == 40 * FRAME_SAMPLES
    buffer.cancel()
    buffer.replace("short", np.full(1001, 2, np.int16))
    short, start, count = timeline.take(buffer, minimum_speech_samples=minimum)
    assert start == 0 and count == 1001
    assert np.all(short.pcm[:1001] == 2) and not short.pcm[1001:].any()


def test_mid_reply_audio_wait_does_not_insert_an_idle_chunk():
    async def run():
        buffer = ResponseBuffer()
        epoch = buffer.begin("speech")
        buffer.position = 32 * FRAME_SAMPLES
        invoked = asyncio.Event()

        async def generate(audio, **kwargs):
            assert np.all(audio.pcm == 1)
            invoked.set()
            worker.closed = True
            return None

        worker = SimpleNamespace(
            closed=False,
            error=None,
            generation_ready=asyncio.Event(),
            batches=asyncio.Queue(1),
            buffer=buffer,
            gpu=SimpleNamespace(generate=generate),
            generating_epoch=None,
        )
        runner = ContinuousRunner(worker)
        task = asyncio.create_task(runner.generate())
        try:
            await asyncio.sleep(0.6)
            assert not invoked.is_set() and runner.timeline.position == 0
            buffer.append("speech", epoch, np.ones(32 * FRAME_SAMPLES, np.int16))
            worker.generation_ready.set()
            await asyncio.wait_for(task, 2)
            assert invoked.is_set() and not worker.error
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_thinking_keeps_idle_motion_and_does_not_consume_partial_reply():
    async def run():
        buffer = ResponseBuffer()
        epoch = buffer.begin("speech")
        partial = np.full(12 * FRAME_SAMPLES, 1000, np.int16)
        buffer.append("speech", epoch, partial)
        generated = []

        async def generate(audio, **kwargs):
            generated.append(audio)
            worker.closed = True
            return None

        worker = SimpleNamespace(
            closed=False,
            error=None,
            generation_ready=asyncio.Event(),
            batches=asyncio.Queue(1),
            buffer=buffer,
            gpu=SimpleNamespace(generate=generate),
            generating_epoch=None,
        )
        runner = ContinuousRunner(worker)
        await asyncio.wait_for(runner.generate(), 1)
        assert len(generated) == 1 and generated[0].response_id == "idle"
        assert not np.any(generated[0].pcm)
        assert buffer.position == 0 and np.array_equal(buffer.samples, partial)
        assert runner.timeline.position == 25 * FRAME_SAMPLES
        assert not worker.error

    asyncio.run(run())


def test_diagnostic_hashing_overlaps_next_gpu_job_and_drains_on_close():
    async def run():
        hashing, release = threading.Event(), threading.Event()
        calls = 0

        def hash_frames(frames):
            hashing.set()
            assert release.wait(2), "The next GPU job waited for diagnostic hashing"
            return {b"frame"}

        async def generate(audio, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                assert await asyncio.to_thread(hashing.wait, 1)
                release.set()
                worker.closed = True
                return None
            return GeneratedBatch(
                audio, np.zeros((25, HEIGHT, WIDTH, 3), np.uint8), 1.0
            )

        worker = SimpleNamespace(
            closed=False,
            error=None,
            generation_ready=asyncio.Event(),
            batches=asyncio.Queue(2),
            buffer=ResponseBuffer(),
            gpu=SimpleNamespace(
                generate=generate,
                info={"memory": {"last_stage_ms": {"denoise": 1000, "decode": 200}}},
            ),
            generating_epoch=None,
            generated_frames=0,
            generation_seconds=0,
            frame_hashes=set(),
            _hash_frames=hash_frames,
        )
        runner = ContinuousRunner(worker)
        assert runner.compute_ema == 1.24
        try:
            await asyncio.wait_for(runner.generate(), 3)
            assert not worker.error and calls == 2
            assert worker.frame_hashes == {b"frame"}
        finally:
            release.set()

    asyncio.run(run())


def test_real_ipc_does_not_reset_history_on_continuous_epoch_change(tmp_path):
    async def run():
        gpu = GPUProcess(
            [sys.executable, "-m", "realtime.tests.fake_gpu"], tmp_path / "gpu.log"
        )
        image = tmp_path / "reference.png"
        image.write_bytes(b"fake")
        timeline, buffer = ContinuousTimeline(), ResponseBuffer()
        try:
            await gpu.start()
            await gpu.open_session("one", image)
            for turn in range(4):
                epoch = buffer.begin(f"turn-{turn}")
                buffer.append(
                    buffer.response_id, epoch, np.ones(39 * FRAME_SAMPLES, np.int16)
                )
                audio, _, _ = timeline.take(buffer)
                assert await gpu.generate(audio, continuous=True) is not None
            await gpu.close_session("one")
            await gpu.open_session("two", image)
            audio, _, _ = ContinuousTimeline().take(ResponseBuffer())
            assert await gpu.generate(audio, continuous=True) is not None
        finally:
            await gpu.close()

    asyncio.run(run())


def test_startup_reserve_absorbs_a_slow_second_chunk_without_inserting_speech_gaps(
    tmp_path, monkeypatch
):
    from PIL import Image
    from livekit import rtc

    monkeypatch.setattr(rtc, "VideoFrame", lambda *args: SimpleNamespace(data=args[-1]))

    async def run():
        image = tmp_path / "reference.png"
        Image.new("RGB", (WIDTH, HEIGHT)).save(image)
        buffer = ResponseBuffer()
        buffer.replace("speech", np.full(89 * FRAME_SAMPLES, 1234, np.int16))
        packets, positions, calls = [], [], []

        async def generate(audio, **kwargs):
            calls.append(audio.sample_start)
            if len(calls) > 3:
                await asyncio.Event().wait()
            # The initial batch covers 1s; the next takes 1.12s to render.
            # Without a reserve this creates silence/a repeated pose mid-reply.
            await asyncio.sleep(0.01 if len(calls) == 1 else 1.12)
            frames = np.zeros((len(audio.pcm) // FRAME_SAMPLES, 1, 1, 3), np.uint8)
            return SimpleNamespace(audio=audio, frames=frames, compute_seconds=1.12)

        async def push(frame, pcm):
            packets.append(int(pcm[0]))
            positions.append(frame.playback_position)
            await asyncio.sleep(0.04)
            if len(packets) == 89:
                worker.closed = True

        worker = SimpleNamespace(
            closed=False,
            error=None,
            avatar_id="ref",
            buffer=buffer,
            settings=SimpleNamespace(avatar_startup_buffer_ms=480),
            catalog=SimpleNamespace(image=lambda _: image),
            batches=asyncio.Queue(2),
            generation_ready=asyncio.Event(),
            gpu=SimpleNamespace(generate=generate),
            audio=SimpleNamespace(queued_duration=0),
            sync=SimpleNamespace(push_pair=push),
            generating_epoch=None,
            frames=0,
            speech_frames=0,
            held_frames=0,
            idle_motion_frames=0,
            published_response_samples=0,
            first_chunk_published_at=None,
            first_media_ms=None,
            request_time=None,
            generated_frames=0,
            generation_seconds=0,
            frame_hashes=set(),
            _hash_frames=lambda _: set(),
        )
        runner = ContinuousRunner(worker)
        producer = asyncio.create_task(runner.generate())
        try:
            await asyncio.wait_for(runner.publish(), 7)
            assert not worker.error
            assert packets == [1234] * 89
            assert [p[1] for p in positions] == [i * FRAME_SAMPLES for i in range(89)]
            assert all(p[2] for p in positions)
            assert runner.speech_underflows == worker.held_frames == 0
            assert runner.startup_buffer_ms >= 450
            assert runner.speech_preroll_frames == 0
        finally:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)

    asyncio.run(run())


def test_compute_seed_is_read_after_the_resident_gpu_is_acquired():
    async def run():
        async def generate(audio, **kwargs):
            worker.closed = True
            return None

        worker = SimpleNamespace(
            gpu=None,
            closed=False,
            error=None,
            batches=asyncio.Queue(2),
            buffer=ResponseBuffer(),
            generation_ready=asyncio.Event(),
        )
        runner = ContinuousRunner(worker)
        worker.gpu = SimpleNamespace(
            generate=generate,
            info={"memory": {"last_stage_ms": {"denoise": 1050, "decode": 220}}},
        )
        await runner.generate()
        assert runner.compute_ema == 1.31

    asyncio.run(run())


def test_barge_in_discards_current_and_queued_talking_tails_without_reset(
    tmp_path, monkeypatch
):
    from PIL import Image
    from livekit import rtc

    monkeypatch.setattr(rtc, "VideoFrame", lambda *args: SimpleNamespace(data=args[-1]))

    async def run():
        image = tmp_path / "reference.png"
        Image.new("RGB", (WIDTH, HEIGHT)).save(image)
        buffer = ResponseBuffer()
        buffer.replace("speech", np.full(57 * FRAME_SAMPLES, 1200, np.int16))
        worker = SimpleNamespace(
            closed=False,
            error=None,
            avatar_id="ref",
            buffer=buffer,
            catalog=SimpleNamespace(image=lambda _: image),
            batches=asyncio.Queue(3),
            generation_ready=asyncio.Event(),
            audio=SimpleNamespace(queued_duration=0),
            generating_epoch=None,
            frames=0,
            speech_frames=0,
            held_frames=0,
            idle_motion_frames=0,
            published_response_samples=0,
            first_chunk_published_at=None,
            first_media_ms=None,
            request_time=None,
        )
        runner = ContinuousRunner(worker)
        for value in (10, 20, 30):
            audio, start, speech = runner.timeline.take(buffer)
            frames = np.full(
                (len(audio.pcm) // FRAME_SAMPLES, 1, 1, 3), value, np.uint8
            )
            worker.batches.put_nowait(
                ContinuousBatch(
                    SimpleNamespace(audio=audio, frames=frames, compute_seconds=0.01),
                    start,
                    speech,
                )
            )
            if speech:
                runner.pending_speech[audio.epoch] += 1
        packets = []

        async def push(frame, pcm):
            packets.append((frame.data[0], int(pcm[0])))
            if len(packets) == 5:
                buffer.cancel()  # interrupt a batch already being published
            if len(packets) == 8:
                worker.closed = True

        worker.sync = SimpleNamespace(push_pair=push)
        await asyncio.wait_for(runner.publish(), 2)
        assert worker.error is None
        assert packets == [(10, 1200)] * 5 + [(30, 0)] * 3
        assert runner.timeline.position == 89 * FRAME_SAMPLES
        assert not runner.pending_speech

    asyncio.run(run())
