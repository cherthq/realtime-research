import asyncio
import io
import os
import struct
import sys

import numpy as np
import pytest

from realtime.contracts import FRAME_SAMPLES, ResponseBuffer
from realtime.gpu.wire import MAX_BODY, MAX_HEADER, encode, read, write
from realtime.gpu_client import GPUProcess
from realtime.streaming import take_batch, LOOKAHEAD_SAMPLES
from realtime.config import Settings
from realtime.gpu_pool import GPUWorkerPool


@pytest.mark.parametrize("shared", [False, True])
def test_frame_transport_preserves_previous_output_while_buffer_is_reused(
    tmp_path, shared
):
    if shared and os.name != "posix":
        pytest.skip("Inherited shared descriptors require POSIX")

    async def run():
        gpu = GPUProcess(
            [sys.executable, "-m", "realtime.tests.fake_gpu"],
            tmp_path / "transport.log",
            shared_frames=shared,
        )
        image = tmp_path / "ref.bin"
        image.write_bytes(b"reference")
        try:
            await gpu.start()
            assert gpu.info["frame_transport"] == ("mmap" if shared else "pipe")
            await gpu.open_session("transport", image)
            buffer = ResponseBuffer()
            buffer.replace("first", np.ones(25 * FRAME_SAMPLES, np.int16))
            first = await gpu.generate(take_batch(buffer))
            old_epoch = buffer.epoch
            buffer.replace("second", np.ones(25 * FRAME_SAMPLES, np.int16))
            second = await gpu.generate(take_batch(buffer))
            assert np.all(first.frames == old_epoch)
            assert np.all(second.frames == buffer.epoch)
        finally:
            await gpu.close()
        assert gpu.frame_buffer is None and gpu.frame_file is None

    asyncio.run(run())


def test_paced_audio_isolation_and_final_padding():
    buffer = ResponseBuffer()
    epoch = buffer.begin("one")
    first = np.arange(25 * FRAME_SAMPLES, dtype=np.int16)
    buffer.append("one", epoch, first[:-1])
    assert take_batch(buffer) is None and buffer.position == 0
    buffer.append("one", epoch, first[-1:])
    assert take_batch(buffer) is None
    tail = np.arange(32 * FRAME_SAMPLES + LOOKAHEAD_SAMPLES + 1001, dtype=np.int16)
    buffer.append("one", epoch, tail[:LOOKAHEAD_SAMPLES])
    a = take_batch(buffer)
    assert np.array_equal(a.pcm, first) and a.valid_samples == len(first)
    assert np.array_equal(a.lookahead, tail[:LOOKAHEAD_SAMPLES])
    assert take_batch(buffer) is None
    buffer.append("one", epoch, tail[LOOKAHEAD_SAMPLES:])
    b = take_batch(buffer)
    assert b.sample_start == len(first) and len(b.pcm) == 32 * FRAME_SAMPLES
    assert take_batch(buffer) is None
    buffer.finish("one", epoch)
    c = take_batch(buffer)
    remaining = LOOKAHEAD_SAMPLES + 1001
    assert c.valid_samples == remaining and not c.pcm[remaining:].any()
    assert not c.lookahead.any()
    assert np.array_equal(
        np.concatenate((a.pcm, b.pcm, c.pcm[:remaining])), np.concatenate((first, tail))
    )
    buffer.cancel()
    assert not buffer.append("one", epoch, first)
    assert take_batch(buffer) is None


def test_wire_rejects_invalid_or_truncated_allocations():
    assert read(io.BytesIO(encode({"type": "hello"}, b"123"))) == (
        {"type": "hello", "body_bytes": 3},
        b"123",
    )
    with pytest.raises(ValueError):
        read(io.BytesIO(struct.pack("!I", MAX_HEADER + 1)))
    with pytest.raises(EOFError):
        read(io.BytesIO(encode({"type": "hello"}, b"123")[:-1]))
    bad = b'{"body_bytes": -1}'
    with pytest.raises(ValueError):
        read(io.BytesIO(struct.pack("!I", len(bad)) + bad))


def test_wire_buffer_write_preserves_bytes_and_handles_short_writes():
    class ShortWriter(io.BytesIO):
        def write(self, data):
            return super().write(data[:7])

    pixels = np.arange(1024, dtype=np.uint16).reshape(16, 64)
    stream = ShortWriter()
    write(stream, {"type": "frames"}, pixels)
    assert stream.getvalue() == encode({"type": "frames"}, pixels.tobytes())
    assert read(io.BytesIO(stream.getvalue()))[1] == pixels.tobytes()
    write(stream, {"type": "cancelled"})


def test_wire_buffer_write_checks_bounds_before_writing_and_fails_on_closed_pipe():
    stream = io.BytesIO()
    with pytest.raises(ValueError, match="payload exceeds"):
        write(stream, {}, np.empty(MAX_BODY + 1, dtype=np.uint8))
    with pytest.raises(ValueError, match="header exceeds"):
        write(stream, {"too_long": "x" * MAX_HEADER})
    with pytest.raises(TypeError):
        write(stream, {}, np.zeros((10, 10), dtype=np.uint8)[:, ::2])
    assert stream.getvalue() == b""

    class ClosedPipe:
        def write(self, data):
            return 0

    with pytest.raises(EOFError):
        write(ClosedPipe(), {})


def test_resident_process_continuity_cancel_new_epoch_and_failure(tmp_path):
    async def run():
        gpu = GPUProcess(
            [sys.executable, "-m", "realtime.tests.fake_gpu"], tmp_path / "gpu.log"
        )
        await gpu.start()
        image = tmp_path / "reference.bin"
        image.write_bytes(b"test reference")
        await gpu.open_session("test-session", image)
        process_id = gpu.process.pid
        buffer = ResponseBuffer()
        buffer.replace(
            "lookahead",
            np.concatenate(
                (
                    np.ones(25 * FRAME_SAMPLES, dtype=np.int16),
                    np.full(LOOKAHEAD_SAMPLES, 7, dtype=np.int16),
                )
            ),
        )
        assert (
            await gpu.generate(take_batch(buffer))
        ).audio.valid_samples == 25 * FRAME_SAMPLES
        buffer.replace("normal", np.ones(57 * FRAME_SAMPLES, dtype=np.int16))
        for _ in range(2):
            batch = take_batch(buffer)
            result = await gpu.generate(batch)
            assert result.frames[0, 0, 0, 0] == batch.epoch
            assert len(result.frames) in (25, 32)
            assert gpu.last_memory == {
                "host_peak_rss_bytes": 123456,
                "scope": "synthetic test fixture",
            }
        buffer.replace("slow", np.ones(1000, dtype=np.int16))
        pending = asyncio.create_task(gpu.generate(take_batch(buffer)))
        await asyncio.sleep(0.1)
        buffer.cancel()
        await gpu.cancel(buffer.epoch)
        assert await pending is None
        buffer.replace("new", np.ones(1001, dtype=np.int16))
        result = await gpu.generate(take_batch(buffer))
        assert len(result.frames) == 1 and result.audio.valid_samples == 1001
        assert gpu.process.pid == process_id
        buffer.replace("strided", np.ones(1001, dtype=np.int16))
        result = await gpu.generate(take_batch(buffer))
        assert result.frames.shape == (1, 480, 640, 3)
        assert np.all(result.frames == buffer.epoch % 256)
        buffer.replace("crash", np.ones(1000, dtype=np.int16))
        with pytest.raises(RuntimeError, match="GPU generation failed"):
            await gpu.generate(take_batch(buffer))
        assert gpu.closed and gpu.process.returncode is not None
        await gpu.close()

    asyncio.run(run())


def test_unresponsive_gpu_deadline_retires_process(tmp_path):
    async def run():
        gpu = GPUProcess(
            [sys.executable, "-m", "realtime.tests.fake_gpu"],
            tmp_path / "gpu.log",
            chunk_timeout=0.1,
        )
        await gpu.start()
        image = tmp_path / "reference.bin"
        image.write_bytes(b"test reference")
        await gpu.open_session("test-session", image)
        buffer = ResponseBuffer()
        buffer.replace("stuck", np.ones(1000, dtype=np.int16))
        with pytest.raises(asyncio.TimeoutError):
            await gpu.generate(take_batch(buffer))
        assert gpu.closed and gpu.process.returncode is not None

    asyncio.run(run())


def test_cancelled_close_waiter_does_not_abandon_process(tmp_path):
    async def run():
        gpu = GPUProcess(
            [sys.executable, "-m", "realtime.tests.fake_gpu"], tmp_path / "gpu.log"
        )
        await gpu.start()
        image = tmp_path / "reference.bin"
        image.write_bytes(b"test reference")
        await gpu.open_session("test-session", image)
        buffer = ResponseBuffer()
        buffer.replace("stuck", np.ones(1000, dtype=np.int16))
        generation = asyncio.create_task(gpu.generate(take_batch(buffer)))
        await asyncio.sleep(0.1)
        closer = asyncio.create_task(gpu.close())
        await asyncio.sleep(0.05)
        closer.cancel()
        await asyncio.gather(closer, return_exceptions=True)
        await gpu.close()
        await asyncio.gather(generation, return_exceptions=True)
        assert gpu.process.returncode is not None and gpu.log.closed

    asyncio.run(run())


def test_model_pool_reuses_weights_but_resets_character_session(tmp_path):
    async def run():
        image1, image2 = tmp_path / "first.bin", tmp_path / "second.bin"
        image1.write_bytes(b"first")
        image2.write_bytes(b"second")

        def factory(command, log, **kwargs):
            return GPUProcess(
                [sys.executable, "-m", "realtime.tests.fake_gpu"],
                tmp_path / "pool.log",
                **kwargs,
            )

        pool = GPUWorkerPool(
            Settings(avatar_catalog="research", _env_file=None), factory=factory
        )
        first = await pool.acquire("first", image1)
        pid = first.process.pid
        with pytest.raises(RuntimeError, match="leased"):
            await pool.acquire("second", image2)
        buffer = ResponseBuffer()
        buffer.replace("one", np.ones(1001, dtype=np.int16))
        assert (await first.generate(take_batch(buffer))).audio.sample_start == 0
        await pool.release("wrong-owner")
        assert pool.owner == "first"
        await pool.release("first")
        assert first.session_id is None
        second = await pool.acquire("second", image2)
        assert (
            second is first
            and second.process.pid == pid
            and second.session_id == "second"
        )
        # A non-reset backend would reject this new session's sample zero.
        buffer.replace("two", np.ones(1001, dtype=np.int16))
        assert (await second.generate(take_batch(buffer))).audio.sample_start == 0
        await pool.release("second")
        await pool.close()
        assert first.process.returncode is not None

    asyncio.run(run())
