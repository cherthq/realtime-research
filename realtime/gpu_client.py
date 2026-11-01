"""Async process boundary; CPU service never imports torch or GPU dependencies."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import math
import mmap
import os
from pathlib import Path
import time
import tempfile

import numpy as np

from .config import ROOT
from .contracts import AudioChunk, FRAME_SAMPLES, HEIGHT, WIDTH
from .gpu.wire import MAX_BODY, encode, read_async
from .streaming import GeneratedBatch


class GPUProcess:
    def __init__(
        self,
        command: list[str],
        log: Path,
        *,
        load_timeout=300,
        chunk_timeout=60,
        shared_frames=True,
    ):
        self.command, self.log_path = command, log
        self.load_timeout, self.chunk_timeout = load_timeout, chunk_timeout
        self.process = None
        self.log = None
        self.info = None
        self.last_memory = None
        self.lock = asyncio.Lock()
        self.write_lock = asyncio.Lock()
        self.closed = False
        self.load_seconds = None
        self._closing = None
        self.session_id = None
        self.idle_clip = None
        self.idle_image_hash = None
        self.shared_frames = shared_frames and os.name == "posix"
        self.frame_file = self.frame_buffer = None

    async def prepare_idle(self, image: Path):
        """Cache one character only, with speech state reset after preparation."""
        from .idle import generate_idle

        image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
        if self.idle_clip is not None and self.idle_image_hash == image_hash:
            return
        self.idle_clip = None
        self.idle_image_hash = None
        if self.session_id is None:
            raise RuntimeError("Open a character session before idle preparation")
        try:
            clip = await generate_idle(self)
            # Clear the idle generation history and IPC epoch before live speech.
            await self.open_session(self.session_id, image)
        except BaseException:
            # An interrupted preparation must not leave half an idle history
            # available for a subsequent caller's speech.
            await self.close()
            raise
        self.idle_clip, self.idle_image_hash = clip, image_hash

    async def start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("wb")
        environment = {
            k: v
            for k, v in os.environ.items()
            if k not in {"OPENAI_API_KEY", "DEMO_ACCESS_TOKEN", "LIVEKIT_API_SECRET"}
        }
        environment["PYTHONUNBUFFERED"] = "1"
        environment["AVATAR_RESOLUTION"] = f"{WIDTH}x{HEIGHT}"
        environment.pop("CHERT_GPU_FRAME_BUFFER_FD", None)
        inherited = {}
        mark = time.monotonic()
        try:
            if self.shared_frames:
                # Anonymous, inherited descriptor: no public path or shared-memory
                # name. One in-flight job owns it; copy before requesting another.
                self.frame_file = tempfile.TemporaryFile()
                self.frame_file.truncate(MAX_BODY)
                self.frame_buffer = mmap.mmap(self.frame_file.fileno(), MAX_BODY)
                descriptor = self.frame_file.fileno()
                environment["CHERT_GPU_FRAME_BUFFER_FD"] = str(descriptor)
                inherited["pass_fds"] = (descriptor,)
            self.process = await asyncio.create_subprocess_exec(
                *self.command,
                cwd=ROOT,
                env=environment,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=self.log,
                **inherited,
            )
            header, body = await asyncio.wait_for(
                read_async(self.process.stdout), self.load_timeout
            )
            if header.get("type") != "ready" or body:
                raise RuntimeError("GPU process did not become ready")
            if header.get("geometry") != {"width": WIDTH, "height": HEIGHT, "fps": 25}:
                raise RuntimeError(
                    "GPU process native geometry differs from the service"
                )
            expected_transport = "mmap" if self.frame_buffer is not None else "pipe"
            if header.get("frame_transport", "pipe") != expected_transport:
                raise RuntimeError("GPU frame transport was not negotiated")
            self.info = header
            self.last_memory = header.get("memory")
            self.load_seconds = time.monotonic() - mark
        except BaseException:
            await self.close()
            raise

    async def _send(self, header, body=b""):
        if self.closed or not self.process or self.process.returncode is not None:
            raise RuntimeError("GPU process is unavailable")
        async with self.write_lock:
            self.process.stdin.write(encode(header, body))
            await asyncio.wait_for(self.process.stdin.drain(), 2)

    async def generate(
        self, audio: AudioChunk, *, continuous=False
    ) -> GeneratedBatch | None:
        async with self.lock:
            try:
                await self._send(
                    {
                        "op": "generate",
                        "session_id": self.session_id,
                        "response_id": audio.response_id,
                        "epoch": audio.epoch,
                        "sample_start": audio.sample_start,
                        "valid_samples": audio.valid_samples,
                        "continuous": continuous,
                        "lookahead_samples": len(audio.lookahead),
                    },
                    np.concatenate((audio.pcm, audio.lookahead))
                    .astype("<i2", copy=False)
                    .tobytes(),
                )
                header, body = await asyncio.wait_for(
                    read_async(self.process.stdout), self.chunk_timeout
                )
                if (
                    header.get("session_id"),
                    header.get("epoch"),
                    header.get("response_id"),
                    header.get("sample_start"),
                ) != (
                    self.session_id,
                    audio.epoch,
                    audio.response_id,
                    audio.sample_start,
                ):
                    raise RuntimeError("GPU returned a different response identity")
                if header.get("type") == "cancelled":
                    return None
                if header.get("type") != "frames":
                    raise RuntimeError(
                        f"GPU generation failed ({header.get('error_type', 'invalid response')})"
                    )
                count = math.ceil(audio.valid_samples / FRAME_SAMPLES)
                expected_bytes = count * HEIGHT * WIDTH * 3
                if header.get("frame_count") != count or expected_bytes > MAX_BODY:
                    raise RuntimeError("GPU returned invalid frame coverage")
                if self.frame_buffer is not None:
                    if body or header.get("frame_bytes") != expected_bytes:
                        raise RuntimeError("GPU returned invalid shared frame coverage")
                    # Queued playback must not alias the next generation's buffer.
                    frames = np.frombuffer(
                        self.frame_buffer, dtype=np.uint8, count=expected_bytes
                    ).copy()
                else:
                    if len(body) != expected_bytes:
                        raise RuntimeError("GPU returned invalid frame coverage")
                    frames = np.frombuffer(body, dtype=np.uint8)
                frames = frames.reshape(count, HEIGHT, WIDTH, 3)
                self.last_memory = header.get("memory")
                return GeneratedBatch(audio, frames, header["compute_seconds"])
            except BaseException:
                # A timed-out or interrupted read leaves IPC framing uncertain.
                # Retire this worker rather than reusing a potentially stale reply.
                await self.close()
                raise

    async def open_session(self, session_id: str, image: Path):
        async with self.lock:
            try:
                await self._send(
                    {
                        "op": "open_session",
                        "session_id": session_id,
                        "image": str(image.resolve()),
                    }
                )
                header, body = await asyncio.wait_for(
                    read_async(self.process.stdout), 30
                )
                if (
                    header.get("type") != "session_ready"
                    or header.get("session_id") != session_id
                    or body
                ):
                    raise RuntimeError("GPU character preparation failed")
                self.session_id = session_id
            except BaseException:
                await self.close()
                raise

    async def close_session(self, session_id: str):
        async with self.lock:
            try:
                if self.session_id != session_id:
                    raise ValueError("GPU session lease changed")
                await self._send({"op": "close_session", "session_id": session_id})
                header, body = await asyncio.wait_for(
                    read_async(self.process.stdout), 3
                )
                if (
                    header.get("type") != "session_closed"
                    or header.get("session_id") != session_id
                    or body
                ):
                    raise RuntimeError("GPU session cleanup failed")
                self.session_id = None
            except BaseException:
                await self.close()
                raise

    async def cancel(self, epoch: int):
        if not self.closed:
            await self._send({"op": "cancel", "epoch": epoch})

    async def close(self):
        if self._closing is None:
            self._closing = asyncio.create_task(self._close())
        # A cancelled request must not abandon an owned GPU process halfway
        # through cleanup; subsequent close calls await the same operation.
        await asyncio.shield(self._closing)

    async def _close(self):
        self.closed = True
        self.idle_clip = None
        self.idle_image_hash = None
        try:
            if self.process and self.process.returncode is None:
                with contextlib.suppress(Exception):
                    self.process.stdin.write(encode({"op": "close"}))
                    await asyncio.wait_for(self.process.stdin.drain(), 1)
                try:
                    await asyncio.wait_for(self.process.wait(), 6)
                except asyncio.TimeoutError:
                    self.process.kill()
                    await asyncio.wait_for(self.process.wait(), 2)
        finally:
            if self.frame_buffer is not None:
                self.frame_buffer.close()
                self.frame_buffer = None
            if self.frame_file is not None:
                self.frame_file.close()
                self.frame_file = None
            if self.log:
                self.log.close()
