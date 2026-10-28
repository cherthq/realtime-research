"""Async process boundary; CPU service never imports torch or GPU dependencies."""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
from pathlib import Path
import time

import numpy as np

from .config import ROOT
from .contracts import AudioChunk, FRAME_SAMPLES, HEIGHT, WIDTH
from .gpu.wire import encode, read_async
from .streaming import GeneratedBatch


class GPUProcess:
    def __init__(
        self, command: list[str], log: Path, *, load_timeout=300, chunk_timeout=60
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

    async def start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("wb")
        environment = {
            k: v
            for k, v in os.environ.items()
            if k not in {"OPENAI_API_KEY", "DEMO_ACCESS_TOKEN", "LIVEKIT_API_SECRET"}
        }
        environment["PYTHONUNBUFFERED"] = "1"
        mark = time.monotonic()
        try:
            self.process = await asyncio.create_subprocess_exec(
                *self.command,
                cwd=ROOT,
                env=environment,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=self.log,
            )
            header, body = await asyncio.wait_for(
                read_async(self.process.stdout), self.load_timeout
            )
            if header.get("type") != "ready" or body:
                raise RuntimeError("GPU process did not become ready")
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

    async def generate(self, audio: AudioChunk) -> GeneratedBatch | None:
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
                    },
                    audio.pcm.astype("<i2", copy=False).tobytes(),
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
                if (
                    header.get("frame_count") != count
                    or len(body) != count * HEIGHT * WIDTH * 3
                ):
                    raise RuntimeError("GPU returned invalid frame coverage")
                frames = np.frombuffer(body, dtype=np.uint8).reshape(
                    count, HEIGHT, WIDTH, 3
                )
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
            if self.log:
                self.log.close()
