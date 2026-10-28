"""One model process across calls; per-call character state is explicitly reset."""

from __future__ import annotations

import asyncio
from pathlib import Path
import time

from .config import ROOT
from .gpu_client import GPUProcess


class GPUWorkerPool:
    def __init__(self, settings, factory=GPUProcess):
        self.settings, self.factory = settings, factory
        self.worker = None
        self.owner = None
        self.lock = asyncio.Lock()

    def snapshot(self):
        process = self.worker.process if self.worker else None
        alive = (
            process is not None
            and process.returncode is None
            and not self.worker.closed
        )
        loaded = bool(alive and self.worker.info is not None)
        return {
            "configured": self.settings.avatarforever_enabled,
            "loaded": loaded,
            "state": "disabled"
            if not self.settings.avatarforever_enabled
            else "leased"
            if loaded and self.owner
            else "loaded"
            if loaded
            else "loading"
            if self.owner
            else "unloaded",
            "qualification": "streaming_experimental",
            "cold_load_seconds": self.worker.load_seconds if loaded else None,
        }

    async def acquire(self, session_id: str, image: Path):
        async with self.lock:
            if self.owner is not None:
                raise RuntimeError("GPU process already leased")
            self.owner = session_id
            try:
                if self.worker is None or self.worker.closed:
                    self.worker = self.factory(
                        [
                            str(self.settings.avatarforever_python),
                            "-m",
                            "realtime.gpu.server",
                            "--weights",
                            str(self.settings.avatarforever_weights),
                            "--image",
                            str(image),
                            *(
                                ["--compile-blocks"]
                                if self.settings.avatarforever_compile_blocks
                                else []
                            ),
                        ],
                        ROOT / "runs/realtime/gpu" / f"process-{time.time_ns()}.log",
                        load_timeout=self.settings.gpu_load_timeout_seconds,
                        chunk_timeout=self.settings.gpu_chunk_timeout_seconds,
                    )
                    await self.worker.start()
                await self.worker.open_session(session_id, image)
                return self.worker
            except BaseException:
                self.owner = None
                if self.worker:
                    await self.worker.close()
                raise

    async def release(self, session_id: str):
        async with self.lock:
            if self.owner != session_id:
                return
            try:
                if self.worker and not self.worker.closed:
                    await self.worker.close_session(session_id)
            finally:
                self.owner = None
                if self.worker and self.worker.closed:
                    self.worker = None

    async def close(self):
        async with self.lock:
            if self.owner is not None:
                raise RuntimeError("End the active GPU session before unloading")
            if self.worker:
                await self.worker.close()
            self.worker = None
