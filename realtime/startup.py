"""Measured, structured startup for the opt-in GPT-Live worker."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time

logger = logging.getLogger("uvicorn.error")


class StartupTrace:
    """Monotonic offsets only: never prompt, audio, tokens, or network URLs."""

    def __init__(self, session_id, clock=time.monotonic):
        self.session_id, self.clock = session_id, clock
        self.started = clock()
        self.events = {}

    def mark(self, event):
        if event not in self.events:
            self.events[event] = round((self.clock() - self.started) * 1000, 2)
            logger.info(
                "call_startup session=%s event=%s elapsed_ms=%s",
                self.session_id,
                event,
                self.events[event],
            )

    async def stage(self, name, operation):
        self.mark(name + "_started")
        try:
            result = await operation()
        except BaseException:
            self.mark(name + "_failed")
            raise
        self.mark(name + "_ready")
        return result

    def snapshot(self):
        return {
            "events_ms": dict(self.events),
            "scope": "server startup; first media is server capture, not handset playout",
        }


async def start_parallel(worker):
    """Overlap connections; admit callers only after the recording/voice barrier.

    A TaskGroup drains every sibling on failure/cancellation BEFORE the caller
    releases the GPU or closes room/voice resources. No paid sessions while idle.
    """
    trace = worker.startup
    transport_ready = asyncio.Event()

    async def media():
        await worker._connect_media()
        transport_ready.set()

    async def recording():
        recorder = getattr(worker, "call_recorder", None)
        if recorder is None:
            return
        trace.mark("recorder_started")
        try:
            async with asyncio.timeout(10):
                await recorder.start()
            await transport_ready.wait()
            # Track subscription acknowledgment, not just signaling connected.
            async with asyncio.timeout(3):
                await recorder.avatar_audio_ready.wait()
            trace.mark("recorder_ready")
        except Exception:
            recorder.error = "Recording connection could not start"
            trace.mark("recorder_failed")
            # Persist immediately even when startup failed before checkpointing.
            with contextlib.suppress(Exception):
                await asyncio.to_thread(
                    recorder.logs.update, recorder.id, "recording", 0, recorder.error
                )

    pool = getattr(worker, "gpu_pool", None)
    cold = pool is not None and not pool.snapshot()["loaded"]
    if cold:
        # Do not open billable voice/recording connections during a cold model load.
        await trace.stage("renderer", worker._prepare_renderer)
    async with asyncio.TaskGroup() as group:
        renderer = (
            None
            if cold
            else group.create_task(trace.stage("renderer", worker._prepare_renderer))
        )
        transport = group.create_task(trace.stage("transport", media))
        group.create_task(recording())
        if worker.voice:
            group.create_task(trace.stage("voice", worker.voice.start))
        if renderer is not None:
            await renderer
        await transport
        worker._start_media()
        trace.mark("media_started")
