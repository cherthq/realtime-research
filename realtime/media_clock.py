"""Pace paired media from audio playout, without a second independent FPS clock."""

import asyncio
from collections import deque
import contextlib
import time

from livekit import rtc


class AudioClock:
    def __init__(self, audio, video, epoch=lambda: None):
        self.audio, self.video = audio, video
        self.epoch = epoch
        self.audio_lock = asyncio.Lock()
        self.queue = asyncio.Queue(maxsize=5)
        self.captured = deque(maxlen=50)
        self.max_lateness_ms = 0.0
        self.closed = False
        self.task = asyncio.create_task(self._video())

    async def push_pair(self, frame, pcm):
        # capture_frame is backpressured by the bounded native audio queue. Its
        # queued_duration locates the start of THIS packet on the playout clock.
        # No per-chunk sleeps, burst catch-up, or separate video FPS loop.
        duration = len(pcm) / self.audio.sample_rate
        async with self.audio_lock:
            position = getattr(frame, "playback_position", None)
            stale = position is not None and self.epoch() not in (None, position[0])
            data = bytes(len(pcm) * 2) if stale else pcm.tobytes()
            await self.audio.capture_frame(
                rtc.AudioFrame(data, self.audio.sample_rate, 1, len(pcm))
            )
            due = time.monotonic() + self.audio.queued_duration - duration
        await self.queue.put((frame, due))
        if self.task.done():
            await self.task

    async def _video(self):
        while True:
            frame, due = await self.queue.get()
            try:
                due += getattr(self.video, "presentation_delay_seconds", 0.0)
                await asyncio.sleep(max(0, due - time.monotonic()))
                self.max_lateness_ms = max(
                    self.max_lateness_ms, (time.monotonic() - due) * 1000
                )
                self.video.capture_frame(frame)
                self.captured.append(time.monotonic())
            finally:
                self.queue.task_done()

    async def mute_audio(self):
        # Preserve queued video and its clock across interruption. Replace the
        # unplayed audio with silence; do not jump over already-generated poses.
        async with self.audio_lock:
            remaining = round(self.audio.queued_duration * self.audio.sample_rate)
            self.audio.clear_queue()
            if remaining:
                await self.audio.capture_frame(
                    rtc.AudioFrame(
                        bytes(remaining * 2), self.audio.sample_rate, 1, remaining
                    )
                )

    async def clear_queue(self):
        async with self.audio_lock:
            self.audio.clear_queue()
        while not self.queue.empty():
            self.queue.get_nowait()
            self.queue.task_done()

    async def aclose(self):
        self.closed = True
        self.task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self.task

    @property
    def actual_fps(self):
        if len(self.captured) < 2:
            return 0.0
        return (len(self.captured) - 1) / (self.captured[-1] - self.captured[0])
