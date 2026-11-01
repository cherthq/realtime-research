"""Latest camera frame only; CPU encoding never runs on the media event loop."""

import asyncio
import base64
import io
import time

from PIL import Image
from livekit import rtc


class CameraInput:
    def __init__(self):
        self.frame = None
        self.received_at = 0.0
        self.generation = 0
        self.track_sid = None
        self.task = None
        self.error = None

    def attach(self, track, sid):
        self.clear()
        self.track_sid = sid
        generation = self.generation
        self.task = asyncio.create_task(self._read(track, generation))
        self.task.add_done_callback(
            lambda task: task.exception() if not task.cancelled() else None
        )

    def clear(self):
        self.generation += 1
        if self.task:
            self.task.cancel()
        self.task = None
        self.frame, self.track_sid, self.received_at = None, None, 0.0

    async def _read(self, track, generation):
        stream = rtc.VideoStream(track, capacity=1)
        try:
            async for event in stream:
                if generation != self.generation:
                    return
                now = time.monotonic()
                if now - self.received_at >= 1:
                    self.frame, self.received_at = event.frame, now
        except asyncio.CancelledError:
            raise
        except Exception:
            if generation == self.generation:
                self.error = "Camera stream unavailable"
                self.frame = None
        finally:
            await stream.aclose()

    @staticmethod
    def encode(frame):
        rgb = (
            frame
            if frame.type == rtc.VideoBufferType.RGB24
            else frame.convert(rtc.VideoBufferType.RGB24)
        )
        image = Image.frombytes("RGB", (rgb.width, rgb.height), bytes(rgb.data))
        image.thumbnail((640, 480))
        output = io.BytesIO()
        image.save(output, "JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode()

    async def snapshot(self):
        generation, frame = self.generation, self.frame
        if frame is None or time.monotonic() - self.received_at > 2:
            return None
        try:
            image = await asyncio.wait_for(asyncio.to_thread(self.encode, frame), 0.2)
        except Exception:
            # A missing/unsupported camera frame must not break a voice turn.
            return None
        return image if generation == self.generation else None

    async def close(self):
        task = self.task
        self.clear()
        if task:
            await asyncio.gather(task, return_exceptions=True)
