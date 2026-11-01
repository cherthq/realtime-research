"""The renderer boundary is independent of HTTP, LiveKit, and voice providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from .geometry import HEIGHT, WIDTH

SAMPLE_RATE = 48_000
FPS = 25  # Native model/audio alignment; do not change to throttle transport.
PUBLISH_FPS = 24  # WebRTC encoder cap; audio keeps its original duration.
FRAME_SAMPLES = SAMPLE_RATE // FPS


@dataclass(frozen=True)
class AudioChunk:
    response_id: str
    epoch: int
    sample_start: int
    pcm: np.ndarray
    valid_samples: int
    lookahead: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int16))

    def __post_init__(self):
        if self.pcm.dtype != np.int16 or self.pcm.ndim != 1:
            raise ValueError("Expected mono PCM16")
        if not 0 <= self.valid_samples <= len(self.pcm):
            raise ValueError("Invalid audio coverage")
        if self.sample_start < 0:
            raise ValueError("Negative audio position")
        if (
            self.lookahead.dtype != np.int16
            or self.lookahead.ndim != 1
            or len(self.lookahead) > SAMPLE_RATE
        ):
            raise ValueError("Invalid bounded PCM lookahead")


@dataclass(frozen=True)
class MediaChunk:
    audio: AudioChunk
    rgb: np.ndarray

    def __post_init__(self):
        if self.rgb.shape != (HEIGHT, WIDTH, 3) or self.rgb.dtype != np.uint8:
            raise ValueError(f"Renderer must emit native {WIDTH}x{HEIGHT} RGB uint8")


class Renderer(Protocol):
    """A renderer sees arrived audio only; it must not access an entire WAV."""

    def render(self, chunk: AudioChunk) -> MediaChunk: ...
    def reset(self) -> None: ...


# Live voice may arrive faster than playback. This CPU PCM backlog uses at most
# 28.8 MB, independent of GPU/video queues, and is discarded on interruption.
# It is a pending-audio guard, not a limit on total reply length or token count.
LIVE_VOICE_PENDING_SECONDS = 300


class ResponseBuffer:
    """Bounded PCM timeline with epoch invalidation and exact partial coverage."""

    def __init__(
        self,
        *,
        max_pending_seconds: int = 20,
        on_change: Callable[[], None] | None = None,
    ):
        if not isinstance(max_pending_seconds, int) or max_pending_seconds <= 0:
            raise ValueError(
                "Pending audio capacity must be a positive number of seconds"
            )
        self.max_pending_seconds = max_pending_seconds
        self.epoch = 0
        self.response_id = "idle"
        self.position = 0
        self.samples = np.zeros(0, dtype=np.int16)
        self.receiving = False
        self.on_change = on_change

    def _changed(self):
        if self.on_change is not None:
            self.on_change()

    def replace(self, response_id: str, samples: np.ndarray):
        if samples.ndim != 1 or samples.dtype != np.int16:
            raise ValueError("Expected mono PCM16")
        if len(samples) > SAMPLE_RATE * 20:
            raise ValueError("Response exceeds the twenty-second demo audio bound")
        self.cancel()
        self.response_id = response_id
        self.samples = samples.copy()
        self._changed()

    def begin(self, response_id: str) -> int:
        self.cancel()
        self.response_id = response_id
        self.receiving = True
        self._changed()
        return self.epoch

    def append(self, response_id: str, epoch: int, samples: np.ndarray) -> bool:
        if epoch != self.epoch or response_id != self.response_id or not self.receiving:
            return False
        if samples.ndim != 1 or samples.dtype != np.int16:
            raise ValueError("Expected mono PCM16")
        if len(self.samples) + len(samples) > SAMPLE_RATE * self.max_pending_seconds:
            raise ValueError(
                f"Pending response exceeded the {self.max_pending_seconds}-second audio queue bound"
            )
        self.samples = np.concatenate((self.samples, samples))
        self._changed()
        return True

    def finish(self, response_id: str, epoch: int):
        if epoch == self.epoch and response_id == self.response_id:
            self.receiving = False
            self._changed()

    def cancel(self):
        self.epoch += 1
        self.response_id = "idle"
        self.samples = np.zeros(0, dtype=np.int16)
        self.position = 0
        self.receiving = False
        self._changed()

    @property
    def active(self):
        return self.receiving or len(self.samples) > 0

    def next(self) -> AudioChunk:
        start = self.position
        # Never insert zero samples inside an unfinished response's own timeline.
        # The publisher may send an explicitly idle transport frame while waiting.
        if self.receiving and len(self.samples) < FRAME_SAMPLES:
            return AudioChunk(
                "idle", self.epoch, start, np.zeros(FRAME_SAMPLES, dtype=np.int16), 0
            )
        part = self.samples[:FRAME_SAMPLES]
        self.samples = self.samples[len(part) :]
        self.position += len(part)
        pcm = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        pcm[: len(part)] = part
        return AudioChunk(self.response_id, self.epoch, start, pcm, len(part))
