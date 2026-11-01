"""Bounded, silent model motion reused between replies, never speech coverage."""

from dataclasses import dataclass

import numpy as np

from .contracts import AudioChunk, FRAME_SAMPLES, HEIGHT, WIDTH
from .streaming import FIRST_FRAMES, NEXT_FRAMES, chunk_frames, LOOKAHEAD_SAMPLES

IDLE_CHUNKS = 9  # 11.24s; one character cache is bounded to about 259 MB of RGB.
IDLE_FRAMES = FIRST_FRAMES + (IDLE_CHUNKS - 1) * NEXT_FRAMES
BLEND_FRAMES = 4


@dataclass(frozen=True)
class IdleClip:
    frames: np.ndarray
    compute_seconds: float
    loop_end: int | None = None
    seam_mae: float | None = None

    def __post_init__(self):
        if (
            self.frames.dtype != np.uint8
            or self.frames.ndim != 4
            or self.frames.shape[1:] != (HEIGHT, WIDTH, 3)
            or not 2 * BLEND_FRAMES < len(self.frames) <= IDLE_FRAMES
        ):
            raise ValueError(
                f"Idle clip must contain bounded native {WIDTH}x{HEIGHT} frames"
            )
        end, seam = select_loop_end(self.frames)
        object.__setattr__(self, "loop_end", end)
        object.__setattr__(self, "seam_mae", seam)
        self.frames.flags.writeable = False


def select_loop_end(frames):
    """Find a similar endpoint and motion direction using a cheap RGB proxy.

    This is not a face/pose metric. A low score cannot certify natural motion.
    Keep at least 60% of the candidate and report the residual for human review.
    """
    small = frames[:, ::16, ::16].astype(np.float32)
    start = max(2 * BLEND_FRAMES, int(len(frames) * 0.6))
    ends = np.arange(start, len(frames))
    appearance = np.abs(small[ends] - small[0]).mean(axis=(1, 2, 3))
    outgoing = small[ends] - small[ends - 1]
    entering = small[1] - small[0]
    velocity = np.abs(outgoing - entering).mean(axis=(1, 2, 3))
    choice = int(np.argmin(appearance + velocity))
    return int(ends[choice]) + 1, float(appearance[choice])


async def generate_idle(gpu) -> IdleClip:
    """Run before a call; the caller must reset the session before speech.

    This uses the same reference, model and prompt with silence as conditioning.
    It is a visual candidate, not a guarantee of closed lips or natural motion.
    Never feed microphone audio or a previous spoken reply into this cache.
    """
    frames = np.empty((IDLE_FRAMES, HEIGHT, WIDTH, 3), dtype=np.uint8)
    position, cursor, seconds = 0, 0, 0.0
    for _ in range(IDLE_CHUNKS):
        count = chunk_frames(position) * FRAME_SAMPLES
        chunk = AudioChunk(
            "idle-preparation",
            0,
            position,
            np.zeros(count, dtype=np.int16),
            count,
            np.zeros(LOOKAHEAD_SAMPLES, dtype=np.int16),
        )
        batch = await gpu.generate(chunk)
        if batch is None:
            raise RuntimeError("Idle preparation was cancelled")
        frames[cursor : cursor + len(batch.frames)] = batch.frames
        cursor += len(batch.frames)
        position += count
        seconds += batch.compute_seconds
    return IdleClip(frames, seconds)


def blend(a: np.ndarray, b: np.ndarray, step: int) -> np.ndarray:
    # Integer arithmetic keeps temporary memory bounded and avoids RGB overflow.
    return (
        (a.astype(np.uint16) * (BLEND_FRAMES - step) + b.astype(np.uint16) * step)
        // BLEND_FRAMES
    ).astype(np.uint8)


class IdlePlayer:
    def __init__(self, clip: IdleClip | None):
        self.clip = clip
        self.position = 0
        self.entry = None
        self.entry_step = 0

    def stop(self):
        self.entry = None
        self.entry_step = 0

    def next(self, last_frame: np.ndarray) -> np.ndarray:
        if self.clip is None:
            return last_frame
        if self.entry is None:
            self.entry = last_frame
            self.entry_step = 0
        # Forward playback with an overlapping seam; never reverse blinks or
        # gestures. The beginning is used in the overlap at the end of the loop.
        frames = self.clip.frames[: self.clip.loop_end]
        index = BLEND_FRAMES + self.position
        rgb = frames[index]
        seam = index - (len(frames) - BLEND_FRAMES)
        # Do not hide a bad pose match behind a long ghosted dissolve. The
        # experiment report exposes that seam; it still needs visual approval.
        if seam >= 0 and self.clip.seam_mae <= 12:
            rgb = blend(rgb, frames[seam], seam + 1)
        self.position = (self.position + 1) % (len(frames) - BLEND_FRAMES)
        if self.entry_step < BLEND_FRAMES:
            self.entry_step += 1
            rgb = blend(self.entry, rgb, self.entry_step)
        return rgb
