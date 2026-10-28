"""CPU-side contracts for bounded native avatar chunks, independent of transport."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .contracts import AudioChunk, FPS, FRAME_SAMPLES, HEIGHT, SAMPLE_RATE, WIDTH

FIRST_FRAMES = 25  # four LTX temporal latents: (4 - 1) * 8 + 1
NEXT_FRAMES = 32  # each following four latents adds 4 * 8 frames


def capabilities(model_id: str) -> dict:
    """Describe the implemented adapter, not an upstream performance claim."""
    if model_id not in {"mock", "avatarforever"}:
        raise ValueError("Unsupported streaming adapter")
    real = model_id == "avatarforever"
    first, following = (FIRST_FRAMES, NEXT_FRAMES) if real else (1, 1)
    return {
        "geometry": {"width": WIDTH, "height": HEIGHT, "fps": FPS, "native": True},
        "audio": {
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "format": "pcm16",
            "packet_samples": 960,
            "first_chunk_samples": first * FRAME_SAMPLES,
            "following_chunk_samples": following * FRAME_SAMPLES,
        },
        "incremental_input": True,
        "additional_future_audio_samples": 0,
        "chunk_buffering_seconds": {"first": first / FPS, "following": following / FPS},
        "conditioning": {
            "image": True,
            "audio": True,
            "prompt": "fixed_video_call_prompt" if real else "unsupported",
        },
        "reset": "reference_and_empty_history_per_response" if real else "stateless",
        "cancel": "invalidate_epoch_and_discard_late_results",
        "qualification": "streaming_experimental" if real else "diagnostic_only",
        "timing_note": "Frames can depend on later samples within the arrived chunk; chunk buffering is included in latency.",
    }


@dataclass(frozen=True)
class GeneratedBatch:
    audio: AudioChunk
    frames: np.ndarray
    compute_seconds: float

    def __post_init__(self):
        expected = math.ceil(self.audio.valid_samples / FRAME_SAMPLES)
        if self.frames.dtype != np.uint8 or self.frames.shape != (
            expected,
            HEIGHT,
            WIDTH,
            3,
        ):
            raise ValueError(
                "Native output must cover valid audio at 25 FPS in direct 640x480"
            )
        if not math.isfinite(self.compute_seconds) or self.compute_seconds <= 0:
            raise ValueError("Missing generation timing")


def chunk_frames(sample_start: int) -> int:
    if sample_start < 0:
        raise ValueError("Negative sample position")
    return FIRST_FRAMES if sample_start == 0 else NEXT_FRAMES


def take_batch(buffer) -> AudioChunk | None:
    """Never consume unfinished partial audio or insert silence into a live turn.

    Only the final chunk is padded for the model's temporal constraint. Its valid
    sample count is carried separately and determines published output coverage.
    """
    count = chunk_frames(buffer.position) * FRAME_SAMPLES
    if not len(buffer.samples) or (buffer.receiving and len(buffer.samples) < count):
        return None
    part = buffer.samples[:count]
    start = buffer.position
    buffer.samples = buffer.samples[len(part) :]
    buffer.position += len(part)
    pcm = np.zeros(count, dtype=np.int16)
    pcm[: len(part)] = part
    return AudioChunk(buffer.response_id, buffer.epoch, start, pcm, len(part))
