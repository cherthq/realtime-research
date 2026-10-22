"""Explicit quality profile; legacy Config and cached results remain unchanged."""

from dataclasses import asdict, dataclass, fields
from functools import lru_cache
import math

from .core import Config, read_json


@dataclass(frozen=True)
class QualityConfig:
    framing_profile: str = "torso"
    scene_detector: str = "content"
    detector_device: str = "cpu"
    width: int = 768
    height: int = 576
    min_seconds: float = 5.0
    long_seconds: float = 60.0
    max_seconds: float = 300.0
    transition_guard: float = 0.25
    scene_threshold: float = 27.0
    fade_threshold: float = 32.0
    face_confidence: float = 0.5
    shoulder_confidence: float = 0.5
    head_margin: float = 0.30
    side_margin: float = 0.12
    chest_depth: float = 0.30
    torso_depth: float = 0.80
    torso_min_depth: float = 0.50
    track_iou: float = 0.15
    timestamp_gap: float = 0.25
    blur_threshold: float = 35.0
    dark_threshold: float = 40.0
    bright_threshold: float = 225.0
    freeze_seconds: float = 1.0
    ocr_fps: float = 2.0
    sync_max_ms: float = 80.0
    sync_min_confidence: float = 3.0
    active_min_score: float = 0.0
    window_seconds: float = 3.0
    window_stride: float = 1.0
    crop_safety: float = 0.04
    tracking_gap: float = 0.25
    secondary_face_confidence: float = 0.8
    near_freeze_difference: float = 0.25
    active_bad_seconds: float = 0.20
    vad_threshold: float = 0.5
    vad_release_threshold: float = 0.35
    vad_min_speech: float = 0.25
    vad_min_silence: float = 0.10
    vad_padding: float = 0.03
    ocr_min_confidence: float = 0.5
    ocr_burst_difference: float = 20.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in {"framing_profile", "scene_detector", "detector_device"}:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Invalid quality setting: {field.name}")
            if value < 0:
                raise ValueError(f"Negative quality setting: {field.name}")
        if not 0 < self.min_seconds <= self.long_seconds <= self.max_seconds:
            raise ValueError("Require 0 < min_seconds <= long_seconds <= max_seconds")
        if (
            not 0 < self.window_stride <= self.window_seconds
            or not 0 < self.ocr_fps <= 30
        ):
            raise ValueError("Invalid worker sampling/window settings")
        if not 0 < self.timestamp_gap or not 0 < self.freeze_seconds:
            raise ValueError("Gap/freeze durations must be positive")
        if not 0 <= self.track_iou <= 1 or self.fade_threshold > 255:
            raise ValueError("Invalid tracking/fade settings")
        if self.framing_profile not in {"head_chest", "torso", "gesture"}:
            raise ValueError("Invalid framing_profile")
        if self.scene_detector not in {"content", "adaptive"}:
            raise ValueError("Invalid scene_detector")
        if self.detector_device not in {"cpu", "gpu"}:
            raise ValueError("detector_device must be cpu or gpu; no automatic fallback")
        if (
            self.tracking_gap > 1
            or self.crop_safety > 0.25
            or not 0 <= self.secondary_face_confidence <= 1
        ):
            raise ValueError("Invalid recovery/safety/secondary-face settings")
        if (
            not 0 <= self.vad_release_threshold <= self.vad_threshold <= 1
            or not 0 <= self.ocr_min_confidence <= 1
        ):
            raise ValueError("Invalid VAD/OCR thresholds")
        if self.torso_min_depth > self.torso_depth:
            raise ValueError("torso_min_depth cannot exceed torso_depth")
        self.geometry()

    @lru_cache(maxsize=128)
    def geometry(self, wider=None):
        depth = (
            self.torso_depth
            if (wider is True or wider is None and self.framing_profile != "head_chest")
            else self.chest_depth
        )
        if wider is None and self.framing_profile == "gesture":
            depth *= 1.15
        return Config(
            width=self.width,
            height=self.height,
            min_seconds=self.min_seconds,
            max_seconds=self.max_seconds,
            sample_fps=2,
            valid_fraction=1,
            face_confidence=self.face_confidence,
            shoulder_confidence=self.shoulder_confidence,
            scene_threshold=self.scene_threshold,
            head_margin=self.head_margin,
            side_margin=self.side_margin,
            torso_depth=depth,
        )

    def to_dict(self):
        return asdict(self)


def load_quality_config(path=None):
    return QualityConfig(**read_json(path)) if path else QualityConfig()
