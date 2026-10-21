"""Dependency-free schemas, geometry, and atomic storage helpers."""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "data" / "facetime"
SPLITS = {"train", "val", "test"}
SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,100}\Z")
VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}\Z")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                     dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
        handle.write(content)
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("expected a JSON object")
            rows.append(row)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path, rows):
    atomic_text(path, "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))


def within(root, relative):
    """Never allow manifests to address artifacts outside the chosen dataset."""
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Artifact path escapes dataset root: {relative}")
    return path


def youtube_url(url):
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port:
        raise ValueError("Use a public HTTPS YouTube video URL (no credentials or port)")
    host = (parsed.hostname or "").lower()
    if host == "youtu.be":
        video_id = parsed.path.strip("/")
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif parsed.path.startswith(("/shorts/", "/live/", "/embed/")):
            video_id = parsed.path.split("/")[2]
        else:
            video_id = ""
    else:
        video_id = ""
    if not VIDEO_ID.fullmatch(video_id):
        raise ValueError(f"Not a supported YouTube video URL: {url}")
    return f"https://www.youtube.com/watch?v={video_id}"


@dataclass(frozen=True)
class Source:
    id: str
    url: str | None = None
    path: str | None = None
    start: float = 0.0
    end: float | None = None
    source: str = "custom"
    license: str = "unknown"
    license_evidence: str = ""
    split: str | None = None
    speaker_id: str | None = None
    conversation_id: str | None = None

    @classmethod
    def parse(cls, row, base):
        unknown = set(row) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown source fields: {sorted(unknown)}")
        value = dict(row)
        if not isinstance(value.get("id"), str) or not SAFE_ID.fullmatch(value["id"]):
            raise ValueError("Source id must contain 1-100 letters, digits, '_' or '-'")
        for key in ("url", "path"):
            if value.get(key) is not None and not isinstance(value[key], str):
                raise ValueError(f"{key} must be a string")
        if bool(value.get("url")) == bool(value.get("path")):
            raise ValueError(f"{value['id']}: provide exactly one of url or path")
        if value.get("url"):
            value["url"] = youtube_url(value["url"])
        else:
            value["path"] = str((Path(base) / value["path"]).resolve())
        for key in ("start", "end"):
            if key == "start" and key in value and value[key] is None:
                raise ValueError("start must be a number, not null")
            if value.get(key) is not None:
                number = value[key]
                if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
                    raise ValueError(f"{key} must be a finite number of seconds")
        if value.get("start", 0) < 0 or (value.get("end") is not None and value["end"] <= value.get("start", 0)):
            raise ValueError("Require 0 <= start < end")
        if value.get("split") is not None and (not isinstance(value["split"], str) or value["split"] not in SPLITS):
            raise ValueError("split must be train, val, test or null")
        for key in ("source", "license", "license_evidence", "speaker_id", "conversation_id"):
            if key in value and value[key] is not None and not isinstance(value[key], str):
                raise ValueError(f"{key} must be a string")
        for key in ("source", "license", "license_evidence"):
            if key in value and value[key] is None:
                raise ValueError(f"{key} must be a string, not null")
        return cls(**value)

    def to_dict(self):
        return asdict(self)


def load_sources(path):
    unique = {}
    for row in read_jsonl(path):
        source = Source.parse(row, Path(path).resolve().parent)
        if source.id in unique and unique[source.id] != source:
            raise ValueError(f"Conflicting duplicate source id: {source.id}")
        unique[source.id] = source
    # Aliases for the same source may leak across splits or be downloaded twice.
    locations = {}
    for source in unique.values():
        location = source.url or source.path
        if location in locations:
            raise ValueError(f"Duplicate source location: {locations[location]} and {source.id}")
        locations[location] = source.id
    return list(unique.values())


@dataclass(frozen=True)
class Config:
    width: int = 768
    height: int = 576
    min_seconds: float = 5.0
    max_seconds: float = 15.0
    sample_fps: float = 2.0
    valid_fraction: float = 0.9
    face_confidence: float = 0.5
    shoulder_confidence: float = 0.5
    scene_threshold: float = 27.0
    head_margin: float = 0.15
    side_margin: float = 0.15
    torso_depth: float = 1.0  # Shoulder-widths below the lower shoulder.

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid numeric setting: {field.name}")
        if any(not isinstance(v, int) or v <= 0 or v % 2 for v in (self.width, self.height)):
            raise ValueError("Output dimensions must be positive even integers")
        if not 0 < self.min_seconds <= self.max_seconds or not 0 < self.sample_fps <= 30:
            raise ValueError("Invalid clip lengths or sample FPS")
        if any(not 0 < v <= 1 for v in (self.valid_fraction, self.face_confidence, self.shoulder_confidence)):
            raise ValueError("Confidence/fraction settings must be in (0, 1]")
        if min(self.head_margin, self.side_margin, self.torso_depth) < 0 or self.scene_threshold <= 0:
            raise ValueError("Margins must be non-negative; scene threshold must be positive")


def load_config(path=None):
    return Config(**read_json(path)) if path else Config()


def intervals(scenes, start, end, config):
    """Split each scene independently; never join across a scene cut."""
    for scene_start, scene_end in scenes:
        cursor, stop = max(start, scene_start), min(end, scene_end)
        while stop - cursor >= config.min_seconds - 1e-7:
            next_time = min(stop, cursor + config.max_seconds)
            yield round(cursor, 6), round(next_time, 6)
            cursor = next_time


def fit_crop(box, width, height, config):
    """Contain required floating-point bounds in an even, exact-ratio rectangle."""
    left, top, right, bottom = box
    if not all(math.isfinite(x) for x in box) or not 0 <= left < right <= width or not 0 <= top < bottom <= height:
        raise ValueError("required_region_outside_source")
    divisor = math.gcd(config.width, config.height)
    unit_w, unit_h = config.width // divisor, config.height // divisor
    scale = math.ceil(max((right - left) / unit_w, (bottom - top) / unit_h,
                          config.width / unit_w, config.height / unit_h))
    # Even dimensions and origins avoid encoder chroma rounding.
    if scale % 2:
        scale += 1
    crop_w, crop_h = unit_w * scale, unit_h * scale
    while crop_w <= width and crop_h <= height:
        min_x = max(0, math.ceil((right - crop_w) / 2) * 2)
        max_x = min(math.floor(left / 2) * 2, (width - crop_w) // 2 * 2)
        min_y = max(0, math.ceil((bottom - crop_h) / 2) * 2)
        max_y = min(math.floor(top / 2) * 2, (height - crop_h) // 2 * 2)
        if min_x <= max_x and min_y <= max_y:
            x = max(min_x, min(max_x, round(((left + right - crop_w) / 2) / 2) * 2))
            y = max(min_y, min(max_y, round(((top + bottom - crop_h) / 2) / 2) * 2))
            return [x, y, crop_w, crop_h]
        crop_w += 2 * unit_w
        crop_h += 2 * unit_h
    raise ValueError("insufficient_source_area_or_resolution")


def assess_samples(samples, width, height, config):
    """Conservative heuristic; approval is still required for real torso/gaze/speech."""
    valid, reasons, regions = 0, {}, []
    for sample in samples:
        reason = sample.get("error")
        faces = sample.get("faces", [])
        poses = sample.get("poses", [])
        if not reason and len(faces) != 1:
            reason = "face_count"
        if not reason and len(poses) != 1:
            reason = "pose_count"
        if not reason:
            face, pose = faces[0], poses[0]
            shoulders = [pose[11], pose[12]]
            if any(min(p.get("visibility", 0), p.get("presence", 0)) < config.shoulder_confidence
                   or not 0 <= p["x"] <= width or not 0 <= p["y"] <= height for p in shoulders):
                reason = "missing_shoulders"
            else:
                a, b = shoulders
                shoulder_width = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
                l, t, r, bot = face["box"]
                # Reject face/pose mismatches and implausibly narrow/profile poses.
                if (shoulder_width < (r - l) * 0.8 or min(a["y"], b["y"]) < (t + bot) / 2
                        or not min(a["x"], b["x"]) <= (l + r) / 2 <= max(a["x"], b["x"])):
                    reason = "face_pose_mismatch"
                else:
                    margin = shoulder_width * config.side_margin
                    region = [min(l, a["x"], b["x"]) - margin,
                              t - (bot - t) * config.head_margin,
                              max(r, a["x"], b["x"]) + margin,
                              max(bot, a["y"], b["y"]) + shoulder_width * config.torso_depth]
                    if not 0 <= region[0] < region[2] <= width or not 0 <= region[1] < region[3] <= height:
                        reason = "torso_region_outside_source"
                    else:
                        regions.append(region)
                        valid += 1
        sample["rejection_reason"] = reason
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
    fraction = valid / len(samples) if samples else 0
    result = {"valid_fraction": fraction, "sample_rejections": reasons, "crop": None}
    if fraction + 1e-9 < config.valid_fraction or not regions:
        result["rejection_reason"] = "insufficient_valid_samples"
        return result
    # Union, not a moving crop: all valid sampled poses must fit the fixed box.
    box = [min(r[0] for r in regions), min(r[1] for r in regions),
           max(r[2] for r in regions), max(r[3] for r in regions)]
    result["required_region"] = box
    try:
        result["crop"] = fit_crop(box, width, height, config)
        result["rejection_reason"] = None
    except ValueError as exc:
        result["rejection_reason"] = str(exc)
    return result
