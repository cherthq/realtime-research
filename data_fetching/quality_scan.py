"""Sequential PTS-aware, bounded-memory full-frame visual analysis. No downloads."""

from collections import Counter, deque
import gzip
import hashlib
import json
import math
import time

from .core import fit_crop, fingerprint, write_json
from .metadata import ranges_union, subtract_ranges
from .quality_tracking import SubjectTracker, recover_rows


def frames(path, start, end):
    """Yield RGB frames with actual PTS and next-frame endpoints (not index/FPS)."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        origin = (stream.start_time or 0) * stream.time_base
        container.seek(
            max(0, int((float(origin) + start) / float(stream.time_base))),
            stream=stream,
        )
        previous, last_step = None, None
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Missing frame PTS; cannot certify coverage")
            stamp = float(frame.pts * frame.time_base - origin)
            if stamp < start - 1e-7:
                continue
            if previous is not None:
                if stamp <= previous[0]:
                    raise ValueError("Non-monotonic frame PTS")
                last_step = stamp - previous[0]
                yield (
                    previous[1],
                    {
                        "time": previous[0],
                        "end": min(stamp, end),
                        "pts": previous[2],
                        "end_pts": frame.pts,
                        "time_base": str(frame.time_base),
                        "corrupt": previous[3],
                    },
                )
            if stamp >= end - 1e-7:
                return
            previous = (
                stamp,
                frame.to_ndarray(format="rgb24"),
                frame.pts,
                frame.is_corrupt,
            )
        if previous is not None:
            if last_step is None:
                raise ValueError(
                    "Insufficient frames to establish final frame duration"
                )
            stop = min(end, previous[0] + last_step)
            yield (
                previous[1],
                {
                    "time": previous[0],
                    "end": stop,
                    "pts": previous[2],
                    "end_pts": previous[2] + round(last_step / float(stream.time_base)),
                    "time_base": str(stream.time_base),
                    "corrupt": previous[3],
                },
            )


def read_frames(path, start=0, end=math.inf):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["time"] >= end:
                break
            if row["time"] >= start - 1e-7 and row["end"] <= end + 1e-7:
                yield row


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area = lambda r: max(0, r[2] - r[0]) * max(0, r[3] - r[1])
    return intersection / max(1, area(a) + area(b) - intersection)


def union_box(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def image_metrics(rgb, face=None, *, gray=None, small=None):
    import cv2
    import numpy as np

    if gray is None:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    tiny = cv2.resize(gray, (17, 16), interpolation=cv2.INTER_AREA)
    bits = (tiny[:, 1:] > tiny[:, :-1]).reshape(-1)
    signature = int.from_bytes(np.packbits(bits).tobytes(), "big")
    region = gray
    if face:
        l, t, r, b = face
        region = gray[
            max(0, int(t)) : min(gray.shape[0], math.ceil(b)),
            max(0, int(l)) : min(gray.shape[1], math.ceil(r)),
        ]
        if not region.size:
            raise ValueError("Empty detected face region")
    # Normalize face ROI size so sharpness threshold is not dominated by resolution.
    normalized = cv2.resize(region, (128, 128), interpolation=cv2.INTER_AREA)
    if small is None:
        small = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
    vertical = np.mean(np.abs(np.diff(small.astype(float), axis=1)), axis=0)
    horizontal = np.mean(np.abs(np.diff(small.astype(float), axis=0)), axis=1)
    return {
        "sharpness": float(cv2.Laplacian(normalized, cv2.CV_64F).var()),
        "face_luma": float(region.mean()),
        "black_fraction": float((region < 8).mean()),
        "white_fraction": float((region > 247).mean()),
        "frame_luma": float(gray.mean()),
        "layout_seam": bool(
            vertical[30:130].max() > 75 or horizontal[25:95].max() > 75
        ),
        "dhash": f"{signature:064x}",
        "pixel_hash": hashlib.sha256(small.tobytes()).hexdigest(),
    }


class TemporalMetrics:
    """Approximate repeat evidence; low motion alone is not a confirmed freeze."""

    def __init__(self, config):
        self.config, self.previous, self.since = config, None, None

    def update(self, rgb, stamp, stop, *, small=None):
        import cv2
        import numpy as np

        if small is None:
            small = cv2.resize(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
                (160, 120),
                interpolation=cv2.INTER_AREA,
            )
        difference = (
            None
            if self.previous is None
            else float(np.abs(small.astype(float) - self.previous).mean())
        )
        if difference is not None and difference <= self.config.near_freeze_difference:
            self.since = stamp if self.since is None else self.since
        else:
            self.since = None
        self.previous = small.astype(float)
        return {
            "mean_frame_difference": difference,
            "repeated_seconds": 0 if self.since is None else stop - self.since,
        }


def scan_source(path, directory, start, end, info, config, detector):
    import cv2
    from scenedetect import AdaptiveDetector, ContentDetector, ThresholdDetector

    began = time.monotonic()
    content = (
        AdaptiveDetector(min_scene_len=1)
        if config.scene_detector == "adaptive"
        else ContentDetector(threshold=config.scene_threshold, min_scene_len=1)
    )
    fade = ThresholdDetector(threshold=config.fade_threshold, min_scene_len=1)
    frame_path = directory / "frames.jsonl.gz"
    temporary = directory / "frames.raw.partial.gz"
    blocked, cuts, times, first_time, last_end = [], [], [], None, start
    tracker, temporal = SubjectTracker(config), TemporalMetrics(config)
    dark_start, last_ms, count = None, -1, 0
    luminance = deque()
    fade_exit, fade_stable, previous_luma = None, 0, None
    reasons = Counter()
    timings = Counter()
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as handle:
        tick = time.monotonic()
        for rgb, timing in frames(path, start, end):
            timings["decode"] += time.monotonic() - tick
            tick = time.monotonic()
            stamp, stop = timing["time"], timing["end"]
            first_time = stamp if first_time is None else first_time
            times.append(stamp)
            # Channel reordering commutes with resize; avoid allocating a full-size BGR copy.
            scaled = cv2.cvtColor(
                cv2.resize(rgb, (320, 180), interpolation=cv2.INTER_AREA),
                cv2.COLOR_RGB2BGR,
            )
            cut_frames = content.process_frame(count, scaled)
            fade_cuts = fade.process_frame(count, scaled)
            for index in cut_frames + fade_cuts:
                if not 0 <= index < len(times):
                    raise ValueError(
                        "Scene detector returned an unmappable frame index"
                    )
                at = times[index]
                blocked.append(
                    [
                        max(start, at - config.transition_guard),
                        min(end, at + config.transition_guard),
                    ]
                )
                cuts.append(
                    {
                        "time": at,
                        "frame": index,
                        "type": "cut" if index in cut_frames else "fade",
                    }
                )
            if cut_frames or fade_cuts:
                detector.reset()
                tracker.reset()
                temporal = TemporalMetrics(config)
            flags = []
            if timing["corrupt"]:
                flags.append("corrupt_frame")
            if stop - stamp > config.timestamp_gap:
                flags.append("timestamp_gap")
            timestamp_ms = round(stamp * 1000)
            if timestamp_ms <= last_ms:
                raise ValueError(
                    "Frame timestamps collide at MediaPipe millisecond precision"
                )
            last_ms = timestamp_ms
            timings["scene_and_bookkeeping"] += time.monotonic() - tick
            tick = time.monotonic()
            detection = detector.detect(rgb, timestamp_ms)
            assessed = tracker.assess(detection, info["width"], info["height"], stamp)
            timings["detection_and_tracking"] += time.monotonic() - tick
            tick = time.monotonic()
            flags.extend(assessed["flags"])
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            small = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
            metrics = image_metrics(rgb, assessed["face"], gray=gray, small=small)
            metrics.update(temporal.update(rgb, stamp, stop, small=small))
            level = metrics["frame_luma"]
            while luminance and stamp - luminance[0][0] > 5:
                luminance.popleft()
            descending = bool(
                luminance
                and max(v for _, v in luminance) > config.fade_threshold + 10
                and max(v for _, v in luminance) - level > 10
            )
            if level <= config.fade_threshold and (
                level <= 4 or descending or dark_start is not None
            ):
                if dark_start is None:
                    # Walk back along the descending ramp, not only the dark midpoint.
                    dark_start, current, stable = stamp, level, 0
                    for prior_time, prior_luma in reversed(luminance):
                        if prior_luma < current - 2:
                            break
                        stable = stable + 1 if abs(prior_luma - current) < 1 else 0
                        if stable >= 3:
                            break
                        dark_start, current = prior_time, prior_luma
                fade_exit, fade_stable = None, 0
            elif dark_start is not None:
                fade_exit = stamp if fade_exit is None else fade_exit
                fade_stable = (
                    fade_stable + 1
                    if previous_luma is not None and abs(level - previous_luma) < 1
                    else 0
                )
                if fade_stable >= 3 or stamp - fade_exit >= 5:
                    blocked.append(
                        [
                            max(start, dark_start - config.transition_guard),
                            min(end, stop + config.transition_guard),
                        ]
                    )
                    dark_start, fade_exit = None, None
            luminance.append((stamp, level))
            previous_luma = level
            row = {
                **timing,
                **assessed,
                "frame_index": count,
                "boundary": bool(cut_frames or fade_cuts),
                "flags": flags,
                "metrics": metrics,
            }
            timings["image_metrics"] += time.monotonic() - tick
            tick = time.monotonic()
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            timings["serialization"] += time.monotonic() - tick
            last_end, count = stop, count + 1
            if count % 250 == 0:
                print(f"scan {stamp:.1f}/{end:.1f}s | {count} frames", flush=True)
            tick = time.monotonic()
    if not count:
        raise ValueError("No frames decoded in scan window")
    if dark_start is not None:
        blocked.append([max(start, dark_start - config.transition_guard), last_end])
    if last_end < end - max(0.1, 1 / info["fps"]):
        raise ValueError("Decode ended before requested scan coverage")
    # A bounded second pass can recover crop proposals between observed anchors.
    # Persist uncertainties; they must never become automatic continuity passes.
    uncertain, proposal, observed, invalid = [], [], [], []
    finalized = directory / "frames.partial.gz"
    tick = time.monotonic()
    with gzip.open(finalized, "wt", encoding="utf-8", compresslevel=1) as handle:
        for row in recover_rows(
            read_frames(temporary), config.tracking_gap, config.track_iou
        ):
            if row["flags"] or not row.get("region"):
                blocked.append([row["time"], row["end"]])
                invalid.append([row["time"], row["end"]])
            if row["uncertainties"]:
                uncertain.append([row["time"], row["end"]])
            if row["track_state"] == "recovered_proposal":
                proposal.append([row["time"], row["end"]])
            elif row["track_state"] == "observed":
                observed.append([row["time"], row["end"]])
            reasons.update(row["flags"] + row["uncertainties"])
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    finalized.replace(frame_path)
    temporary.unlink()
    timings["recovery_and_finalize"] += time.monotonic() - tick
    from .metadata import seconds

    result = {
        "start": first_time,
        "end": last_end,
        "requested": [start, end],
        "frame_count": count,
        "blocked": ranges_union([(a, b) for a, b in blocked if a < b]),
        "cuts": cuts,
        "reasons": dict(reasons),
        "elapsed_seconds": time.monotonic() - began,
        "coverage_method": "all_decoded_frames_actual_pts",
        "device": config.detector_device,
        "stage_seconds": dict(timings),
        "scene_detector": config.scene_detector,
        "framing_profile": config.framing_profile,
        "observed_seconds": seconds(observed),
        "recovered_proposal_seconds": seconds(proposal),
        "uncertain_seconds": seconds(uncertain),
        "invalid_seconds": seconds(invalid),
        "blocked_seconds": seconds(blocked),
    }
    result["frames_per_second"] = count / max(1e-9, result["elapsed_seconds"])
    try:
        import resource
        import sys

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["process_peak_memory_bytes"] = (
            peak if sys.platform == "darwin" else peak * 1024
        )
    except ImportError:
        result["process_peak_memory_bytes"] = None
    write_json(directory / "scan.json", result)
    return result


def clean_windows(frame_path, scan, config):
    """Single ordered read, bounded to one candidate; never rescan gzip per clean span."""
    spans = iter(subtract_ranges(scan["start"], scan["end"], scan["blocked"]))
    span = next(spans, None)
    if span is None:
        return
    geometry = config.geometry()
    current, region, track = [], None, None
    parent = fingerprint([str(frame_path.parent.name), *span])[:24]

    def completed():
        if current and current[-1]["end"] - current[0]["time"] >= config.min_seconds:
            yield current, fit_crop(region, scan["width"], scan["height"], geometry), parent

    for row in read_frames(frame_path):
        while span is not None and row["time"] >= span[1]:
            yield from completed()
            current, region, track = [], None, None
            span = next(spans, None)
            if span is not None:
                parent = fingerprint([str(frame_path.parent.name), *span])[:24]
        if span is None:
            return
        if row["time"] >= span[0] - 1e-7 and row["end"] <= span[1] + 1e-7:
            if row["flags"] or not row["region"]:
                raise ValueError("Blocked/invalid frame reached clean window selection")
            combined = union_box(region, row["region"]) if region else row["region"]
            fits = True
            try:
                fit_crop(combined, scan["width"], scan["height"], geometry)
            except ValueError:
                fits = False
            if current and (
                row["end"] - current[0]["time"] > config.max_seconds + 1e-7
                or row["track_id"] != track
                or not fits
            ):
                yield from completed()
                current, region = [], None
                combined = row["region"]
                fit_crop(combined, scan["width"], scan["height"], geometry)
            current.append(row)
            region, track = combined, row["track_id"]
    yield from completed()
