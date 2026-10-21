"""Resumable sampled screening of whole scenes, preserving long source spans.

This finds candidate boundaries only. It cannot establish that every intervening
frame is clean, that speech is natural, or that the visible person is speaking.
"""

from dataclasses import asdict, replace
import math
from pathlib import Path
import time
import subprocess
import tempfile

from .core import Config, assess_samples, file_hash, fingerprint, read_json, write_json
from .vision import Detector, model_paths, runtime_versions, scene_intervals


PROFILES = {"head_upper_chest": 0.25, "wider_torso_gestures": 1.0}


def clean_runs(samples, start, end, sample_fps, framing, minimum=5):
    """Never bridge a failing sample. Boundary samples are conservatively excluded.

    Chunk edges are computational checkpoints, not scene boundaries. Run this
    once over the complete scene to retain uninterrupted spans above 300 sec.
    """
    result, opened, last = [], None, None
    step = 1 / sample_fps
    for sample in samples:
        good = sample["profiles"].get(framing) is True
        if last is not None and sample["time"] - last > step * 1.6:
            good = False
        if good and opened is None:
            opened = max(start, sample["time"])
        if not good and opened is not None:
            stop = max(opened, sample["time"] - step)
            if stop - opened >= minimum:
                result.append({"start": opened, "end": stop, "framing": framing,
                  "method": "sampled_video_geometry_only",
                  "issues": ["overlays_unverified", "compression_filters_unverified",
                             "speech_and_speaker_match_unverified", "full_interval_uninspected",
                             "sampled_boundary_timing_unverified", "stable_framing_unverified"]})
            opened = None
        last = sample["time"]
    if opened is not None and last is not None:
        stop = min(end, last)  # Do not extrapolate beyond the final observed frame.
        if stop - opened >= minimum:
            result.append({"start": opened, "end": stop, "framing": framing,
              "method": "sampled_video_geometry_only",
              "issues": ["overlays_unverified", "compression_filters_unverified",
                         "speech_and_speaker_match_unverified", "full_interval_uninspected",
                         "sampled_boundary_timing_unverified", "stable_framing_unverified"]})
    return result


def sample_window(original, start, end, info, detector, config):
    import numpy as np

    # Decode once per checkpoint, rather than seeking and decoding a keyframe
    # group for every sample. Keep landmark coordinates on the original grid.
    width = min(960, info["width"])
    width -= width % 2
    height = int(round(info["height"] * width / info["width"] / 2)) * 2
    error_file = tempfile.TemporaryFile()
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
      "-seek_timestamp", "1", "-ss", str(info["video_start"] + start), "-i", str(original),
      "-t", str(end - start), "-an", "-vf", f"fps={config.sample_fps},scale={width}:{height}",
      "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=error_file)
    samples = []
    decoded = 0
    try:
        for index in range(math.ceil((end - start) * config.sample_fps)):
            timestamp = start + index / config.sample_fps
            needed = width * height * 3
            data = bytearray()
            while len(data) < needed:
                chunk = process.stdout.read(needed - len(data))
                if not chunk:
                    break
                data.extend(chunk)
            entry = {"time": timestamp, "profiles": {}, "reasons": {}}
            if len(data) != needed:
                entry["error"] = "frame_decode_failed"
                samples.append(entry)
                continue
            decoded += 1
            detected = detector.detect(np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3))
            sx, sy = info["width"] / width, info["height"] / height
            for face in detected["faces"]:
                face["box"] = [face["box"][0] * sx, face["box"][1] * sy, face["box"][2] * sx, face["box"][3] * sy]
            for pose in detected["poses"]:
                for point in pose:
                    point["x"] *= sx
                    point["y"] *= sy
            for name, depth in PROFILES.items():
                assessment = assess_samples([dict(detected)], info["width"], info["height"],
                                            replace(config, torso_depth=depth, valid_fraction=1.0))
                entry["profiles"][name] = assessment["crop"] is not None
                entry["reasons"][name] = assessment["rejection_reason"] or assessment["sample_rejections"]
            samples.append(entry)
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=30)
        error_file.close()
    if not decoded:
        raise ValueError("No frames decoded; scan checkpoint not completed")
    return samples


def scan_source(queue, source, *, deadline, window_seconds=120):
    original = Path(source["original"])
    if file_hash(original) != source["sha256"]:
        raise ValueError("Original hash changed; refusing scan")
    import json
    info = json.loads(source["media"])
    paths, hashes = model_paths(queue.root.parent / "facetime")
    config = Config(sample_fps=2.0, valid_fraction=1.0)
    code = {name: file_hash(Path(__file__).parent / name) for name in ("core.py", "vision.py", "long_scan.py")}
    settings = {"config": asdict(config), "profiles": PROFILES, "models": hashes,
                "packages": runtime_versions(), "code": code, "scan_version": 1}
    revision = fingerprint(settings)
    folder = queue.root / "scans" / source["id"] / revision
    folder.mkdir(parents=True, exist_ok=True)
    scenes_path = folder / "scenes.json"
    if not scenes_path.exists():
        scenes = scene_intervals(original, 0, info["duration"], config)
        write_json(scenes_path, {"settings": settings, "scenes": scenes})
    scenes = read_json(scenes_path)["scenes"]
    detector = Detector(paths, config)
    attempted = 0
    try:
        for scene_start, scene_end in scenes:
            final_path = folder / f"scene_{scene_start:.6f}_{scene_end:.6f}.json"
            if final_path.exists():
                completed = read_json(final_path)
                queue.complete_window(source["id"], revision, scene_start, scene_end, completed["candidates"])
                continue
            # A sub-sample scene (often the final container/frame-duration
            # remainder) may emit no frame through FFmpeg's fps filter. It
            # cannot support a candidate; retain empty coverage without claiming
            # that it was visually inspected or retrying it forever.
            if scene_end - scene_start < 1 / config.sample_fps:
                write_json(final_path, {"candidates": [], "reason": "scene_shorter_than_sample_period"})
                queue.complete_window(source["id"], revision, scene_start, scene_end, [])
                continue
            samples = []
            cursor = scene_start
            while cursor < scene_end - 1e-6:
                stop = min(scene_end, cursor + window_seconds)
                path = folder / f"samples_{cursor:.6f}_{stop:.6f}.json"
                if path.exists():
                    chunk = read_json(path)
                else:
                    if time.monotonic() >= deadline:
                        return {"source_id": source["id"], "state": "checkpointed", "windows_sampled": attempted}
                    chunk = sample_window(original, cursor, stop, info, detector, config)
                    write_json(path, chunk)  # Atomic checkpoint before coverage commit.
                    attempted += 1
                queue.complete_window(source["id"], revision, cursor, stop, [])
                samples.extend(chunk)
                cursor = stop
            candidates = []
            for framing in PROFILES:
                candidates.extend(clean_runs(samples, scene_start, scene_end, config.sample_fps, framing))
            # Same timestamps in two profiles represent one interval; prefer wider.
            candidates = list({(r["start"], r["end"]): r for r in candidates}.values())
            write_json(final_path, {"candidates": candidates})
            queue.complete_window(source["id"], revision, scene_start, scene_end, candidates)
        with queue.transaction():
            queue.db.execute("UPDATE sources SET state='screened',error=NULL WHERE id=?", (source["id"],))
        return {"source_id": source["id"], "state": "screened", "windows_sampled": attempted}
    finally:
        detector.close()
