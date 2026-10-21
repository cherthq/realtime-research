"""Fetch, processing, and explicit human-review export stages."""

from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from importlib.metadata import version
import os
from pathlib import Path
import shutil
import sys
from urllib.request import urlopen

from . import __version__
from .core import (Config, SPLITS, file_hash, fingerprint, intervals, assess_samples,
                   read_json, read_jsonl, within, write_json, write_jsonl)
from .media import probe, render_clip, run, validate_clip


MODEL_URLS = {
    "face": "https://storage.googleapis.com/mediapipe-models/face_detector/"
            "blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}


@contextmanager
def dataset_lock(root):
    """Single writer per dataset. Never remove another process's stale lock."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".pipeline.lock"
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(f"Dataset busy: {path}. If a process crashed, verify it stopped before removing its lock.") from exc
    try:
        with handle:
            handle.write(f"pid={os.getpid()}\n")
        yield
    finally:
        path.unlink(missing_ok=True)


def install_models(root):
    """Explicit network operation; save hashes and verify them on every use."""
    manifest_path = root / "models" / "models.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    for name, url in MODEL_URLS.items():
        relative = "models/" + url.rsplit("/", 1)[1]
        destination = within(root, relative)
        if name in manifest:
            record = manifest[name]
            if record["url"] != url or record["path"] != relative or file_hash(destination) != record["sha256"]:
                raise ValueError(f"Existing {name} detector differs from recorded artifact; left untouched")
            continue
        if destination.exists():
            raise ValueError(f"Unrecorded detector exists: {destination}; left untouched")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        size = 0
        with urlopen(url, timeout=60) as response, temporary.open("wb") as handle:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                size += len(chunk)
                if size > 50 * 1024 * 1024:
                    raise ValueError("Unexpected detector size (>50 MiB)")
                handle.write(chunk)
        if size < 1000:
            raise ValueError("Detector download unexpectedly small")
        digest = file_hash(temporary)
        temporary.replace(destination)
        manifest[name] = {"url": url, "path": relative, "sha256": digest}
        write_json(manifest_path, manifest)
    return manifest


def import_talkinghead(ids_file, split, limit):
    from .core import VIDEO_ID

    rows, seen = [], set()
    for line in Path(ids_file).read_text(encoding="utf-8-sig").splitlines():
        video_id = line.strip()
        if not video_id:
            continue
        if not VIDEO_ID.fullmatch(video_id):
            raise ValueError("Expected *_video_ids.txt, not the *_video_tubes.txt crop table")
        if video_id in seen:
            continue
        seen.add(video_id)
        rows.append({"id": video_id, "url": f"https://www.youtube.com/watch?v={video_id}",
                     "source": "TalkingHead-1KH", "split": split,
                     "license": "unknown",
                     "license_evidence": "Upstream reports CC BY 3.0; verify this source individually: "
                     "https://github.com/tcwang0509/TalkingHead-1KH#licenses"})
        if len(rows) >= limit:
            break
    return rows


def source_record_path(root, source):
    return root / "sources" / f"{source.id}.json"


def validate_source(root, source, record):
    if record["definition"] != source.to_dict():
        raise ValueError(f"{source.id}: source definition changed; use a new source id or dataset root")
    path = within(root, record["original"])
    if file_hash(path) != record["sha256"]:
        raise ValueError(f"{source.id}: original file changed or is incomplete; left untouched")
    if source.path and file_hash(source.path) != record["sha256"]:
        raise ValueError(f"{source.id}: local input changed since fetch; use a new id")
    return path


def fetch_one(root, source):
    record_path = source_record_path(root, source)
    if record_path.exists():
        record = read_json(record_path)
        validate_source(root, source, record)
        return "cached"
    # Definition-specific directories prevent partial downloads from another URL
    # or interval being reused under a colliding user-chosen id.
    directory = root / "originals" / source.id / fingerprint(source.to_dict())[:16]
    directory.mkdir(parents=True, exist_ok=True)
    info = {}
    if source.path:
        source_path = Path(source.path)
        if not source_path.is_file():
            raise ValueError(f"Local source missing: {source_path}")
        extension = source_path.suffix.lower()
        if extension not in {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}:
            raise ValueError("Unsupported local media extension")
        original = directory / ("source" + extension)
        temporary = directory / ("source" + extension + ".part")
        shutil.copyfile(source_path, temporary)
        temporary.replace(original)
    else:
        from yt_dlp import YoutubeDL

        options = {"format": "bv[height<=1080]+ba/b[height<=1080]",
                   "outtmpl": str(directory / "source.%(ext)s"), "merge_output_format": "mkv",
                   "noplaylist": True, "continuedl": True, "overwrites": False,
                   "retries": 5, "fragment_retries": 5, "concurrent_fragment_downloads": 2,
                   "socket_timeout": 30, "cachedir": False, "quiet": False,
                   "writeinfojson": True}
        # Python API does not read CLI config. Explicitly enable an installed
        # supported runtime instead of requiring a new global Deno installation.
        if shutil.which("deno"):
            options["js_runtimes"] = {"deno": {}}
        elif shutil.which("node"):
            options["js_runtimes"] = {"node": {}}
        with YoutubeDL(options) as downloader:
            downloaded = downloader.extract_info(source.url, download=True)
        # Keep attribution fields, not expiring signed media URLs or HTTP headers.
        info = {key: downloaded.get(key) for key in
                ("id", "title", "uploader", "uploader_id", "channel_url", "webpage_url",
                 "license", "upload_date", "duration", "description")}
        info["yt_dlp_version"] = version("yt-dlp")
        originals = [p for p in directory.iterdir()
                     if p.stem == "source" and p.suffix in {".mkv", ".mp4", ".webm", ".mov"}]
        if len(originals) != 1:
            raise ValueError("Download did not produce exactly one merged source file")
        original = originals[0]
    media_info = probe(original)
    if media_info["audio_index"] is None:
        raise ValueError("Source contains no audio stream")
    if source.start >= media_info["duration"] or (source.end is not None and source.end > media_info["duration"] + 0.05):
        raise ValueError("Requested interval is outside source video duration")
    write_json(directory / "attribution.json", info)
    write_json(record_path, {"definition": source.to_dict(),
                             "original": original.relative_to(root).as_posix(),
                             "sha256": file_hash(original), "media": media_info,
                             "attribution": info})
    return "downloaded" if source.url else "copied"


def fetch_sources(root, sources):
    report = []
    for source in sources:
        try:
            status = fetch_one(root, source)
            report.append({"id": source.id, "status": status})
        except Exception as exc:
            report.append({"id": source.id, "status": "failed", "error": str(exc)})
        print(f"{source.id}: {report[-1]['status']}", flush=True)
    write_json(root / "reports" / "fetch.json", report)
    return report


def code_fingerprint():
    return fingerprint({p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")})


def artifact_paths(root, record):
    return {name: within(root, entry["path"]) for name, entry in record["artifacts"].items()}


def validate_candidate(root, record):
    if record.get("state") != "candidate":
        raise ValueError("Only successful candidates can be exported")
    paths = artifact_paths(root, record)
    for name, path in paths.items():
        if file_hash(path) != record["artifacts"][name]["sha256"]:
            raise ValueError(f"Candidate artifact changed: {path}")
    validate_clip(paths["video"], paths["audio"], Config(**record["config"]), record["end"] - record["start"])
    return paths


def process_sources(root, sources, config, *, max_clips=20, scan_seconds=300):
    from .vision import (Detector, model_paths, runtime_versions, sample_clip,
                         save_previews, scene_intervals)

    paths, hashes = model_paths(root)
    versions = runtime_versions()
    runtime = {"packages": versions, "python": sys.version.split()[0],
               "ffmpeg": run(["ffmpeg", "-version"]).splitlines()[0],
               "ffprobe": run(["ffprobe", "-version"]).splitlines()[0]}
    settings = {"config": asdict(config), "models": hashes, "runtime": runtime,
                "code": code_fingerprint(), "version": __version__}
    detector = Detector(paths, config)
    report = []
    try:
        for source in sources:
            try:
                original_record = read_json(source_record_path(root, source))
                original = validate_source(root, source, original_record)
                info = probe(original)
                end = min(source.end if source.end is not None else info["duration"],
                          source.start + scan_seconds)
                if source.start >= end:
                    raise ValueError("Empty processing interval")
                source_key = fingerprint({"source": original_record, "settings": settings})
                # Cache scene boundaries for this exact media/config scan window.
                scene_key = fingerprint({"source": source_key, "start": source.start, "end": end})
                scene_path = root / "scenes" / f"{source.id}_{scene_key[:16]}.json"
                if scene_path.exists():
                    scenes = read_json(scene_path)
                else:
                    scenes = scene_intervals(original, source.start, end, config)
                    write_json(scene_path, scenes)
                for index, (start, stop) in enumerate(intervals(scenes, source.start, end, config)):
                    if index >= max_clips:
                        break
                    key = fingerprint({"source_key": source_key, "start": start, "end": stop})
                    clip_id = f"{source.id}_{round(start * 1000)}_{round(stop * 1000)}_{key[:12]}"
                    directory = root / "clips" / clip_id
                    metadata_path = directory / "candidate.json"
                    if metadata_path.exists():
                        record = read_json(metadata_path)
                        if record["fingerprint"] != key:
                            raise ValueError("Candidate fingerprint mismatch; left untouched")
                        if record["state"] == "candidate":
                            validate_candidate(root, record)
                        elif record["state"] != "rejected":
                            raise ValueError("Invalid candidate state")
                        report.append({"id": clip_id, "state": record["state"], "cached": True,
                                       "reason": record.get("rejection_reason")})
                        continue
                    directory.mkdir(parents=True, exist_ok=True)
                    detection_path = directory / "detections.json"
                    if detection_path.exists():
                        detection_record = read_json(detection_path)
                        if detection_record["fingerprint"] != key:
                            raise ValueError("Detection fingerprint mismatch")
                        samples = detection_record["samples"]
                    else:
                        samples = sample_clip(original, start, stop, info, config, detector)
                        write_json(detection_path, {"fingerprint": key, "samples": samples})
                    assessment = assess_samples(samples, info["width"], info["height"], config)
                    record = {"id": clip_id, "source_id": source.id, "fingerprint": key,
                              "start": start, "end": stop, "config": asdict(config),
                              "processing": settings, "source": original_record,
                              "review_status": "pending", **assessment}
                    artifacts = {"detections": detection_path}
                    # Decode failures must not prevent a useful rejection record.
                    try:
                        previews = save_previews(original, directory, samples, assessment["crop"], info)
                        artifacts.update({f"preview_{i}": p for i, p in enumerate(previews)})
                    except (RuntimeError, ValueError) as exc:
                        record["preview_error"] = str(exc)
                        if assessment["crop"]:
                            raise
                    if assessment["crop"]:
                        artifacts.update(render_clip(original, directory, start, stop, assessment["crop"], config, info))
                        record["state"] = "candidate"
                    else:
                        record["state"] = "rejected"
                    record["artifacts"] = {name: {"path": path.relative_to(root).as_posix(),
                                                        "sha256": file_hash(path)}
                                           for name, path in artifacts.items()}
                    # Written last: a crash cannot create a completed candidate.
                    write_json(metadata_path, record)
                    report.append({"id": clip_id, "state": record["state"],
                                   "reason": record["rejection_reason"]})
                    print(f"{clip_id}: {record['state']}", flush=True)
            except Exception as exc:
                report.append({"source_id": source.id, "state": "failed", "error": str(exc)})
                print(f"{source.id}: failed: {exc}", flush=True)
    finally:
        detector.close()
    write_json(root / "reports" / "process.json", report)
    return report


def candidate_records(root):
    return [read_json(path) for path in sorted((root / "clips").glob("*/candidate.json"))]


def review_template(root):
    return [{"id": row["id"], "status": "pending", "source_approved": False,
             "license": row["source"]["definition"].get("license", "unknown"),
             "license_evidence": row["source"]["definition"].get("license_evidence", ""),
             "split": row["source"]["definition"].get("split"),
             "speaker_id": row["source"]["definition"].get("speaker_id"),
             "conversation_id": row["source"]["definition"].get("conversation_id"),
             "notes": "Review torso visibility, identity, camera-facing appearance, speech and lip-sync"}
            for row in candidate_records(root) if row["state"] == "candidate"]


def build_export(root, reviews_path):
    candidates = {row["id"]: row for row in candidate_records(root)}
    seen, groups, output, approved_intervals = set(), {}, [], set()
    for review in read_jsonl(reviews_path):
        clip_id = review.get("id")
        if not isinstance(clip_id, str) or clip_id in seen or clip_id not in candidates:
            raise ValueError(f"Unknown or duplicate review id: {clip_id}")
        seen.add(clip_id)
        if review.get("status") not in {"pending", "approved", "rejected"}:
            raise ValueError(f"Invalid review status: {clip_id}")
        if review["status"] != "approved":
            continue
        record = candidates[clip_id]
        if review.get("source_approved") is not True:
            raise ValueError(f"{clip_id}: source permission approval required")
        if not isinstance(review.get("license"), str) or not isinstance(review.get("license_evidence"), str):
            raise ValueError(f"{clip_id}: license and license_evidence must be strings")
        license_name = review["license"].strip()
        evidence = review["license_evidence"].strip()
        if license_name.lower() in {"", "unknown", "pending", "unverified"} or not evidence:
            raise ValueError(f"{clip_id}: verified license and evidence required")
        split = review.get("split")
        if split not in SPLITS:
            raise ValueError(f"{clip_id}: train/val/test assignment required")
        # No automatic identity recognition: reviewers must assign these groups.
        for field in ("speaker_id", "conversation_id"):
            value = review.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{clip_id}: reviewer-assigned {field} required")
        for field, value in (("source_id", record["source_id"]),
                             ("original_sha256", record["source"]["sha256"]),
                             ("speaker_id", review["speaker_id"]),
                             ("conversation_id", review["conversation_id"])):
            group = field, value.strip()
            if group in groups and groups[group] != split:
                raise ValueError(f"Cross-split leakage: {field}={value}")
            groups[group] = split
        interval_key = (record["source"]["sha256"], record["start"], record["end"])
        if interval_key in approved_intervals:
            raise ValueError(f"{clip_id}: duplicate source interval from another processing revision")
        approved_intervals.add(interval_key)
        validate_candidate(root, record)
        definition = record["source"]["definition"]
        row = {"id": clip_id, "split": split, "source": definition["url"] or definition["path"],
               "license": license_name, "license_evidence": evidence,
               "speaker_id": review["speaker_id"], "conversation_id": review["conversation_id"]}
        for name in ("video", "audio", "reference_image"):
            row[name] = record["artifacts"][name]["path"]
        output.append(row)
    return output


def summary(root):
    records = candidate_records(root)
    return {"counts": dict(Counter(row["state"] for row in records)),
            "rejections": dict(Counter(row["rejection_reason"] for row in records if row["state"] == "rejected")),
            "candidate_hours": sum(row["end"] - row["start"] for row in records
                                   if row["state"] == "candidate") / 3600,
            "note": "Candidates are NOT approved training data. Includes all retained processing revisions."}
