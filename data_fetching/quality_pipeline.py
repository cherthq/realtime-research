"""Quality-long orchestration; legacy pipeline stays available as smoke-test."""

from dataclasses import asdict, replace
from bisect import bisect_right
import importlib.metadata
import shutil
import sys
import time

from .core import file_hash, fingerprint, read_json, write_json
from .media import probe, render_clip, frame_png, validate_clip
from .metadata import check, disposition, interval_id, utc_now, validate_v2
from .pipeline import (
    code_fingerprint,
    source_record_path,
    validate_candidate,
    validate_source,
)
from .quality_audio import waveform_check
from .quality_backends import invoke, worker_job
from .quality_scan import (
    TemporalMetrics,
    clean_windows,
    frames,
    image_metrics,
    scan_source,
)
from .quality_tracking import SubjectTracker, mapped_region, region_inside


def containment(rows, crop):
    x, y, width, height = crop
    clearances = []
    for row in rows:
        if row.get("flags") or not row.get("region"):
            raise ValueError("Framing requires valid geometry on every frame")
        l, t, r, b = row["region"]
        margin = min(l - x, t - y, x + width - r, y + height - b)
        if margin < -1e-6:
            raise ValueError("Fixed crop truncates a required region")
        clearances.append(margin)
    if not clearances:
        raise ValueError("No framing coverage")
    return min(clearances)


def visual_checks(rows, config, duration):
    findings, layout = [], []
    for row in rows:
        m, a, b = row["metrics"], row["time"], row["end"]
        for code, flagged in (
            ("possible_face_blur", m["sharpness"] < config.blur_threshold),
            (
                "poor_face_exposure",
                m["face_luma"] < config.dark_threshold
                or m["face_luma"] > config.bright_threshold
                or m["black_fraction"] > 0.5
                or m["white_fraction"] > 0.5,
            ),
            (
                "possible_frozen_imagery",
                m.get("repeated_seconds", 0) >= config.freeze_seconds,
            ),
        ):
            if flagged:
                findings.append(
                    {"code": code, "start": a, "end": b, "origin": "heuristic"}
                )
        if m["layout_seam"]:
            layout.append(
                {
                    "code": "possible_panel_layout",
                    "start": a,
                    "end": b,
                    "origin": "heuristic",
                }
            )
    return {
        "visual_quality": check(
            "flag" if findings else "unverified",
            "full_frame_face_metrics",
            [(0, duration)],
            findings,
            reason="heuristic_thresholds_require_calibration_and_review",
        ),
        "layout": check(
            "flag" if layout else "unverified",
            "panel_seam_and_person_checks",
            [(0, duration)],
            layout,
            reason="layout_heuristic_not_semantic_proof",
        ),
        "appearance": check(
            "unverified",
            "human_review_required",
            reason="check_nontext_logos_filters_synthetic_and_background_defects",
        ),
    }


def rendered_tracks(video, directory, config, paths, source_rows=None, crop=None):
    """Check source projection AND independent output anatomy, without double margins."""
    from .vision import Detector
    import cv2

    info, rows = probe(video), []
    # Required torso was chosen in source space. On output, independently check
    # visible head/shoulders, not a newly expanded torso heuristic a second time.
    verification = replace(
        config,
        framing_profile="head_chest",
        chest_depth=0,
        torso_depth=0,
        torso_min_depth=0,
        crop_safety=0,
    )
    tracker, temporal = SubjectTracker(verification), TemporalMetrics(config)
    source_times = (
        [r["time"] - source_rows[0]["time"] for r in source_rows] if source_rows else []
    )
    detector = Detector(paths, config.geometry(), video=True, device=config.detector_device)
    try:
        for rgb, timing in frames(video, 0, info["duration"]):
            if (
                timing["corrupt"]
                or timing["end"] - timing["time"] > config.timestamp_gap
            ):
                raise ValueError(
                    "Corrupt output frame or unresolved output timestamp gap"
                )
            detection = detector.detect(rgb, round(timing["time"] * 1000))
            assessed = tracker.assess(
                detection, info["width"], info["height"], timing["time"]
            )
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            small = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
            metrics = image_metrics(rgb, assessed["face"], gray=gray, small=small)
            metrics.update(temporal.update(rgb, timing["time"], timing["end"], small=small))
            row = {
                **timing,
                **assessed,
                "metrics": metrics,
                "measured_core_region": assessed["core_region"],
            }
            if source_rows:
                index = max(0, bisect_right(source_times, timing["time"] + 0.002) - 1)
                source = source_rows[index]
                expected_end = source["end"] - source_rows[0]["time"]
                if timing["time"] > expected_end + 0.002:
                    raise ValueError(
                        "Rendered frame cannot be mapped to inspected source PTS"
                    )
                expected = mapped_region(
                    source["region"], crop, info["width"], info["height"]
                )
                row.update(
                    region=expected,
                    source_frame_index=source["frame_index"],
                    source_track_state=source["track_state"],
                    projected_core_region=mapped_region(
                        source["core_region"], crop, info["width"], info["height"]
                    ),
                )
                if not region_inside(expected, info["width"], info["height"], 0.01):
                    row["flags"].append("source_region_truncated_by_render")
                if source["track_state"] != "observed":
                    row["uncertainties"].append("source_visibility_not_observed")
            rows.append(row)
    finally:
        detector.close()
    if not rows or rows[0]["time"] > 0.1 or rows[-1]["end"] < info["duration"] - 0.1:
        raise ValueError("Incomplete rendered-frame inspection")
    write_json(directory / "output-tracks.json", rows)
    return rows


def make_candidate(
    root,
    source,
    original_record,
    original,
    directory,
    rows,
    crop,
    parent,
    scan_path,
    settings,
    config,
    registry,
    paths,
):
    started = time.monotonic()
    start, end = rows[0]["time"], rows[-1]["end"]
    duration = end - start
    stable = interval_id(
        original_record["sha256"],
        rows[0]["pts"],
        rows[-1]["end_pts"],
        rows[0]["time_base"],
    )
    processing_id = fingerprint(settings)
    clearance = containment(rows, crop)
    render_key = fingerprint(
        [
            original_record["sha256"],
            source.id,
            config.detector_device,
            start,
            end,
            crop,
            asdict(config.geometry()),
            settings["code"],
            settings["ffmpeg"],
            settings["packages"],
            settings["models"],
        ]
    )
    render_directory = root / "quality-renders" / render_key
    render_directory.mkdir(parents=True, exist_ok=True)
    render_complete = render_directory / "complete.json"
    if render_complete.exists():
        render_manifest = read_json(render_complete)
        for name, digest in render_manifest.items():
            if file_hash(render_directory / name) != digest:
                raise ValueError("Cached rendered media changed; left untouched")
        validate_clip(
            render_directory / "video.mp4",
            render_directory / "audio.wav",
            config.geometry(),
            duration,
        )
        output_rows = read_json(render_directory / "output-tracks.json")
    else:
        render_clip(
            original,
            render_directory,
            start,
            end,
            crop,
            config.geometry(),
            original_record["media"],
        )
        output_rows = rendered_tracks(
            render_directory / "video.mp4", render_directory, config, paths, rows, crop
        )
        render_manifest = {
            name: file_hash(render_directory / name)
            for name in (
                "video.mp4",
                "audio.wav",
                "reference.png",
                "output-tracks.json",
            )
        }
        write_json(render_complete, render_manifest)
    for name, digest in render_manifest.items():
        target = directory / name
        if target.exists() and file_hash(target) != digest:
            raise ValueError(
                "Existing candidate render differs from cache; left untouched"
            )
        if not target.exists():
            shutil.copy2(render_directory / name, target)
    performance = {"render_and_output_check_seconds": time.monotonic() - started}
    tick = time.monotonic()
    outputs = {
        "video": directory / "video.mp4",
        "audio": directory / "audio.wav",
        "reference_image": directory / "reference.png",
    }
    output_duration = probe(outputs["video"])["duration"]
    framing_failures = [
        {"code": reason, "start": r["time"], "end": r["end"], "origin": "model"}
        for r in output_rows
        for reason in r["flags"] + r.get("uncertainties", [])
    ]
    source_uncertainties = [
        {
            "code": reason,
            "start": r["time"] - start,
            "end": r["end"] - start,
            "origin": "source_tracking",
        }
        for r in rows
        for reason in r.get("uncertainties", [])
    ]
    checked_range = [(0, output_duration)]
    checks = {
        "continuity": check(
            "flag" if source_uncertainties else "pass",
            "actual_pts_all_frame_shot_track",
            [(start, end)],
            [
                {**f, "start": f["start"] + start, "end": f["end"] + start}
                for f in source_uncertainties
            ],
            coordinate_system="source_video_seconds",
            frame_count=len(rows),
        ),
        "framing": check(
            "flag" if framing_failures or source_uncertainties else "pass",
            "projected_source_region_and_independent_output_core",
            checked_range,
            framing_failures + source_uncertainties,
            min_source_clearance_px=clearance,
            source_frame_count=len(rows),
            output_frame_count=len(output_rows),
        ),
        "render_validation": check("pass", "ffmpeg_probe_decode", checked_range),
    }
    checks.update(visual_checks(output_rows, config, output_duration))
    checks["audio_quality"] = waveform_check(outputs["audio"])
    job = worker_job(
        outputs["video"],
        outputs["audio"],
        directory / "output-tracks.json",
        output_duration,
        config,
    )
    performance["metrics_and_audio_seconds"] = time.monotonic() - tick
    for name in ("ocr", "vad"):
        tick = time.monotonic()
        checks.update(invoke(name, registry, job, directory))
        performance[f"{name}_seconds"] = time.monotonic() - tick
    speech = checks["speech"].get("speech_ranges")
    for name in ("talknet", "syncnet"):
        tick = time.monotonic()
        checks.update(invoke(name, registry, {**job, "speech": speech}, directory))
        performance[f"{name}_seconds"] = time.monotonic() - tick
    tick = time.monotonic()
    # Sidecars contain raw measurements/windows; summary retains findings/coverage.
    write_json(directory / "quality-details.json", checks)
    for result in checks.values():
        result.pop("windows", None)
    for label, row in (
        ("first", output_rows[0]),
        ("middle", output_rows[len(output_rows) // 2]),
        ("last", output_rows[-1]),
    ):
        (directory / f"preview_{label}.png").write_bytes(
            frame_png(outputs["video"], row["time"], probe(outputs["video"]))
        )
    # Evidence for the worst flag from each check: still + playable contextual clip.
    evidence = []
    for name, result in checks.items():
        findings = result.get("findings", [])
        if not findings:
            continue
        at = max(0, min(output_duration - 0.05, findings[0]["start"]))
        evidence_path = directory / f"evidence_{name}.png"
        evidence_path.write_bytes(
            frame_png(outputs["video"], at, probe(outputs["video"]))
        )
        from .media import run

        preview = directory / f"evidence_{name}.mp4"
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-ss",
                str(max(0, at - 1)),
                "-i",
                outputs["video"],
                "-t",
                str(min(3, output_duration - max(0, at - 1))),
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-c:a",
                "aac",
                preview,
            ]
        )
        evidence.append(
            {
                "check": name,
                "time": at,
                "image": evidence_path.name,
                "video": preview.name,
            }
        )
    local_files = [
        p for p in directory.iterdir() if p.is_file() and p.name != "candidate.json"
    ]
    artifacts = {
        name: {"path": path.relative_to(root).as_posix(), "sha256": file_hash(path)}
        for name, path in outputs.items()
    }
    for path in local_files:
        if path not in outputs.values():
            artifacts["evidence_" + path.name] = {
                "path": path.relative_to(root).as_posix(),
                "sha256": file_hash(path),
            }
    artifacts["source_frame_scan"] = {
        "path": scan_path.relative_to(root).as_posix(),
        "sha256": file_hash(scan_path),
    }
    wide_fits = all(
        r["wide_region"]
        and crop[0] <= r["wide_region"][0]
        and crop[1] <= r["wide_region"][1]
        and r["wide_region"][2] <= crop[0] + crop[2]
        and r["wide_region"][3] <= crop[1] + crop[3]
        for r in rows
    )
    record = {
        "schema_version": 2,
        "id": directory.name,
        "source_id": source.id,
        "source": original_record,
        "interval_id": stable,
        "processing_id": processing_id,
        "parent_span_id": parent,
        "track_id": rows[0]["track_id"],
        "start": start,
        "end": end,
        "pts_bounds": [rows[0]["pts"], rows[-1]["end_pts"]],
        "time_base": rows[0]["time_base"],
        "crop": crop,
        "config": asdict(config.geometry()),
        "quality_config": config.to_dict(),
        "processing": settings,
        "checks": checks,
        "framing": "wider_torso_gestures" if wide_fits else "head_upper_chest",
        "requested_framing_profile": config.framing_profile,
        "tracking_summary": {
            "observed_frames": sum(r["track_state"] == "observed" for r in rows),
            "recovered_proposal_frames": sum(
                r["track_state"] == "recovered_proposal" for r in rows
            ),
            "uncertain_frames": sum(bool(r.get("uncertainties")) for r in rows),
        },
        "duration_bucket": "long"
        if duration >= config.long_seconds - 1e-6
        else "short",
        "speech_bucket": "unverified"
        if speech is None
        else "talking"
        if speech
        else "listening",
        "state": "candidate",
        "review_status": "pending",
        "qc_disposition": disposition(checks),
        "speaker_id": None,
        "conversation_id": None,
        "split": None,
        "artifacts": artifacts,
        "evidence": evidence,
        "created_at": utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "performance": {**performance, "evidence_and_metadata_seconds": time.monotonic() - tick},
    }
    validate_v2(record)
    write_json(directory / "candidate.json", record)
    return record


def process_quality(
    root, sources, config, registry, *, scan_seconds=300, max_clips=20, exclusions=(),
    jobs=1, persistent_workers=True,
):
    """Parent owns the dataset lock/report; spawned jobs have disjoint source artifacts."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    from .quality_sessions import worker_session

    if isinstance(jobs, bool) or not isinstance(jobs, int) or not 1 <= jobs <= 64:
        raise ValueError("jobs must be an integer in 1..64")
    sources = list(sources)
    if len({s.id for s in sources}) != len(sources):
        raise ValueError("Parallel source IDs must be unique")
    options = {"scan_seconds": scan_seconds, "max_clips": max_clips, "exclusions": list(exclusions)}
    if jobs == 1:
        with worker_session(persistent_workers):
            return _process_quality(root, sources, config, registry, **options)
    completed = {}
    report = []
    with ProcessPoolExecutor(max_workers=min(jobs, max(1, len(sources))),
                             mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = {executor.submit(_quality_task, root, source, config, registry,
                                   options, persistent_workers): source.id for source in sources}
        for future in as_completed(futures):
            source_id = futures[future]
            try:
                rows = future.result()
            except Exception as exc:
                rows = [{"source_id": source_id, "state": "failed", "error": str(exc)}]
            completed[source_id] = rows
            write_json(root / "reports" / "sources" / f"{source_id}.json", rows)
            report = [row for source in sources for row in completed.get(source.id, [])]
            write_json(root / "reports" / "quality-process.json", report)
    write_json(root / "reports" / "quality-process.json", report)
    return report


def _quality_task(root, source, config, registry, options, persistent_workers):
    from .quality_sessions import worker_session
    with worker_session(persistent_workers):
        return _process_quality(root, [source], config, registry, save_report=False, **options)


def _process_quality(
    root, sources, config, registry, *, scan_seconds=300, max_clips=20, exclusions=(),
    save_report=True,
):
    from .vision import Detector, model_paths, runtime_versions

    paths, hashes = model_paths(root)
    from .media import run

    settings = {
        "profile": "quality-long",
        "config": config.to_dict(),
        "models": hashes,
        "packages": {**runtime_versions(), "av": importlib.metadata.version("av")},
        "python": sys.version,
        "ffmpeg": run(["ffmpeg", "-version"]).splitlines()[0],
        "ffprobe": run(["ffprobe", "-version"]).splitlines()[0],
        "code": code_fingerprint(),
        "workers": {k: v["fingerprint"] for k, v in registry.items()},
        "exclusions": list(exclusions),
    }
    report = []
    for source in sources:
        source_started = time.monotonic()
        try:
            record = read_json(source_record_path(root, source))
            original = validate_source(root, source, record)
            info = record["media"]
            end = min(
                source.end if source.end is not None else info["duration"],
                source.start + scan_seconds,
            )
            scan_keys = (
                "width",
                "height",
                "face_confidence",
                "shoulder_confidence",
                "head_margin",
                "side_margin",
                "chest_depth",
                "torso_depth",
                "torso_min_depth",
                "scene_threshold",
                "fade_threshold",
                "track_iou",
                "timestamp_gap",
                "transition_guard",
                "framing_profile",
                "scene_detector",
                "detector_device",
                "crop_safety",
                "tracking_gap",
                "secondary_face_confidence",
                "near_freeze_difference",
            )
            source_key = fingerprint(
                [
                    record["sha256"],
                    source.id,  # Identical-byte reuploads must not race on scan artifacts.
                    source.start,
                    end,
                    {k: config.to_dict()[k] for k in scan_keys},
                    settings["models"],
                    settings["code"],
                    settings["packages"],
                ]
            )
            stage = root / "quality-scans" / source_key
            stage.mkdir(parents=True, exist_ok=True)
            if (stage / "complete.json").exists():
                cached = read_json(stage / "complete.json")
                if file_hash(stage / "frames.jsonl.gz") != cached["sha256"]:
                    raise ValueError("Source frame scan changed")
                scan = read_json(stage / "scan.json")
                if file_hash(stage / "scan.json") != cached["scan_sha256"]:
                    raise ValueError("Scan metadata changed")
            else:
                detector = Detector(paths, config.geometry(), video=True, device=config.detector_device)
                try:
                    scan = scan_source(
                        original, stage, source.start, end, info, config, detector
                    )
                finally:
                    detector.close()
                scan.update(source_sha256=record["sha256"], source_id=source.id)
                write_json(stage / "scan.json", scan)
                write_json(
                    stage / "complete.json",
                    {
                        "sha256": file_hash(stage / "frames.jsonl.gz"),
                        "scan_sha256": file_hash(stage / "scan.json"),
                    },
                )
            scan = {**scan, "width": info["width"], "height": info["height"]}
            from .metadata import finite, ranges_union

            blocked = list(scan["blocked"])
            for excluded in exclusions:
                if excluded.get("source_id") == source.id:
                    a, b = excluded.get("start"), excluded.get("end")
                    if (
                        not finite(a)
                        or not finite(b)
                        or not 0 <= a < b <= info["duration"]
                        or not excluded.get("reason")
                        or not excluded.get("evidence")
                    ):
                        raise ValueError(
                            "Exclusions require valid source bounds, reason and evidence"
                        )
                    blocked.append([a, b])
            scan["blocked"] = ranges_union(blocked)
            attempted = 0
            for rows, crop, parent in clean_windows(
                stage / "frames.jsonl.gz", scan, config
            ):
                if attempted >= max_clips:
                    break
                attempted += 1
                key = fingerprint(
                    [source_key, rows[0]["pts"], rows[-1]["end_pts"], settings]
                )
                directory = root / "clips" / f"q2_{source.id}_{key[:24]}"
                directory.mkdir(parents=True, exist_ok=True)
                if (directory / "candidate.json").exists():
                    candidate = read_json(directory / "candidate.json")
                    validate_v2(candidate)
                    validate_candidate(root, candidate)
                    cached = True
                else:
                    candidate = make_candidate(
                        root,
                        source,
                        record,
                        original,
                        directory,
                        rows,
                        crop,
                        parent,
                        stage / "frames.jsonl.gz",
                        settings,
                        config,
                        registry,
                        paths,
                    )
                    cached = False
                report.append(
                    {
                        "id": candidate["id"],
                        "state": candidate["state"],
                        "qc_disposition": candidate["qc_disposition"],
                        "cached": cached,
                    }
                )
                print(f"{candidate['id']}: {candidate['qc_disposition']}", flush=True)
            report.append(
                {
                    "source_id": source.id,
                    "state": "scanned",
                    "scan": scan,
                    "candidate_limit_reached": attempted >= max_clips,
                    "unscanned_tail_seconds": max(
                        0, (source.end or info["duration"]) - end
                    ),
                    "processing_elapsed_seconds": time.monotonic() - source_started,
                }
            )
        except Exception as exc:
            report.append(
                {"source_id": source.id, "state": "failed", "error": str(exc)}
            )
            print(f"{source.id}: quality processing failed: {exc}", flush=True)
    if save_report:
        write_json(root / "reports" / "quality-process.json", report)
    return report
