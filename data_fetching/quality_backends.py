"""Explicit local-model registry and isolated, fail-closed worker protocol."""

import json
from pathlib import Path
import subprocess

from .core import file_hash, fingerprint, read_json, write_json
from .metadata import check, validate_check

BACKEND_CHECKS = {"ocr": ("text_overlay",), "vad": ("speech",),
                  "talknet": ("active_speaker",), "syncnet": ("lip_sync",)}


def load_registry(path=None):
    if path is None:
        return {}
    registry = read_json(path)
    if registry.get("schema_version") != 1 or not isinstance(registry.get("backends"), dict):
        raise ValueError("Worker registry requires schema_version 1 and backends object")
    base = Path(path).resolve().parent
    result = {}
    for name, entry in registry["backends"].items():
        if name not in BACKEND_CHECKS:
            raise ValueError(f"Unknown backend: {name}")
        if not isinstance(entry, dict) or not isinstance(entry.get("enabled", False), bool):
            raise ValueError(f"{name}: enabled must be a boolean")
        if not entry.get("enabled", False):
            continue
        entry = dict(entry)
        if entry.get("terms_reviewed") is not True or not isinstance(entry.get("license_evidence"), str) or not entry["license_evidence"].strip():
            raise ValueError(f"{name}: review code/weight terms and record license_evidence first")
        timeout = entry.get("timeout_seconds", 1800)
        if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 86400:
            raise ValueError(f"{name}: timeout_seconds must be an integer in 1..86400")
        python = (base / entry["python"]).resolve()
        if not python.is_file():
            raise ValueError(f"{name}: worker Python missing: {python}")
        entry["python"] = str(python)
        assets = entry.get("assets", [])
        if not assets:
            raise ValueError(f"{name}: explicit local assets and SHA-256 values required")
        for asset in assets:
            asset["path"] = str((base / asset["path"]).resolve())
            if file_hash(asset["path"]) != asset["sha256"]:
                raise ValueError(f"{name}: model asset hash mismatch")
        for key in ("weights", "model_dir", "repo"):
            if key in entry:
                entry[key] = str((base / entry[key]).resolve())
        if "weights" in entry and entry["weights"] not in [a["path"] for a in assets]:
            raise ValueError(f"{name}: weights must be listed in assets")
        if "model_dir" in entry:
            actual = {str(p.resolve()) for p in Path(entry["model_dir"]).rglob("*") if p.is_file()}
            if actual != {a["path"] for a in assets}:
                raise ValueError(f"{name}: model directory inventory differs from hashed assets")
        if "repo" in entry:
            def git(*args):
                run = subprocess.run(["git", "-C", entry["repo"], *args], capture_output=True, text=True, check=True)
                return run.stdout.strip()
            if git("rev-parse", "HEAD") != entry.get("revision"):
                raise ValueError(f"{name}: upstream revision mismatch")
            if git("status", "--porcelain", "--untracked-files=all"):
                raise ValueError(f"{name}: upstream checkout must be clean (keep weights outside repo)")
        try:
            package_probe = subprocess.run([entry["python"], "-c",
                "import importlib.metadata,json,sys; print(json.dumps({'python':sys.version,'packages':sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions())}))"],
                capture_output=True, text=True, timeout=30, check=True)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError(f"{name}: worker interpreter/package probe failed") from exc
        entry["runtime"] = json.loads(package_probe.stdout)
        entry["fingerprint"] = fingerprint(entry)
        result[name] = entry
    return result


def invoke(name, registry, job, directory):
    keys = BACKEND_CHECKS[name]
    if name not in registry:
        return {k: check("unverified", name, reason="backend_not_configured") for k in keys}
    entry = registry[name]
    job_id = fingerprint({"job": job, "backend": entry["fingerprint"], "name": name})
    request = {"protocol_version": 1, "job_id": job_id, "backend": name, "settings": entry, **job}
    request_path = directory / f"{name}-request.json"
    response_path = directory / f"{name}-{job_id[:16]}-response.json"
    write_json(request_path, request)
    if not response_path.exists():
        command = [entry["python"], "-B", "-m", "data_fetching.quality_worker", "--request", str(request_path),
                   "--response", str(response_path)]
        try:
            from .quality_sessions import active_session
            session = active_session()
            if session is not None and name in {"ocr", "vad"}:
                session.invoke(name, entry, request_path, response_path, directory / f"{name}.log")
            else:
                with (directory / f"{name}.log").open("w", encoding="utf-8") as log:
                    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], stdout=log, stderr=log,
                                   timeout=entry.get("timeout_seconds", 1800), check=True)
        except (OSError, subprocess.SubprocessError) as exc:
            return {k: check("error", name, reason=type(exc).__name__, log=f"{name}.log") for k in keys}
    try:
        response = read_json(response_path)
        if response.get("protocol_version") != 1 or response.get("job_id") != job_id or response.get("backend") != name:
            raise ValueError("Worker response/job mismatch")
        result = response["checks"]
        if set(result) != set(keys):
            raise ValueError("Worker returned unexpected checks")
        for value in result.values():
            validate_check(value)
            for a, b in value["analyzed_ranges"]:
                if a < 0 or b > job["duration"] + 0.05:
                    raise ValueError("Worker coverage escapes clip")
            if value["status"] == "pass":
                from .metadata import seconds
                if name == "vad" and seconds(value["analyzed_ranges"]) < job["duration"] - 0.05:
                    raise ValueError("VAD pass does not cover clip")
                if name in {"talknet", "syncnet"}:
                    from .quality_audio import speech_coverage
                    if job.get("speech") is None or speech_coverage(job["speech"], value["analyzed_ranges"]) < 0.999:
                        raise ValueError("Audio-visual pass does not cover speech")
            value["backend_revision"] = entry["fingerprint"]
            value["worker_evidence"] = response_path.name
        return result
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {k: check("error", name, reason=f"invalid_response: {exc}") for k in keys}


def worker_job(video, audio, tracks, duration, config, speech=None):
    return {"video": str(video.resolve()), "audio": str(audio.resolve()), "tracks": str(tracks.resolve()),
            "duration": duration, "config": config.to_dict(), "speech": speech,
            "input_hashes": {"video": file_hash(video), "audio": file_hash(audio), "tracks": file_hash(tracks)}}
