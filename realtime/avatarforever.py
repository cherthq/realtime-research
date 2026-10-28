"""Pinned native file-input smoke runner. This does not claim incremental inference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import wave

from scripts.baseline_geometry import load_geometry, validate_geometry

ROOT = Path(__file__).resolve().parents[1]
CODE_REVISION = "de4deb6a55c9567a17dad58d67390fe089993630"
WEIGHT_REVISION = "622ea82482c09e7fb2d8e29cb7ff1d009c820f05"
GEMMA_REVISION = "68f7ee4fbd59087436ada77ed2d62f373fdd4482"


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_installation(weights: Path) -> dict:
    source = ROOT / "vendor/avatarforever"
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if revision != CODE_REVISION or dirty:
        raise ValueError("AvatarForever source pin or tracked files changed")
    revisions = json.loads(
        (weights / "avatarforever-revisions.json").read_text(encoding="utf-8")
    )
    expected = {
        "LetsThink/AvatarForever": WEIGHT_REVISION,
        "google/gemma-3-12b-it-qat-q4_0-unquantized": GEMMA_REVISION,
    }
    if {r["repo_id"]: r["revision"] for r in revisions} != expected:
        raise ValueError("Weight download provenance does not match the pinned profile")
    return {"code_revision": revision, "weight_revisions": revisions}


def valid_frame_count(samples: int, rate: int) -> int:
    if samples <= 0 or rate <= 0:
        raise ValueError("Audio must contain valid samples")
    needed = math.ceil(samples * 25 / rate)
    return 8 * math.ceil(max(0, needed - 1) / 8) + 1


def build_command(
    python: str,
    image: Path,
    audio: Path,
    checkpoint_root: Path,
    output: Path,
    frames: int,
) -> list[str]:
    geometry = load_geometry()
    if (geometry["width"], geometry["height"]) != (640, 480):
        raise ValueError("This AvatarForever profile requires direct 640x480")
    if (frames - 1) % 8 or frames < 1:
        raise ValueError("AvatarForever frame count must follow 8n+1")
    return [
        python,
        str(ROOT / "vendor/avatarforever/inference.py"),
        "--distilled-checkpoint-path",
        str(checkpoint_root / "AvatarForever/avatarforever-ltx-2.3-22b.safetensors"),
        "--gemma-root",
        str(checkpoint_root / "gemma-3-12b-it-qat-q4_0-unquantized"),
        "--first-frame-condition-image-path",
        str(image),
        "--audio-path",
        str(audio),
        "--width",
        "640",
        "--height",
        "480",
        "--frame-rate",
        "25",
        "--num-frames",
        str(frames),
        "--seed",
        "0",
        "--ar-video-chunk-size",
        "4",
        "--ar-history-chunk-count",
        "1",
        "--ar-history-feature-cache",
        "--fast-infer",
        "--output-path",
        str(output),
    ]


def probe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def load_frozen_case(pack: Path, portrait_id: str, scenario_id: str) -> dict:
    pack = pack.resolve()
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["status"] != "frozen"
        or manifest["suite_id"] != "facetime-avatar-voices-v2"
    ):
        raise ValueError("Select a complete, frozen avatar-matched v2 pack")
    portrait = manifest["portraits"][portrait_id]
    records = [manifest["audio"][name] for name in portrait["audio_ids"]]
    record = next(item for item in records if item["scenario_id"] == scenario_id)

    def asset(relative):
        path = (pack / relative).resolve()
        if not path.is_relative_to(pack):
            raise ValueError("Manifest path leaves its pack directory")
        return path

    image, audio = asset(portrait["path"]), asset(record["path"])
    if file_hash(image) != portrait["sha256"] or file_hash(audio) != record["sha256"]:
        raise ValueError("Input hash mismatch; frozen pack was modified")
    with wave.open(str(audio), "rb") as source:
        samples, rate = source.getnframes(), source.getframerate()
        if (rate, source.getnchannels(), source.getsampwidth()) != (48000, 1, 2):
            raise ValueError("Expected 48kHz mono PCM16 master")
    if samples <= 0 or record.get("samples") != samples:
        raise ValueError("Frozen waveform sample count changed")
    return {
        "image": image,
        "audio": audio,
        "portrait": portrait,
        "record": record,
        "samples": samples,
        "rate": rate,
        "manifest_sha256": file_hash(pack / "manifest.json"),
    }


def run(args):
    pack = args.pack.resolve()
    case = load_frozen_case(pack, args.portrait, args.scenario)
    image, audio = case["image"], case["audio"]
    portrait, record = case["portrait"], case["record"]
    samples, rate = case["samples"], case["rate"]
    frames = valid_frame_count(samples, rate)
    destination = args.output.resolve()
    command = build_command(
        args.python,
        image,
        audio,
        args.weights.resolve(),
        destination / "native.mp4",
        frames,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "command": command,
                    "valid_audio_seconds": samples / rate,
                    "conditioning_frames": frames,
                    "qualification": "offline_only",
                },
                indent=2,
            )
        )
        return
    source = ROOT / "vendor/avatarforever"
    installation = verify_installation(args.weights)
    revision, revisions = (
        installation["code_revision"],
        installation["weight_revisions"],
    )
    if destination.exists():
        raise ValueError("Output directory already exists; choose a new run ID")
    destination.mkdir(parents=True)
    metadata = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": "avatarforever",
        "qualification": "offline_only",
        "command": command,
        "code_revision": revision,
        "weight_revisions": revisions,
        "manifest_sha256": file_hash(pack / "manifest.json"),
        "portrait": args.portrait,
        "scenario": args.scenario,
        "audio_sha256": record["sha256"],
        "image_sha256": portrait["sha256"],
        "voice_selection": portrait["voice_selection"],
        "geometry": load_geometry(),
        "valid_samples": samples,
        "sample_rate": rate,
        "conditioning_frames": frames,
        "timing_note": "Process timing includes model loading and compilation; not warm or realtime latency.",
    }

    def save():
        (destination / "run.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    save()
    started = time.monotonic()
    try:
        with (destination / "inference.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                command, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        metadata["cold_inclusive_seconds"] = time.monotonic() - started
        raw = destination / "native.mp4"
        raw_probe = probe(raw)
        validate_geometry(
            next(s for s in raw_probe["streams"] if s["codec_type"] == "video"),
            load_geometry(),
        )
        output = destination / "output.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(raw),
                "-i",
                str(audio),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-t",
                f"{samples / rate:.9f}",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
        )
        result = probe(output)
        video = next(s for s in result["streams"] if s["codec_type"] == "video")
        validate_geometry(video, load_geometry())
        duration = float(video.get("duration", result["format"]["duration"]))
        if abs(duration - samples / rate) > 1 / 25 + 0.02:
            raise ValueError(
                "Video duration does not cover the master audio within one frame"
            )
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                "-i",
                str(output),
                "-f",
                "null",
                "-",
            ],
            check=True,
        )
        metadata.update(
            status="completed", probe=result, output_sha256=file_hash(output)
        )
    except BaseException as exc:
        metadata.update(status="failed", failure_type=type(exc).__name__)
        raise
    finally:
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--portrait", choices=["ref01", "ref02", "ref03"], default="ref01"
    )
    parser.add_argument("--scenario", default="s01_greeting")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable in the dedicated GPU environment",
    )
    parser.add_argument("--dry-run", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
