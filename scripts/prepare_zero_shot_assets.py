"""Freeze local avatar inputs; never loads avatar weights or starts cloud jobs."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import wave

import numpy as np
from PIL import Image
from scipy.signal import resample_poly


REPO = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    # Exclusive creation protects already-frozen inputs from accidental overwrite.
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def asset_path(root, relative):
    relative = Path(relative)
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Asset path escapes package: {relative}")
    return path


def samples_for(seconds, rate):
    value = float(seconds) * rate
    if not math.isfinite(value) or value <= 0 or abs(value - round(value)) > 1e-6:
        raise ValueError("Silence must contain a positive integer number of samples")
    return round(value)


def to_pcm(audio, source_rate, target_rate, gain):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("TTS returned empty, non-mono or non-finite audio")
    divisor = math.gcd(source_rate, target_rate)
    audio = resample_poly(audio, target_rate // divisor, source_rate // divisor)
    audio = audio * gain
    if np.max(np.abs(audio)) >= 1.0:
        raise ValueError("Audio would clip; version the suite to change its fixed gain")
    pcm = np.rint(audio * 32767).astype("<i2")
    if not np.any(pcm):
        raise ValueError("TTS returned silent speech")
    return pcm


def assemble(scenario, settings, synthesize):
    rate = settings["master_sample_rate_hz"]
    pieces, timeline = [], []
    cursor = 0

    def append(pcm, kind, **details):
        nonlocal cursor
        stop = cursor + len(pcm)
        timeline.append(
            {
                "kind": kind,
                "start_sample": cursor,
                "end_sample": stop,
                "start_seconds": cursor / rate,
                "end_seconds": stop / rate,
                **details,
            }
        )
        pieces.append(pcm)
        cursor = stop

    def silence(seconds, role):
        append(np.zeros(samples_for(seconds, rate), dtype="<i2"), "silence", role=role)

    silence(settings["leading_silence_seconds"], "leading")
    for segment in scenario["segments"]:
        if segment["kind"] == "silence":
            silence(segment["duration_seconds"], "scripted_pause")
        elif segment["kind"] == "speech" and segment.get("text", "").strip():
            audio, source_rate = synthesize(segment["text"], scenario["tts_speed"])
            append(
                to_pcm(audio, source_rate, rate, settings["fixed_gain"]),
                "speech",
                text=segment["text"],
                tts_speed=scenario["tts_speed"],
                source_sample_rate_hz=source_rate,
            )
        else:
            raise ValueError(f"Invalid segment: {segment}")
    silence(settings["trailing_silence_seconds"], "trailing")
    return np.concatenate(pieces), timeline


def check_audio(pcm, timeline, scenario, settings):
    expected = [
        {"kind": "silence", "duration_seconds": settings["leading_silence_seconds"]},
        *scenario["segments"],
        {"kind": "silence", "duration_seconds": settings["trailing_silence_seconds"]},
    ]
    if len(timeline) != len(expected):
        raise ValueError("Timeline segment count mismatch")
    cursor = 0
    for actual, requested in zip(timeline, expected):
        start, stop = actual["start_sample"], actual["end_sample"]
        if (
            start != cursor
            or not start < stop <= len(pcm)
            or actual["kind"] != requested["kind"]
        ):
            raise ValueError("Invalid/non-contiguous timeline")
        rate = settings["master_sample_rate_hz"]
        if (
            actual["start_seconds"] != start / rate
            or actual["end_seconds"] != stop / rate
        ):
            raise ValueError("Timeline seconds disagree with sample indices")
        if actual["kind"] == "silence":
            if stop - start != samples_for(requested["duration_seconds"], rate):
                raise ValueError("Incorrect pause length")
            if np.any(pcm[start:stop]):
                raise ValueError("Annotated silence is not exactly zero PCM")
        elif (
            not np.any(pcm[start:stop])
            or actual["text"] != requested["text"]
            or actual["tts_speed"] != scenario["tts_speed"]
        ):
            raise ValueError("Speech content/settings mismatch or silent speech")
        cursor = stop
    if cursor != len(pcm) or np.any(np.abs(pcm.astype(np.int32)) >= 32767):
        raise ValueError("Truncated timeline or clipped PCM")


def image_records(root, portrait_spec):
    records = []
    for portrait in portrait_spec["portraits"]:
        path = asset_path(root, portrait["path"])
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            if width * 3 != height * 4 or min(width, height) < 480:
                raise ValueError(f"Expected a high-resolution 4:3 portrait: {path}")
        records.append(
            {**portrait, "width": width, "height": height, "sha256": sha256(path)}
        )
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("Duplicate portrait IDs")
    return records


def prepare(root, suite_path, portrait_path, model_dir):
    from kokoro_onnx import Kokoro
    import onnxruntime as ort

    suite, portraits = load_json(suite_path), load_json(portrait_path)
    settings = suite["audio_preparation"]
    for field in ("provider", "model_revision", "voice_id"):
        if not settings.get(field):
            raise ValueError(f"Missing TTS {field}")
    if importlib.metadata.version("kokoro-onnx") != settings["wrapper_version"]:
        raise ValueError("Install the pinned input-preparation dependencies")
    if (root / "assets.json").exists() or (root / "audio").exists():
        raise FileExistsError(
            "Inputs/audio already exist; validate or use a new versioned folder"
        )
    images = image_records(root, portraits)
    if len(images) != suite["reference_image_count"]:
        raise ValueError("Incorrect reference image count")
    checkpoints = settings["checkpoints"]
    for item in checkpoints:
        if sha256(model_dir / item["filename"]) != item["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {item['filename']}")
    # Explicit CPU provider and bounded threads: no GPU or cloud runtime needed.
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_dir / checkpoints[0]["filename"]),
        options,
        providers=["CPUExecutionProvider"],
    )
    speed_input = next(item for item in session.get_inputs() if item.name == "speed")
    if speed_input.type not in {"tensor(float)", "tensor(double)", "tensor(float16)"}:
        raise ValueError("Model cannot preserve the fractional s04 speed setting")
    tts = Kokoro.from_session(session, str(model_dir / checkpoints[1]["filename"]))

    def synthesize(text, speed):
        return tts.create(
            text,
            voice=settings["voice_id"],
            speed=speed,
            lang=settings["tts_language"],
            trim=settings["trim_tts_segment_edges"],
            sentence_pause=settings["sentence_pause_seconds"],
            clause_pause=settings["clause_pause_seconds"],
        )

    (root / "audio").mkdir()
    audio_records = []
    for scenario in suite["scenarios"]:
        print(f"Synthesizing {scenario['id']} with {settings['voice_id']}", flush=True)
        pcm, timeline = assemble(scenario, settings, synthesize)
        check_audio(pcm, timeline, scenario, settings)
        relative = f"audio/{scenario['id']}.wav"
        with asset_path(root, relative).open("xb") as stream:
            with wave.open(stream, "wb") as output:
                output.setparams(
                    (
                        1,
                        2,
                        settings["master_sample_rate_hz"],
                        0,
                        "NONE",
                        "not compressed",
                    )
                )
                output.writeframes(pcm.tobytes())
        speech_samples = sum(
            item["end_sample"] - item["start_sample"]
            for item in timeline
            if item["kind"] == "speech"
        )
        words = sum(
            len(item["text"].split()) for item in timeline if item["kind"] == "speech"
        )
        audio_records.append(
            {
                "scenario_id": scenario["id"],
                "path": relative,
                "sha256": sha256(root / relative),
                "samples": len(pcm),
                "duration_seconds": len(pcm) / settings["master_sample_rate_hz"],
                "speech_region_words_per_minute": words
                * 60
                * settings["master_sample_rate_hz"]
                / speech_samples,
                "peak_absolute_pcm": int(np.max(np.abs(pcm.astype(np.int32)))),
                "timeline": timeline,
            }
        )
    manifest = []
    for image in images:
        for audio in audio_records:
            manifest.append(
                {
                    "id": f"{image['id']}__{audio['scenario_id']}",
                    "scenario_id": audio["scenario_id"],
                    "reference_image": image["path"],
                    "audio": audio["path"],
                    "prompt": suite["avatar_prompt"],
                    "split": "val",
                    "source": "synthetic_image_and_fixed_synthetic_speech",
                    "license": {
                        "image": "generated_asset_subject_to_provider_terms",
                        "tts_weights": settings["model_license"],
                    },
                    "reference_image_sha256": image["sha256"],
                    "audio_sha256": audio["sha256"],
                    "preprocessing": "original_image; fixed_gain; scipy.resample_poly; PCM16; exact_zero_pauses",
                    "tts_provenance": "assets.json#tts",
                    "silence_intervals": [
                        item for item in audio["timeline"] if item["kind"] == "silence"
                    ],
                }
            )
    with (root / "inputs.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        stream.writelines(
            json.dumps(item, ensure_ascii=False) + "\n" for item in manifest
        )
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "kokoro-onnx",
            "numpy",
            "onnxruntime",
            "scipy",
            "Pillow",
            "espeakng-loader",
            "phonemizer",
        )
    }
    write_json(
        root / "assets.json",
        {
            "schema_version": 1,
            "suite_id": suite["suite_id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "suite_sha256": sha256(suite_path),
            "portrait_spec_sha256": sha256(portrait_path),
            "inputs_sha256": sha256(root / "inputs.jsonl"),
            "review_status": "technical_checks_only_user_listening_review_pending",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "package_versions": versions,
            "tts": {
                **settings,
                "providers": session.get_providers(),
                "resampling": "scipy.signal.resample_poly default window=('kaiser', 5.0), padtype=constant",
                "pcm_conversion": "rint(float_audio * fixed_gain * 32767), little-endian int16",
                "timing_note": "Speech regions include natural TTS pauses; not forced phoneme alignments.",
            },
            "images": images,
            "audio": audio_records,
        },
    )
    validate(root, suite_path, portrait_path)


def validate(root, suite_path, portrait_path):
    suite = load_json(suite_path)
    assets = load_json(root / "assets.json")
    if assets["suite_sha256"] != sha256(suite_path) or assets[
        "portrait_spec_sha256"
    ] != sha256(portrait_path):
        raise ValueError("Input specification changed since freeze")
    if assets["tts"] != {
        **suite["audio_preparation"],
        **{
            key: assets["tts"][key]
            for key in ("providers", "resampling", "pcm_conversion", "timing_note")
        },
    }:
        raise ValueError("TTS provenance mismatch")
    if assets["images"] != image_records(root, load_json(portrait_path)):
        raise ValueError("Portrait provenance/content mismatch")
    if len(assets["images"]) != suite["reference_image_count"]:
        raise ValueError("Incorrect image count")
    scenarios = {item["id"]: item for item in suite["scenarios"]}
    if len(assets["audio"]) != len(scenarios) or {
        item["scenario_id"] for item in assets["audio"]
    } != set(scenarios):
        raise ValueError("Missing/duplicate scenario audio")
    for item in assets["audio"]:
        path = asset_path(root, item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Changed audio: {path.name}")
        with wave.open(str(path), "rb") as audio:
            if (
                audio.getnchannels(),
                audio.getsampwidth(),
                audio.getframerate(),
                audio.getcomptype(),
            ) != (1, 2, 48000, "NONE"):
                raise ValueError("Expected mono 48kHz PCM16 WAV")
            pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
        if len(pcm) != item["samples"] or item["duration_seconds"] != len(pcm) / 48000:
            raise ValueError("Audio duration/sample count mismatch")
        check_audio(
            pcm,
            item["timeline"],
            scenarios[item["scenario_id"]],
            suite["audio_preparation"],
        )
    if sha256(root / "inputs.jsonl") != assets["inputs_sha256"]:
        raise ValueError("Changed input manifest")
    rows = [
        json.loads(line)
        for line in (root / "inputs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    expected = {
        (image["path"], audio["path"])
        for image in assets["images"]
        for audio in assets["audio"]
    }
    if (
        len(rows) != len(expected)
        or {(row["reference_image"], row["audio"]) for row in rows} != expected
    ):
        raise ValueError("Input matrix is incomplete or duplicated")
    for row in rows:
        if row["split"] != "val" or row["prompt"] != suite["avatar_prompt"]:
            raise ValueError("Input split/prompt mismatch")
        for field in ("reference_image", "audio"):
            if sha256(asset_path(root, row[field])) != row[f"{field}_sha256"]:
                raise ValueError("Input row hash mismatch")
        audio = next(item for item in assets["audio"] if item["path"] == row["audio"])
        if row["scenario_id"] != audio["scenario_id"] or row["silence_intervals"] != [
            item for item in audio["timeline"] if item["kind"] == "silence"
        ]:
            raise ValueError("Input scenario/silence annotation mismatch")
    print(
        f"PASS: {len(assets['images'])} portraits, {len(assets['audio'])} WAVs, {len(rows)} input pairs; hashes and exact pauses verified."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", type=Path, default=REPO / "configs/zero_shot_suite.json"
    )
    parser.add_argument(
        "--portraits", type=Path, default=REPO / "configs/zero_shot_portraits.json"
    )
    parser.add_argument(
        "--output", type=Path, default=REPO / "data/eval/facetime-zero-shot-v1"
    )
    parser.add_argument(
        "--model-dir", type=Path, default=REPO / "checkpoints/kokoro-v1.0"
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate(args.output, args.suite, args.portraits)
    else:
        prepare(args.output, args.suite, args.portraits, args.model_dir)


if __name__ == "__main__":
    main()
