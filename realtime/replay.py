"""Frozen avatar-matched v2 audio; never modifies the fixed-voice v1 pack."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .audio import PCMResampler
from .config import ROOT
from .speech import synthesize_script, wav_bytes
from .voices import VoiceService, digest, save_json


async def freeze_pack(service: VoiceService, destination: Path) -> dict:
    destination = destination.resolve()
    original = service.settings.assets_dir.resolve()
    if (
        destination == original
        or destination.is_relative_to(original)
        or original.is_relative_to(destination)
    ):
        raise ValueError("The v2 pack must be separate from v1 assets")
    assignments = {
        avatar: service.resolve(avatar) for avatar in service.catalog.portraits
    }
    suite = json.loads(
        (ROOT / "configs/zero_shot_suite.json").read_text(encoding="utf-8")
    )
    inputs = {
        "assignments": assignments,
        "suite": suite,
        "images": {
            a: digest(service.catalog.image(a).read_bytes()) for a in assignments
        },
        "format": "mono-pcm16-48000",
        "preparation_revision": 1,
    }
    fingerprint = digest(inputs)
    destination.mkdir(parents=True, exist_ok=True)
    lock = destination / ".preparing"
    # Only the process that successfully creates this lock may remove it.
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(fingerprint)
    try:
        manifest_path = destination / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("input_sha256") != fingerprint:
                raise ValueError(
                    "Frozen inputs changed. Choose a new output directory; existing pack was preserved."
                )
        else:
            manifest = {
                "schema_version": 1,
                "suite_id": "facetime-avatar-voices-v2",
                "status": "preparing",
                "input_sha256": fingerprint,
                "inputs": inputs,
                "audio": {},
                "portraits": {},
            }
            save_json(manifest_path, manifest)
        for avatar in assignments:
            image = service.catalog.image(avatar).read_bytes()
            path = destination / "images" / f"{avatar}.png"
            path.parent.mkdir(exist_ok=True)
            if path.exists() and digest(path.read_bytes()) != digest(image):
                raise ValueError(
                    "Existing v2 portrait changed; use a new output directory"
                )
            path.write_bytes(image)
            manifest["portraits"][avatar] = {
                "path": str(path.relative_to(destination).as_posix()),
                "sha256": digest(image),
                "voice_selection": assignments[avatar],
                "audio_ids": [],
            }
            profile = assignments[avatar]["profile"]
            profile_hash = digest(profile)
            for scenario in suite["scenarios"]:
                audio_id = f"{profile_hash[:16]}-{scenario['id']}"
                manifest["portraits"][avatar]["audio_ids"].append(audio_id)
                relative = f"audio/{audio_id}.wav"
                path = destination / relative
                if audio_id in manifest["audio"]:
                    if (
                        not path.is_file()
                        or digest(path.read_bytes())
                        != manifest["audio"][audio_id]["sha256"]
                    ):
                        raise ValueError(
                            "Frozen waveform missing or modified; do not silently regenerate it"
                        )
                    continue
                parts, segments = [], []
                cursor = 0

                def append(kind, samples, **metadata):
                    nonlocal cursor
                    parts.append(samples)
                    segments.append(
                        {
                            "kind": kind,
                            "start_sample": cursor,
                            "end_sample": cursor + len(samples),
                            **metadata,
                        }
                    )
                    cursor += len(samples)

                def silence(seconds):
                    append("silence", np.zeros(round(seconds * 48_000), dtype=np.int16))

                silence(suite["audio_preparation"]["leading_silence_seconds"])
                for segment in scenario["segments"]:
                    if segment["kind"] == "silence":
                        silence(segment["duration_seconds"])
                        continue
                    speech = await synthesize_script(
                        service._client(),
                        profile["speech_model"],
                        profile["voice"],
                        segment["text"],
                        profile["instructions"],
                        scenario["tts_speed"],
                    )
                    resampler = PCMResampler(24_000, 48_000)
                    samples = np.concatenate(
                        (resampler.push(speech["pcm"]), resampler.finish())
                    )
                    append(
                        "speech",
                        samples,
                        transcript=speech["transcript"],
                        provider_model=speech["resolved_model"],
                    )
                silence(suite["audio_preparation"]["trailing_silence_seconds"])
                samples = np.concatenate(parts)
                audio = wav_bytes(samples.tobytes(), 48_000)
                path.parent.mkdir(exist_ok=True)
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(audio)
                temporary.replace(path)
                manifest["audio"][audio_id] = {
                    "path": relative,
                    "sha256": digest(audio),
                    "scenario_id": scenario["id"],
                    "profile_sha256": profile_hash,
                    "voice": profile["voice"],
                    "segments": segments,
                    "sample_rate": 48_000,
                    "samples": cursor,
                    "duration_seconds": cursor / 48_000,
                }
                save_json(manifest_path, manifest)
        manifest["status"] = "frozen"
        save_json(manifest_path, manifest)
        return manifest
    finally:
        lock.unlink()
