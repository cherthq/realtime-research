"""Local-only OCR/VAD/TalkNet/SyncNet adapters. Run in explicitly configured envs.

No upstream demo entrypoints are executed: several demos download weights or
delete scratch directories. Requests and responses are versioned JSON files.
"""

import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from .core import file_hash, fingerprint, read_json, write_json
from .media import run
from .metadata import check, seconds
from .quality_audio import (
    score_findings,
    speech_coverage,
    speech_segments,
    speech_windows,
)

# Imported upstream code must not dirty its pinned checkout with bytecode files.
sys.dont_write_bytecode = True
_MODEL_CACHE = None


def cached_model(kind, settings, create):
    """Only --serve retains models; recurrent VAD state remains per-request."""
    if _MODEL_CACHE is None:
        return create()
    key = (kind, fingerprint(settings))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE.clear()
        _MODEL_CACHE[key] = create()
    return _MODEL_CACHE[key]


@contextmanager
def offline_inference():
    """Block Python network access, including incidental dependency downloads.

    This is not an OS security sandbox. FFmpeg inputs are pre-hashed local files.
    """

    def blocked(*args, **kwargs):
        raise RuntimeError("Offline inference: provision local dependencies explicitly")

    with (
        patch("socket.socket.connect", blocked),
        patch("socket.socket.connect_ex", blocked),
        patch("socket.getaddrinfo", blocked),
    ):
        yield


def vad(job):
    import numpy as np
    import onnxruntime as ort

    # Explicit decoding; no torch.hub/package helper that may retrieve a model.
    pcm = run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-i",
            job["audio"],
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "pipe:1",
        ],
        binary=True,
    )
    audio = np.frombuffer(pcm, dtype="<f4")
    session = cached_model("vad", job["settings"], lambda: ort.InferenceSession(
        job["settings"]["weights"], providers=["CPUExecutionProvider"]
    ))
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, 64), dtype=np.float32)
    windows = []
    for index in range(0, len(audio), 512):
        block = audio[index : index + 512]
        if len(block) < 512:
            block = np.pad(block, (0, 512 - len(block)))
        block = block.reshape(1, -1)
        probability, state = session.run(
            None,
            {
                "input": np.concatenate([context, block], axis=1),
                "state": state,
                "sr": np.array(16000, dtype=np.int64),
            },
        )
        context = block[:, -64:]
        a, b = index / 16000, min(job["duration"], (index + 512) / 16000)
        probability = float(probability.reshape(-1)[0])
        if a >= b:
            break
        windows.append({"start": a, "end": b, "probability": probability})
    speech = speech_segments(windows, job["duration"], job["config"])
    return {
        "speech": check(
            "pass" if speech else "flag",
            "silero_vad_onnx",
            [(0, job["duration"])],
            speech_ranges=speech,
            speech_ratio=seconds(speech) / job["duration"],
            windows=windows,
        )
    }


def ocr(job):
    import cv2
    from .quality_scan import frames

    # model_dir is mandatory and its complete inventory has been hash-checked.
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    settings = job["settings"]
    fonts = [
        settings.get("font_file", ""),
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    font = next((Path(p) for p in fonts if p and Path(p).is_file()), None)
    if font is None:
        raise ValueError(
            "OCR needs a local font_file; automatic font downloads are disabled"
        )
    # PaddleX imports visualization fonts even for detection-only inference.
    os.environ["PADDLE_PDX_LOCAL_FONT_FILE_PATH"] = str(font.resolve())
    from paddlex import create_model

    model = cached_model("ocr", settings, lambda: create_model(
        model_dir=settings["model_dir"],
        model_name=settings["model_name"],
        device=settings.get("device", "cpu"),
    ))
    windows, findings, next_sample = [], [], 0.0
    from .quality_text import classify_text

    tracks = read_json(Path(job["tracks"]))
    track_index, previous = 0, None
    for rgb, timing in frames(job["video"], 0, job["duration"]):
        import numpy as np

        small = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (160, 120)).astype(
            float
        )
        change = (
            previous is not None
            and float(np.abs(small - previous).mean())
            > job["config"]["ocr_burst_difference"]
        )
        previous = small
        if (
            not change
            and timing["time"] + 1e-7 < next_sample
            and timing["end"] < job["duration"] - 0.1
        ):
            continue
        next_sample = timing["time"] + 1 / job["config"]["ocr_fps"]
        polygons = []
        for prediction in model.predict(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), batch_size=1
        ):
            value = prediction.json
            value = json.loads(value) if isinstance(value, str) else value
            value = value.get("res", value)
            for polygon, score in zip(
                value.get("dt_polys", []), value.get("dt_scores", [])
            ):
                if float(score) >= job["config"]["ocr_min_confidence"]:
                    while (
                        track_index + 1 < len(tracks)
                        and tracks[track_index + 1]["time"] <= timing["time"]
                    ):
                        track_index += 1
                    track = tracks[track_index] if tracks else {}
                    polygons.append(
                        {
                            "polygon": polygon,
                            "confidence": float(score),
                            **classify_text(
                                polygon,
                                rgb.shape[1],
                                rgb.shape[0],
                                track.get("face"),
                                track.get("region"),
                            ),
                        }
                    )
        windows.append(
            {"start": timing["time"], "end": timing["end"], "polygons": polygons}
        )
        if polygons:
            findings.append(
                {
                    "code": "suspected_text_not_overlay_proof",
                    "start": timing["time"],
                    "end": timing["end"],
                    "origin": "model",
                    "polygons": polygons,
                }
            )
    inspected = [[w["start"], w["end"]] for w in windows]
    return {
        "text_overlay": check(
            "flag" if findings else "unverified",
            "paddle_text_detection",
            inspected,
            findings,
            windows=windows,
            sampling_fps=job["config"]["ocr_fps"],
            reason="sampled_text_detection_cannot_certify_graphic_absence",
        )
    }


def proxy(job, start, end, directory):
    """Fixed tracked-face analysis proxy, never the exported training video."""
    tracks = read_json(Path(job["tracks"]))
    measured = [
        r
        for r in tracks
        if r["time"] < end
        and r["end"] > start
        and r.get("face")
        and r.get("track_state", "observed") == "observed"
    ]
    required = [
        (max(start, a), min(end, b))
        for a, b in job.get("speech", [[start, end]])
        if max(start, a) < min(end, b)
    ]
    if (
        speech_coverage(
            required, [(max(start, r["time"]), min(end, r["end"])) for r in measured]
        )
        < 0.999
    ):
        raise ValueError(
            "Missing measured face coverage during speech; interpolation is not evidence"
        )
    boxes = [r["face"] for r in measured]
    if not boxes:
        raise ValueError("Missing face-track coverage for speech window")
    l, t = min(b[0] for b in boxes), min(b[1] for b in boxes)
    r, b = max(b[2] for b in boxes), max(b[3] for b in boxes)
    from .media import probe

    media = probe(job["video"])
    size = int(math.ceil(max(r - l, b - t) * 1.4 / 2) * 2)
    if size > min(media["width"], media["height"]):
        raise ValueError("Face analysis square cannot fit source")
    x = int(max(0, min(media["width"] - size, (l + r - size) / 2)) // 2 * 2)
    y = int(max(0, min(media["height"] - size, (t + b - size) / 2)) // 2 * 2)
    if not x <= l < r <= x + size or not y <= t < b <= y + size:
        raise ValueError("Face proxy would truncate tracked face")
    video, audio = directory / "face.avi", directory / "audio.wav"
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-i",
            job["video"],
            "-ss",
            str(start),
            "-t",
            str(end - start),
            "-vf",
            f"crop={size}:{size}:{x}:{y},scale=224:224,fps=25",
            "-c:v",
            "mjpeg",
            "-q:v",
            "2",
            "-c:a",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            video,
        ]
    )
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-i",
            video,
            "-vn",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            audio,
        ]
    )
    return (
        video,
        audio,
        {
            "start": start,
            "end": end,
            "fps": 25,
            "crop": [x, y, size, size],
            "offset_convention": "syncnet_vshift_minus_best_index_frames_at_25fps",
        },
    )


def load_talknet(settings):
    import torch

    sys.path.insert(0, settings["repo"])
    # Construct only model + classifier; upstream talkNet() forces CUDA and an optimizer.
    from model.talkNetModel import talkNetModel
    from loss import lossAV

    device = settings.get("device", "cpu")
    model, classifier = talkNetModel().to(device).eval(), lossAV().to(device).eval()
    state = torch.load(settings["weights"], map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(
        {
            k.removeprefix("model."): v
            for k, v in state.items()
            if k.startswith("model.")
        },
        strict=True,
    )
    classifier.load_state_dict(
        {
            k.removeprefix("lossAV."): v
            for k, v in state.items()
            if k.startswith("lossAV.")
        },
        strict=True,
    )
    return model, classifier, device


def talknet_score(model, classifier, device, video, audio):
    import cv2
    import numpy as np
    import python_speech_features
    import torch
    from scipy.io import wavfile

    rate, waveform = wavfile.read(audio)
    features = python_speech_features.mfcc(
        waveform, rate, numcep=13, winlen=0.025, winstep=0.010
    )
    capture, images = cv2.VideoCapture(str(video)), []
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(cv2.resize(image, (224, 224)), cv2.COLOR_BGR2GRAY)
            images.append(gray[56:168, 56:168])
    finally:
        capture.release()
    length = min(len(images), len(features) // 4)
    if length < 25:
        raise ValueError("Speech window too short for TalkNet")
    with torch.inference_mode():
        a = torch.tensor(
            features[: length * 4], dtype=torch.float32, device=device
        ).unsqueeze(0)
        v = torch.tensor(
            np.stack(images[:length]), dtype=torch.float32, device=device
        ).unsqueeze(0)
        a, v = model.forward_cross_attention(
            model.forward_audio_frontend(a), model.forward_visual_frontend(v)
        )
        scores = classifier(model.forward_audio_visual_backend(a, v), labels=None)
    values = (
        scores.detach().cpu().numpy().reshape(-1)
        if hasattr(scores, "detach")
        else np.asarray(scores).reshape(-1)
    )
    return float(np.median(values)), float(np.min(values)), values.tolist()


def audiovisual(job, kind):
    config, settings = job["config"], job["settings"]
    key = "active_speaker" if kind == "talknet" else "lip_sync"
    if job.get("speech") is None:
        return {key: check("unverified", kind, reason="speech_check_unavailable")}
    speech = job["speech"]
    windows = speech_windows(
        job["duration"], speech, config["window_seconds"], config["window_stride"]
    )
    if not windows:
        return {key: check("unverified", kind, reason="no_supported_speech_windows")}
    if kind == "talknet":
        model, classifier, device = load_talknet(settings)
    else:
        sys.path.insert(0, settings["repo"])
        from SyncNetInstance import SyncNetInstance

        model = SyncNetInstance(device=settings.get("device", "cpu"))
        model.loadParameters(settings["weights"])
    results, findings, inspected = [], [], []
    with tempfile.TemporaryDirectory(prefix="facetime-av-") as temp:
        scratch = Path(temp)
        for index, (a, b) in enumerate(windows):
            directory = scratch / str(index)
            directory.mkdir()
            video, audio, mapping = proxy(job, a, b, directory)
            if kind == "talknet":
                score, minimum, frame_scores = talknet_score(
                    model, classifier, device, video, audio
                )
                bad = score_findings(
                    frame_scores,
                    a,
                    speech,
                    config["active_min_score"],
                    config["active_bad_seconds"],
                )
                row = {
                    "start": a,
                    "end": b,
                    "score": score,
                    "minimum_score": minimum,
                    "frame_scores": frame_scores,
                    "bad_speech_ranges": bad,
                    "proxy": mapping,
                }
                flagged = bool(bad)
                code = "possible_offscreen_or_mismatched_speech"
            else:
                options = SimpleNamespace(
                    tmp_dir=str(directory), reference="sync", batch_size=20, vshift=10
                )
                offset, confidence, distances = model.evaluate(options, str(video))
                lag, confidence = float(offset) * 40.0, float(confidence)
                row = {
                    "start": a,
                    "end": b,
                    "offset_ms": lag,
                    "confidence": confidence,
                    "proxy": mapping,
                }
                flagged = (
                    abs(lag) > config["sync_max_ms"]
                    or confidence < config["sync_min_confidence"]
                )
                code = "possible_lip_sync_mismatch_or_low_confidence"
            inspected.append([a, b])
            results.append(row)
            if flagged:
                findings.append({"start": a, "end": b, "code": code, "origin": "model"})
    coverage = speech_coverage(speech, inspected)
    # Scores stay flags until checkpoint-specific calibration is explicitly recorded.
    calibrated = settings.get("calibration_evidence") and settings.get(
        "sync_sign_validated" if kind == "syncnet" else "score_validated"
    )
    status = (
        "flag"
        if findings
        else "pass"
        if calibrated and coverage >= 0.999
        else "unverified"
    )
    return {
        key: check(
            status,
            kind,
            inspected,
            findings,
            windows=results,
            speech_coverage_fraction=coverage,
            calibration_evidence=settings.get("calibration_evidence"),
            reason=None
            if status == "pass"
            else "review_scores_and_calibration_required",
        )
    }


def execute(job):
    if job.get("protocol_version") != 1:
        raise ValueError("Unsupported worker protocol")
    for key, digest in job["input_hashes"].items():
        if file_hash(job[key]) != digest:
            raise ValueError("Worker input changed")
    for asset in job["settings"]["assets"]:
        if file_hash(asset["path"]) != asset["sha256"]:
            raise ValueError("Worker weights changed")
    kind = job["backend"]
    if kind not in {"vad", "ocr", "talknet", "syncnet"}:
        raise ValueError("Unsupported worker backend")
    with offline_inference():
        checks = (
            vad(job)
            if kind == "vad"
            else ocr(job)
            if kind == "ocr"
            else audiovisual(job, kind)
        )
    packages = {}
    for name in (
        "numpy",
        "torch",
        "onnxruntime",
        "paddleocr",
        "paddlex",
        "paddlepaddle",
        "python-speech-features",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "protocol_version": 1,
        "job_id": job["job_id"],
        "backend": kind,
        "checks": checks,
        "runtime": {"python": sys.version, "packages": packages},
    }


def main():
    global _MODEL_CACHE
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--request", type=Path)
    cli.add_argument("--response", type=Path)
    cli.add_argument("--serve", action="store_true", help="Read request/response paths as JSON lines on stdin")
    args = cli.parse_args()
    if args.serve:
        if args.request or args.response:
            cli.error("--serve cannot be combined with --request/--response")
        _MODEL_CACHE = {}
        for line in sys.stdin:
            envelope = json.loads(line)
            request, response = Path(envelope["request"]), Path(envelope["response"])
            if response.exists():
                raise ValueError("Refusing to overwrite worker response")
            job = read_json(request)
            try:
                result = execute(job)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                _MODEL_CACHE.clear()
                result = {"protocol_version": 1, "job_id": job.get("job_id"),
                          "backend": job.get("backend"), "checks": {},
                          "error": f"{type(exc).__name__}: {exc}"}
            write_json(response, result)
        _MODEL_CACHE.clear()
        return
    if not args.request or not args.response:
        cli.error("--request and --response are required without --serve")
    if args.response.exists():
        raise ValueError("Refusing to overwrite worker response")
    write_json(args.response, execute(read_json(args.request)))


if __name__ == "__main__":
    main()
