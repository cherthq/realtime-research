"""Paced/throughput replay of frozen v2 audio through the actual GPU process.

Uses arrived 20ms PCM packets and the same native chunk contract as a call.
Records generation/queue metrics, optionally saves review videos. It does not
measure browser playout, network latency, identity fidelity, or model lip-sync.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import wave

import numpy as np

from .avatarforever import file_hash, load_frozen_case, probe, verify_installation
from .config import ROOT
from .contracts import FRAME_SAMPLES, HEIGHT, SAMPLE_RATE, WIDTH, ResponseBuffer
from .gpu_client import GPUProcess
from .streaming import take_batch


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class ReplayFailure(RuntimeError):
    def __init__(self, cause, metrics):
        super().__init__(str(cause))
        self.cause_type = type(cause).__name__
        self.metrics = metrics


async def replay(gpu, samples, *, mode, response_id, epoch, on_batch=None):
    """Feed incrementally even in throughput mode. Overload is a paced failure."""
    if (
        mode not in {"paced", "throughput"}
        or samples.dtype != np.int16
        or samples.ndim != 1
    ):
        raise ValueError("Invalid replay input")
    if not 0 < len(samples) <= SAMPLE_RATE * 600:
        raise ValueError("Replay duration must be between zero and ten minutes")
    buffer = ResponseBuffer()
    buffer.epoch = epoch - 1
    buffer.begin(response_id)
    started = time.monotonic()
    released = 0
    peak_pending = 0
    events = []
    hashes = set()
    frames, duplicates, valid_samples, compute_seconds = 0, 0, 0, 0.0
    first_decoded = None
    feeder_done = asyncio.Event()

    def report(completed):
        return {
            "completed": completed,
            "mode": mode,
            "duration_seconds": len(samples) / SAMPLE_RATE,
            "released_samples": released,
            "valid_samples": valid_samples,
            "generated_frames": frames,
            "distinct_frame_hashes": len(hashes),
            "exact_duplicate_frames": duplicates,
            "compute_seconds": compute_seconds,
            "generation_fps": frames / compute_seconds if compute_seconds else None,
            "compute_realtime_factor": compute_seconds / (valid_samples / SAMPLE_RATE)
            if valid_samples
            else None,
            "replay_wall_seconds": time.monotonic() - started,
            "first_decoded_seconds": first_decoded,
            "peak_pending_audio_seconds": peak_pending / SAMPLE_RATE,
            "events": events,
            "process_memory": getattr(gpu, "last_memory", None),
            "measurement_scope": "Decoded model chunks only; no browser, audio-device, network, or lip-sync measurement.",
        }

    async def feed():
        nonlocal released, peak_pending
        try:
            for offset in range(0, len(samples), 960):
                if mode == "paced":
                    # A packet becomes available at the end of its capture window.
                    await asyncio.sleep(
                        max(
                            0,
                            started
                            + min(offset + 960, len(samples)) / SAMPLE_RATE
                            - time.monotonic(),
                        )
                    )
                else:
                    # Throughput backpressure is explicit; never call it live pacing.
                    while len(buffer.samples) + 960 > SAMPLE_RATE * 20:
                        await asyncio.sleep(0.001)
                packet = samples[offset : offset + 960]
                buffer.append(response_id, epoch, packet)
                released += len(packet)
                peak_pending = max(peak_pending, len(buffer.samples))
                await asyncio.sleep(0)
            buffer.finish(response_id, epoch)
        finally:
            feeder_done.set()

    feed_task = asyncio.create_task(feed())
    try:
        while buffer.active or not feeder_done.is_set():
            if feed_task.done():
                feed_task.result()
            chunk = take_batch(buffer)
            if chunk is None:
                await asyncio.sleep(0.001)
                continue
            dispatch = time.monotonic() - started
            available_at_dispatch = released
            if chunk.sample_start + chunk.valid_samples > available_at_dispatch:
                raise AssertionError("Unarrived audio reached the renderer")
            batch = await gpu.generate(chunk)
            if batch is None:
                raise RuntimeError("Replay unexpectedly cancelled")
            arrived = time.monotonic() - started
            first_decoded = arrived if first_decoded is None else first_decoded
            if batch.audio.sample_start != valid_samples:
                raise ValueError("Output skipped or repeated PCM samples")
            valid_samples += batch.audio.valid_samples
            frames += len(batch.frames)
            compute_seconds += batch.compute_seconds
            for frame in batch.frames:
                digest = hashlib.sha256(frame).digest()
                duplicates += digest in hashes
                hashes.add(digest)
            events.append(
                {
                    "sequence": len(events),
                    "sample_start": chunk.sample_start,
                    "valid_samples": chunk.valid_samples,
                    "available_samples_at_dispatch": available_at_dispatch,
                    "dispatch_seconds": dispatch,
                    "decoded_seconds": arrived,
                    "compute_seconds": batch.compute_seconds,
                    "frames": len(batch.frames),
                    "process_memory": getattr(gpu, "last_memory", None),
                }
            )
            if on_batch:
                await on_batch(batch)
        await feed_task
        if valid_samples != len(samples) or frames != math.ceil(
            len(samples) / FRAME_SAMPLES
        ):
            raise ValueError("Replay output does not cover the complete master audio")
    except Exception as exc:
        # Preserve overload timing and memory evidence instead of discarding all
        # completed chunks when a long evaluation fails.
        raise ReplayFailure(exc, report(False)) from exc
    finally:
        feed_task.cancel()
        await asyncio.gather(feed_task, return_exceptions=True)
    return report(True)


class VideoWriter:
    def __init__(self, directory, audio):
        self.directory, self.audio = directory, audio
        self.process = None
        self.log = None

    async def start(self):
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise RuntimeError("Install FFmpeg/FFprobe to save review videos")
        self.log = (self.directory / "encode.log").open("wb")
        self.process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            "640x480",
            "-framerate",
            "25",
            "-i",
            "pipe:0",
            "-i",
            str(self.audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(self.directory / "output.mp4"),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=self.log,
        )

    async def append(self, batch):
        self.process.stdin.write(batch.frames.tobytes())
        await asyncio.wait_for(self.process.stdin.drain(), 30)

    async def finish(self, valid_samples):
        self.process.stdin.close()
        result = await asyncio.wait_for(self.process.wait(), 30)
        if result:
            raise RuntimeError("Review video encoding failed")
        self.log.close()
        path = self.directory / "output.mp4"
        metadata = await asyncio.to_thread(probe, path)
        video = next(
            stream for stream in metadata["streams"] if stream["codec_type"] == "video"
        )
        from scripts.baseline_geometry import load_geometry, validate_geometry

        validate_geometry(video, load_geometry())
        if int(video.get("nb_frames", 0)) != math.ceil(valid_samples / FRAME_SAMPLES):
            raise ValueError("Review video lost or repeated valid generated frames")
        duration = float(video.get("duration", metadata["format"]["duration"]))
        if abs(duration - valid_samples / SAMPLE_RATE) > 1 / 25 + 0.02:
            raise ValueError("Review video duration does not match the audio")
        audio = next(
            stream for stream in metadata["streams"] if stream["codec_type"] == "audio"
        )
        if abs(float(audio["duration"]) - valid_samples / SAMPLE_RATE) > 0.025:
            raise ValueError("Review soundtrack does not cover the master audio")
        await asyncio.to_thread(
            subprocess.run,
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                "-i",
                str(path),
                "-f",
                "null",
                "-",
            ],
            check=True,
        )
        return {"path": "output.mp4", "sha256": file_hash(path), "probe": metadata}

    async def close(self):
        if self.process and self.process.returncode is None:
            self.process.kill()
            await self.process.wait()
        if self.log:
            self.log.close()


async def run(args):
    manifest = json.loads((args.pack / "manifest.json").read_text(encoding="utf-8"))
    portraits = (
        sorted(manifest["portraits"]) if args.portrait == "all" else [args.portrait]
    )
    if args.scenario == "all":
        scenarios = [
            case["id"]
            for case in json.loads((ROOT / "configs/zero_shot_suite.json").read_text())[
                "scenarios"
            ]
        ]
    else:
        scenarios = [args.scenario]
    if args.duration_seconds and (len(portraits) != 1 or len(scenarios) != 1):
        raise ValueError(
            "Select one portrait and one script for a continuous-duration test"
        )
    cases = {
        (portrait, scenario): load_frozen_case(args.pack, portrait, scenario)
        for portrait in portraits
        for scenario in scenarios
    }
    # Resolving a venv's Python symlink selects the base interpreter and loses
    # the GPU environment's installed packages. Make it absolute without dereferencing.
    command = [
        str(args.gpu_python.absolute()),
        "-m",
        "realtime.gpu.server",
        "--weights",
        str(args.weights.resolve()),
        "--image",
        str(next(iter(cases.values()))["image"]),
        *(["--compile-blocks"] if args.compile_blocks else []),
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "command": command,
                    "mode": args.mode,
                    "outputs": len(cases) * args.repeat,
                    "duration_seconds": args.duration_seconds,
                    "save_video": args.save_video,
                },
                indent=2,
            )
        )
        return
    installation = verify_installation(args.weights)
    args.output.mkdir(parents=True, exist_ok=False)
    gpu = GPUProcess(command, args.output / "gpu.log", load_timeout=args.load_timeout)
    summary = {
        "status": "running",
        "qualification": "unverified",
        "mode": args.mode,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "installation": installation,
        "adapter_sha256": file_hash(ROOT / "realtime/gpu/avatarforever.py"),
        "input_manifest_sha256": file_hash(args.pack / "manifest.json"),
        "runs": [],
        "browser_latency_ms": None,
        "model_lip_sync": None,
        "seed": 0,
        "geometry": {"width": WIDTH, "height": HEIGHT, "fps": 25},
    }
    epoch = 0
    try:
        await gpu.start()
        summary.update(hardware=gpu.info, cold_load_seconds=gpu.load_seconds)
        for portrait in portraits:
            prepare_started = time.monotonic()
            await gpu.open_session(portrait, cases[(portrait, scenarios[0])]["image"])
            preparation_seconds = time.monotonic() - prepare_started
            for scenario in scenarios:
                case = cases[(portrait, scenario)]
                with wave.open(str(case["audio"]), "rb") as stream:
                    samples = np.frombuffer(
                        stream.readframes(stream.getnframes()), dtype="<i2"
                    ).copy()
                if args.duration_seconds:
                    target = round(args.duration_seconds * SAMPLE_RATE)
                    samples = np.tile(samples, math.ceil(target / len(samples)))[
                        :target
                    ]
                for iteration in range(args.repeat):
                    directory = args.output / f"{portrait}-{scenario}-{iteration:03d}"
                    directory.mkdir()
                    audio_path = case["audio"]
                    if args.duration_seconds:
                        audio_path = directory / "continuous-input.wav"
                        with wave.open(str(audio_path), "wb") as stream:
                            stream.setparams(
                                (
                                    1,
                                    2,
                                    SAMPLE_RATE,
                                    len(samples),
                                    "NONE",
                                    "not compressed",
                                )
                            )
                            stream.writeframes(
                                samples.astype("<i2", copy=False).tobytes()
                            )
                    writer = (
                        VideoWriter(directory, audio_path) if args.save_video else None
                    )
                    metadata = {
                        "status": "running",
                        "portrait": portrait,
                        "scenario": scenario,
                        "iteration": iteration,
                        "voice_selection": case["portrait"]["voice_selection"],
                        "image_sha256": case["portrait"]["sha256"],
                        "source_audio_sha256": case["record"]["sha256"],
                        "effective_audio_sha256": file_hash(audio_path),
                        "warm": len(summary["runs"]) > 0,
                        "continuous_repeated_script": bool(args.duration_seconds),
                        "portrait_preparation_seconds": preparation_seconds,
                    }
                    epoch += 1
                    try:
                        if writer:
                            await writer.start()
                        metadata["metrics"] = await replay(
                            gpu,
                            samples,
                            mode=args.mode,
                            response_id=directory.name,
                            epoch=epoch,
                            on_batch=writer.append if writer else None,
                        )
                        if writer:
                            metadata["video"] = await writer.finish(len(samples))
                        metadata["status"] = "completed"
                    except BaseException as exc:
                        metadata.update(
                            status="failed",
                            failure_type=type(exc).__name__,
                            failure=str(exc),
                        )
                        if isinstance(exc, ReplayFailure):
                            metadata.update(
                                metrics=exc.metrics, failure_type=exc.cause_type
                            )
                        raise
                    finally:
                        if writer:
                            await writer.close()
                        write_json(directory / "run.json", metadata)
                        summary["runs"].append(
                            {
                                "path": str(directory.relative_to(args.output)),
                                "status": metadata["status"],
                                "warm": metadata["warm"],
                                "metrics": metadata.get("metrics"),
                            }
                        )
                        write_json(args.output / "summary.json", summary)
                    print(f"Completed {directory.name}", flush=True)
            await gpu.close_session(portrait)
        warm = [
            run["metrics"]["first_decoded_seconds"]
            for run in summary["runs"]
            if run["status"] == "completed" and run["warm"]
        ]
        summary.update(
            status="completed",
            qualification="streaming_experimental",
            warm_turn_count=len(warm),
            warm_first_decoded_p50_seconds=float(np.percentile(warm, 50))
            if warm
            else None,
            warm_first_decoded_p95_seconds=float(np.percentile(warm, 95))
            if warm
            else None,
        )
    except BaseException as exc:
        summary.update(status="failed", failure_type=type(exc).__name__)
        raise
    finally:
        await gpu.close()
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gpu-python", type=Path, default=ROOT / ".venv-avatarforever/bin/python"
    )
    parser.add_argument(
        "--compile-blocks",
        action="store_true",
        help="Warm regional compilation before replay; recorded separately from inference",
    )
    parser.add_argument(
        "--load-timeout",
        type=float,
        default=300,
        help="Cold load and compile allowance in seconds",
    )
    parser.add_argument("--mode", choices=["paced", "throughput"], default="paced")
    parser.add_argument(
        "--portrait", choices=["ref01", "ref02", "ref03", "all"], default="ref01"
    )
    parser.add_argument("--scenario", default="s01_greeting")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.repeat <= 100 or (
        args.duration_seconds is not None and not 0 < args.duration_seconds <= 600
    ):
        parser.error("Use 1–100 repeats and a duration in (0, 600] seconds")
    if not math.isfinite(args.load_timeout) or not 0 < args.load_timeout <= 1800:
        parser.error("Use a cold-load timeout in (0, 1800] seconds")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
