"""Run one native FlashTalk file-input baseline with in-process chunk metrics.

This is not wall-clock-paced streaming, a warm latency test, or a FaceTime stack.
The pinned upstream implementation is imported without modifying its checkout.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace
import wave


REPO = Path(__file__).resolve().parents[1]
SOURCE_REVISION = "2ba2e1e801d59a9f526212a5aae5d374db8f2978"

sys.path.insert(0, str(REPO))
from scripts.baseline_geometry import (  # noqa: E402
    configure_flashtalk,
    load_geometry,
    validate_geometry,
)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def input_path(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("Input path escaped asset directory")
    return path


def load_input(root, input_id):
    assets = json.loads((root / "assets.json").read_text())
    if digest(root / "inputs.jsonl") != assets["inputs_sha256"]:
        raise ValueError("Frozen input manifest hash mismatch")
    rows = [
        json.loads(line) for line in (root / "inputs.jsonl").read_text().splitlines()
    ]
    matches = [row for row in rows if row["id"] == input_id]
    if len(matches) != 1:
        raise ValueError("Expected exactly one matching input")
    row = matches[0]
    for key in ("reference_image", "audio"):
        if digest(input_path(root, row[key])) != row[f"{key}_sha256"]:
            raise ValueError(f"Changed input {key}")
    with wave.open(str(input_path(root, row["audio"])), "rb") as audio:
        seconds = audio.getnframes() / audio.getframerate()
    return row, seconds


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def collect_memory(stop, records):
    """Driver-visible memory sampled separately from CUDA allocator peaks."""
    while not stop.is_set():
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            )
            for line in output.splitlines():
                pid, memory = [part.strip() for part in line.split(",")]
                if pid == str(os.getpid()):
                    records.append(
                        {
                            "monotonic_seconds": time.monotonic(),
                            "used_mib": float(memory),
                        }
                    )
        except (OSError, ValueError, subprocess.SubprocessError):
            pass  # Unavailable samples are not reported as zero.
        stop.wait(0.5)


def main(argv=None, shared_state=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets", type=Path, default=REPO / "data/eval/facetime-zero-shot-v1"
    )
    parser.add_argument("--input-id", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-offload", action="store_true")
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "status": "started",
        "model": "flashtalk",
        "worker_sha256": digest(Path(__file__)),
        "input_id": args.input_id,
        "seed": args.seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_mode": "preloaded_file_with_chunked_audio_embedding",
        "wall_clock_paced": False,
        "future_audio_isolation_tested": False,
        "warm_latency_available": False,
        "cpu_offload": args.cpu_offload,
        "timing_boundary": "First new decoded chunk available on CPU, before file encoding; cold process/model load included in startup.",
        "unsupported_metrics": [
            "delivered_call_fps",
            "network_latency",
            "barge_in",
            "lip_sync_score",
        ],
    }
    metrics, chunks, memory = {}, [], []
    stop = threading.Event()
    sampler = threading.Thread(target=collect_memory, args=(stop, memory), daemon=True)
    started = time.monotonic()

    def terminate(signum, frame):
        raise TimeoutError(f"Baseline worker received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    try:
        profile = load_geometry()
        metadata["geometry_profile"] = profile
        row, audio_seconds = load_input(args.assets, args.input_id)
        source = REPO / "vendor/flashtalk"
        if git(source, "rev-parse", "HEAD") != SOURCE_REVISION or git(
            source, "status", "--porcelain"
        ):
            raise ValueError("Upstream source must match its clean pinned revision")
        metadata.update(
            input=row,
            audio_seconds=audio_seconds,
            source_revision=SOURCE_REVISION,
            checkpoints=json.loads(
                (REPO / "checkpoints/flashtalk-revisions.json").read_text()
            ),
        )
        with (args.output / "requirements.freeze.txt").open("w") as freeze:
            subprocess.run(
                [sys.executable, "-m", "pip", "freeze"], stdout=freeze, check=True
            )
        metadata["hardware"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        sys.path.insert(0, str(source))
        os.chdir(source)
        import torch

        metadata["torch"] = torch.__version__
        metadata["cuda_runtime"] = torch.version.cuda
        torch.cuda.reset_peak_memory_stats()
        sampler.start()
        if shared_state is not None and "upstream" in shared_state:
            upstream = shared_state["upstream"]
        else:
            spec = importlib.util.spec_from_file_location(
                "chert_native_flashtalk", source / "generate_video.py"
            )
            upstream = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(upstream)
            if shared_state is not None:
                shared_state["upstream"] = upstream
                shared_state["originals"] = {
                    name: getattr(upstream, name)
                    for name in (
                        "get_pipeline",
                        "get_base_data",
                        "get_audio_embedding",
                        "run_pipeline",
                        "save_video",
                    )
                }
        import flash_talk.inference as inference

        configure_flashtalk(inference, profile)
        metadata["native_inference_parameters"] = dict(upstream.infer_params)
        # The batch driver redirects Python stderr per request. Rebind loguru so
        # its first request's now-closed stream is not retained by later jobs.
        from loguru import logger

        logger.remove()
        logger.add(sys.stderr)
        metadata["native_audio_preprocessing"] = (
            "librosa.load(mono=True,sr=16000), chunk-local upstream loudness_norm(-23 LUFS); master WAV unchanged"
        )
        metadata["generation_timing_note"] = (
            "Generation loop includes audio-file loading, chunk embeddings, model/VAE execution, CPU transfers and cold compilation. File encoding/mux is separate."
        )
        metadata["geometry_note"] = (
            "Direct model generation at the recorded 4:3 dimensions. Reference is resized at matching aspect ratio; no output stretching, padding or portrait cropping. Master image remains unchanged."
        )
        originals = (
            shared_state["originals"]
            if shared_state is not None
            else {
                name: getattr(upstream, name)
                for name in (
                    "get_pipeline",
                    "get_base_data",
                    "get_audio_embedding",
                    "run_pipeline",
                    "save_video",
                )
            }
        )
        original_pipeline = originals["get_pipeline"]
        original_prepare = originals["get_base_data"]
        original_audio = originals["get_audio_embedding"]
        original_generate = originals["run_pipeline"]
        original_save = originals["save_video"]
        reused = shared_state is not None and "pipeline" in shared_state
        metadata["model_reused"] = reused
        metadata["warm_latency_available"] = reused
        markers = {}

        def pipeline(*positional, **keywords):
            mark = time.monotonic()
            if reused:
                value = shared_state["pipeline"]
            else:
                value = original_pipeline(*positional, **keywords)
                if shared_state is not None:
                    shared_state["pipeline"] = value
            torch.cuda.synchronize()
            metrics["model_load_seconds"] = time.monotonic() - mark
            return value

        def prepare(*positional, **keywords):
            mark = time.monotonic()
            value = original_prepare(*positional, **keywords)
            torch.cuda.synchronize()
            metrics["reference_prompt_preprocessing_seconds"] = time.monotonic() - mark
            markers["generation_start"] = time.monotonic()
            return value

        def audio(*positional, **keywords):
            mark = time.monotonic()
            value = original_audio(*positional, **keywords)
            torch.cuda.synchronize()
            metrics["audio_embedding_seconds"] = (
                metrics.get("audio_embedding_seconds", 0) + time.monotonic() - mark
            )
            return value

        def generate(*positional, **keywords):
            mark = time.monotonic()
            value = original_generate(*positional, **keywords)
            # Preserve the upstream tensor values and overlap removal. Transfer
            # here rather than at its later .cpu() to timestamp usable output.
            value = value.cpu()
            available = time.monotonic()
            new_frames = len(value) - upstream.infer_params["motion_frames_num"]
            if new_frames <= 0:
                raise ValueError("No new decoded frames")
            chunks.append(
                {
                    "index": len(chunks),
                    "new_frames": new_frames,
                    "generation_and_cpu_transfer_seconds": available - mark,
                    "available_since_process_start_seconds": available - started,
                }
            )
            markers["last_chunk"] = available
            print(json.dumps({"event": "decoded_chunk", **chunks[-1]}), flush=True)
            return value

        def save(*positional, **keywords):
            mark = time.monotonic()
            value = original_save(*positional, **keywords)
            metrics["file_encoding_mux_seconds"] = time.monotonic() - mark
            return value

        upstream.get_pipeline, upstream.get_base_data = pipeline, prepare
        upstream.get_audio_embedding, upstream.run_pipeline, upstream.save_video = (
            audio,
            generate,
            save,
        )
        video = args.output / "res_video.mp4"
        parameters = SimpleNamespace(
            ckpt_dir=str(REPO / "checkpoints/SoulX-FlashTalk-14B"),
            wav2vec_dir=str(REPO / "checkpoints/chinese-wav2vec2-base"),
            cpu_offload=args.cpu_offload,
            base_seed=args.seed,
            input_prompt=row["prompt"],
            cond_image=str(input_path(args.assets, row["reference_image"])),
            audio_path=str(input_path(args.assets, row["audio"])),
            audio_encode_mode="stream",
            save_file=str(video),
        )
        upstream._validate_args(parameters)
        upstream.generate(parameters)
        generation_seconds = markers["last_chunk"] - markers["generation_start"]
        frames = sum(chunk["new_frames"] for chunk in chunks)
        generated_seconds = frames / upstream.infer_params["tgt_fps"]
        metrics.update(
            cold_first_new_decoded_chunk_cpu_seconds=chunks[0][
                "available_since_process_start_seconds"
            ],
            generation_seconds=generation_seconds,
            generated_new_frames=frames,
            generated_new_video_seconds=generated_seconds,
            generation_fps=frames / generation_seconds,
            realtime_factor=generation_seconds / generated_seconds,
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        if reused:
            metrics["warm_request_first_new_decoded_chunk_cpu_seconds"] = metrics.pop(
                "cold_first_new_decoded_chunk_cpu_seconds"
            )
            metadata["timing_boundary"] = (
                "Warm request start to first new decoded chunk on CPU; includes input checks/preparation, excludes prior model load. File-input test, not delivered call latency."
            )
        useful_frames = min(
            frames, math.ceil(audio_seconds * upstream.infer_params["tgt_fps"])
        )
        metrics.update(
            generated_padding_frames=max(0, frames - useful_frames),
            useful_generation_fps=useful_frames / generation_seconds,
            useful_realtime_factor=generation_seconds / audio_seconds,
        )
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_streams",
                    "-show_format",
                    "-of",
                    "json",
                    str(video),
                ],
                text=True,
            )
        )
        video_stream = next(
            stream for stream in probe["streams"] if stream["codec_type"] == "video"
        )
        validate_geometry(video_stream, profile)
        if not any(stream["codec_type"] == "audio" for stream in probe["streams"]):
            raise ValueError("Output missing audio")
        metadata["output_probe"] = probe
        metadata["output_sha256"] = digest(video)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"], check=True
        )
        metrics["muxed_video_seconds"] = float(video_stream["duration"])
        metrics["audio_video_duration_difference_seconds"] = (
            float(video_stream["duration"]) - audio_seconds
        )
        if abs(metrics["audio_video_duration_difference_seconds"]) > 0.25:
            raise ValueError("Output duration differs from input by more than 250ms")
        metadata["status"] = "passed_smoke_output_checks"
    except Exception:
        metadata["status"] = "failed"
        metadata["error"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        stop.set()
        if sampler.is_alive():
            sampler.join(timeout=6)
        metrics["total_process_seconds"] = time.monotonic() - started
        if sys.platform == "linux":
            import resource

            metrics["host_peak_rss_bytes"] = (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            )
        else:
            metrics["host_peak_rss_bytes"] = None
        metrics["sampled_process_gpu_peak_mib"] = max(
            (item["used_mib"] for item in memory), default=None
        )
        metrics["gpu_memory_sample_count"] = len(memory)
        metadata["memory_sampling_interval_seconds"] = 0.5
        for filename, content in (
            ("metadata.json", metadata),
            ("metrics.json", metrics),
            ("chunks.json", chunks),
            ("gpu-memory.json", memory),
        ):
            (args.output / filename).write_text(
                json.dumps(content, indent=2) + "\n", encoding="utf-8"
            )
        print(
            json.dumps({"status": metadata["status"], "metrics": metrics}), flush=True
        )
    return 0 if metadata["status"] == "passed_smoke_output_checks" else 1


if __name__ == "__main__":
    raise SystemExit(main())
