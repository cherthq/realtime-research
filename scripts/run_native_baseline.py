"""Bounded file-input smoke runner for Live Avatar and LiveAct.

Reports full-process runtime only. No first-frame or model-only FPS claims are
made without in-process hooks. The frozen masters and upstream sources survive.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import traceback

from scripts.run_flashtalk_baseline import digest, git, input_path, load_input
from scripts.baseline_geometry import load_geometry, validate_geometry


REPO = Path(__file__).resolve().parents[1]


def build_command(
    model, python, source, checkpoints, assets, row, output, seed, seconds
):
    image = str(input_path(assets, row["reference_image"]))
    audio = str(input_path(assets, row["audio"]))
    profile = load_geometry()
    size = f"{profile['width']}*{profile['height']}"
    if model == "liveavatar":
        return [
            str(python),
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=1",
            str(REPO / "scripts/liveavatar_geometry_entry.py"),
            "--ulysses_size",
            "1",
            "--task",
            "s2v-14B",
            "--size",
            size,
            "--base_seed",
            str(seed),
            "--training_config",
            "liveavatar/configs/s2v_causal_sft.yaml",
            "--offload_model",
            "True",
            "--t5_cpu",
            "--convert_model_dtype",
            "--prompt",
            row["prompt"],
            "--image",
            image,
            "--audio",
            audio,
            "--infer_frames",
            "48",
            "--load_lora",
            "--lora_path_dmd",
            str(checkpoints / "Live-Avatar/liveavatar.safetensors"),
            "--sample_steps",
            "4",
            "--sample_guide_scale",
            "0",
            "--num_clip",
            str(math.ceil(seconds * 25 / 48) + 1),
            "--num_gpus_dit",
            "1",
            "--sample_solver",
            "euler",
            "--single_gpu",
            "--offload_kv_cache",
            "--ckpt_dir",
            str(checkpoints / "Wan2.2-S2V-14B"),
            "--save_file",
            str(output / "video.mp4"),
        ]
    if model == "liveact":
        return [
            str(python),
            str(source / "generate.py"),
            "--size",
            size,
            "--ckpt_dir",
            str(checkpoints / "LiveAct"),
            "--wav2vec_dir",
            str(checkpoints / "chinese-wav2vec2-base"),
            "--fps",
            "24",
            "--input_json",
            str(output / "input.json"),
            "--seed",
            str(seed),
            "--t5_cpu",
            "--fp8_kv_cache",
            "--dura_print",
        ]
    raise ValueError(f"Unsupported model {model}")


def sample_device_memory(stop, samples):
    while not stop.is_set():
        try:
            value = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            ).strip()
            samples.append(
                {"monotonic_seconds": time.monotonic(), "device_used_mib": float(value)}
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        stop.wait(0.5)


def stop_child(child):
    """Reap the isolated native process group, including torchrun descendants."""
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait(timeout=10)
        return
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=10)


def interrupted(signum, frame):
    raise RuntimeError(f"Native worker interrupted by signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["liveavatar", "liveact"])
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument(
        "--assets", type=Path, default=REPO / "data/eval/facetime-zero-shot-v1"
    )
    parser.add_argument("--input-id", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--checkpoints", type=Path, default=REPO / "checkpoints")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "started",
        "model": args.model,
        "input_id": args.input_id,
        "seed": args.seed,
        "timeout_seconds": args.timeout,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_mode": "preloaded_file",
        "first_frame_latency_seconds": None,
        "model_only_generation_fps": None,
        "lip_sync_score": None,
        "note": "File-input functional smoke only; process time includes loading, compilation, generation and saving. Device memory includes driver/context, not just tensors.",
    }
    stop, samples = threading.Event(), []
    sampler = threading.Thread(
        target=sample_device_memory, args=(stop, samples), daemon=True
    )
    child, started = None, time.monotonic()
    previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
    try:
        profile = load_geometry()
        record["geometry_profile"] = profile
        if args.model == "liveavatar":
            record["geometry_entry_sha256"] = digest(
                REPO / "scripts/liveavatar_geometry_entry.py"
            )
            record["cache_adapter_sha256"] = digest(
                REPO / "scripts/liveavatar_cache.py"
            )
            record["conditioning_cache_policy"] = (
                "Grow conditioning K/V capacity from actual prefill token count; retain values, dtype/device and original attention computation. No token truncation."
            )
        source = REPO / "vendor" / args.model
        model = json.loads((REPO / "configs/models.json").read_text())[args.model]
        if git(source, "rev-parse", "HEAD") != model["commit"] or git(
            source, "status", "--porcelain"
        ):
            raise ValueError("Upstream source must match its clean pinned revision")
        active = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip()
        if active:
            raise RuntimeError("GPU is busy; run each baseline sequentially")
        row, seconds = load_input(args.assets, args.input_id)
        checkpoints = args.checkpoints.resolve()
        record.update(
            input=row,
            audio_seconds=seconds,
            source_revision=model["commit"],
            checkpoints=json.loads(
                (checkpoints / f"{args.model}-revisions.json").read_text()
            ),
        )
        record["worker_sha256"] = digest(Path(__file__))
        record["checkpoint_root"] = str(checkpoints)
        if args.model == "liveact":
            (args.output / "input.json").write_text(
                json.dumps(
                    [
                        {
                            "prompt": row["prompt"],
                            "cond_image": str(
                                input_path(args.assets, row["reference_image"])
                            ),
                            "cond_audio": str(input_path(args.assets, row["audio"])),
                        }
                    ],
                    indent=2,
                )
            )
            record["precision_note"] = (
                "BF16 weights with FP8 KV cache; native single-GPU memory-saving configuration, may alter quality. No FP8 GEMM or block offload."
            )
            record["native_audio_preprocessing"] = (
                "Upstream SoX tempo(25/24), 16kHz resampling and gain 3.0; master WAV unchanged."
            )
        else:
            record["precision_note"] = (
                "BF16 conversion, CPU T5/model offload and CPU KV-cache offload for the 80GB single-GPU budget; no --fp8 weight quantization. Cache transfers affect speed."
            )
        command = build_command(
            args.model,
            args.python,
            source,
            checkpoints,
            args.assets,
            row,
            args.output,
            args.seed,
            seconds,
        )
        record["command"] = command
        with (args.output / "requirements.freeze.txt").open("w") as freeze:
            subprocess.run(
                [str(args.python), "-m", "pip", "freeze"], stdout=freeze, check=True
            )
        record["hardware"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "8",
            "TORCHINDUCTOR_COMPILE_THREADS": "8",
            "ENABLE_COMPILE": "true",
            "USE_CHANNELS_LAST_3D": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        sampler.start()
        with (args.output / "inference.log").open("w") as log:
            child = subprocess.Popen(
                command,
                cwd=source if args.model == "liveavatar" else args.output,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            process_start = time.monotonic()
            try:
                returncode = child.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                stop_child(child)
                raise TimeoutError(f"Model exceeded {args.timeout} second smoke limit")
        record["process_seconds"] = time.monotonic() - process_start
        record["returncode"] = returncode
        if returncode:
            raise RuntimeError(
                f"Native inference exited {returncode}; see inference.log"
            )
        video = (
            args.output / "video.mp4"
            if args.model == "liveavatar"
            else args.output
            / f"{Path(row['reference_image']).stem}_{Path(row['audio']).stem}.mp4"
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
        stream = next(
            item for item in probe["streams"] if item["codec_type"] == "video"
        )
        validate_geometry(stream, profile)
        if not any(item["codec_type"] == "audio" for item in probe["streams"]):
            raise ValueError("Output has no audio")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"], check=True
        )
        record.update(
            output_video=video.name,
            output_sha256=digest(video),
            output_probe=probe,
            audio_video_duration_difference_seconds=float(stream["duration"]) - seconds,
        )
        if abs(record["audio_video_duration_difference_seconds"]) > 0.25:
            raise ValueError("Output duration differs from master by over 250ms")
        record["status"] = "passed_smoke_output_checks"
    except Exception:
        record.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        stop_child(child)
        signal.signal(signal.SIGTERM, previous_sigterm)
        stop.set()
        if sampler.is_alive():
            sampler.join(timeout=6)
        record.update(
            total_seconds=time.monotonic() - started,
            peak_device_used_mib=max(
                (item["device_used_mib"] for item in samples), default=None
            ),
            memory_sample_count=len(samples),
            memory_sampling_interval_seconds=0.5,
        )
        (args.output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        (args.output / "gpu-memory.json").write_text(
            json.dumps(samples, indent=2) + "\n"
        )
        print(
            json.dumps(
                {
                    key: record.get(key)
                    for key in (
                        "model",
                        "status",
                        "process_seconds",
                        "peak_device_used_mib",
                        "error",
                    )
                }
            ),
            flush=True,
        )
    return 0 if record["status"] == "passed_smoke_output_checks" else 1


if __name__ == "__main__":
    raise SystemExit(main())
