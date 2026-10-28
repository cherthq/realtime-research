"""Audit the expected native matrix without inventing missing quality metrics."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.baseline_geometry import validate_geometry
from scripts.run_flashtalk_baseline import REPO, digest


def read_record(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def collect(root, assets):
    inputs = [
        json.loads(line)
        for line in (assets / "inputs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seeds = read_record(assets / "suite.json")["seeds"]
    rows = []
    for model in ("flashtalk", "liveavatar", "liveact"):
        for seed in seeds:
            for item in inputs:
                case = root / model / f"{item['id']}__seed{seed}"
                row = {
                    "model": model,
                    "input_id": item["id"],
                    "seed": seed,
                    "status": "incomplete" if case.exists() else "not_started",
                    "geometry_profile": None,
                    "worker_wall_seconds": None,
                    "timing_condition": "warm_or_cold_unknown",
                    "generation_fps": None,
                    "useful_generation_fps": None,
                    "first_cpu_chunk_seconds": None,
                    "gpu_peak_mib": None,
                    "gpu_memory_boundary": None,
                    "width": None,
                    "height": None,
                    "output_video_seconds": None,
                    "lip_sync_score": None,
                    "output_video": None,
                    "error": None,
                }
                try:
                    record = read_record(
                        case
                        / ("metadata.json" if model == "flashtalk" else "result.json")
                    )
                    if record:
                        row["status"] = record["status"]
                        profile = record.get("geometry_profile")
                        row["geometry_profile"] = (
                            profile.get("profile_id") if profile else None
                        )
                        row["error"] = record.get("error")
                        if model == "flashtalk":
                            metrics = read_record(case / "metrics.json")
                            row.update(
                                worker_wall_seconds=metrics.get(
                                    "total_process_seconds"
                                ),
                                timing_condition="warm"
                                if record.get("model_reused")
                                else "cold",
                                generation_fps=metrics.get("generation_fps"),
                                useful_generation_fps=metrics.get(
                                    "useful_generation_fps"
                                ),
                                first_cpu_chunk_seconds=metrics.get(
                                    "warm_request_first_new_decoded_chunk_cpu_seconds",
                                    metrics.get(
                                        "cold_first_new_decoded_chunk_cpu_seconds"
                                    ),
                                ),
                                gpu_peak_mib=metrics.get(
                                    "sampled_process_gpu_peak_mib"
                                ),
                                gpu_memory_boundary="sampled_process",
                            )
                            filename = "res_video.mp4"
                        else:
                            row.update(
                                worker_wall_seconds=record.get("total_seconds"),
                                timing_condition="cold",
                                gpu_peak_mib=record.get("peak_device_used_mib"),
                                gpu_memory_boundary="sampled_device",
                            )
                            filename = record.get("output_video")
                        streams = record.get("output_probe", {}).get("streams", [])
                        video = next(
                            (
                                stream
                                for stream in streams
                                if stream["codec_type"] == "video"
                            ),
                            {},
                        )
                        row.update(
                            width=video.get("width"),
                            height=video.get("height"),
                            output_video_seconds=video.get("duration"),
                        )
                        if record["status"] == "passed_smoke_output_checks":
                            if profile:
                                validate_geometry(video, profile)
                            if not filename or Path(filename).name != filename:
                                raise ValueError("Missing or unsafe output filename")
                            output = case / filename
                            if (
                                not output.is_file()
                                or digest(output) != record["output_sha256"]
                            ):
                                raise ValueError("Output missing or SHA256 changed")
                            row["status"] = "validated_output"
                            row["output_video"] = str(output.relative_to(root))
                except (OSError, ValueError, KeyError, TypeError) as error:
                    row.update(status="audit_failed", error=str(error))
                rows.append(row)
    return rows


def summarize(root, assets):
    rows = collect(root, assets)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expected": len(rows),
        "counts": dict(Counter(row["status"] for row in rows)),
        "by_model": {
            model: dict(Counter(row["status"] for row in rows if row["model"] == model))
            for model in ("flashtalk", "liveavatar", "liveact")
        },
        "complete": bool(rows)
        and all(row["status"] == "validated_output" for row in rows),
        "limitations": [
            "Validated means decodable output and recorded hash, not good avatar quality.",
            "Only FlashTalk has generation-loop and CPU-chunk instrumentation.",
            "Cold process time and warm generation-loop time are different boundaries.",
            "No live transport, lip-sync or perceptual quality score is established.",
        ],
    }
    for filename, value in (("summary.json", summary),):
        temporary = root / f"{filename}.tmp"
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(root / filename)
    temporary = root / "outputs.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(root / "outputs.csv")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--assets", type=Path, default=REPO / "data/eval/facetime-zero-shot-v1"
    )
    args = parser.parse_args()
    print(json.dumps(summarize(args.output.resolve(), args.assets.resolve()), indent=2))


if __name__ == "__main__":
    main()
