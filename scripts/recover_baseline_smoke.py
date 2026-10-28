"""Validate a completed native smoke output after launcher cleanup timed out.

The failed result is never modified and this output never counts in the matrix.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from scripts.baseline_geometry import load_geometry, validate_geometry
from scripts.run_flashtalk_baseline import REPO, digest, load_input


def recover(case):
    result_path = case / "result.json"
    record = json.loads(result_path.read_text())
    log = (case / "inference.log").read_text()
    if (
        record["model"] != "liveavatar"
        or record["status"] != "failed"
        or "TimeoutError" not in record.get("error", "")
        or "INFO: Finished." not in log
        or "Merge completed, saved to" not in log
    ):
        raise ValueError(
            "Recovery requires a Live Avatar cleanup timeout after Finished/mux markers"
        )
    profile = load_geometry()
    if record["geometry_profile"] != profile:
        raise ValueError("Geometry profile changed")
    row, seconds = load_input(
        REPO / "data/eval/facetime-zero-shot-v1", record["input_id"]
    )
    if row != record["input"]:
        raise ValueError("Input manifest changed")
    video = case / "video.mp4"
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
    stream = next(item for item in probe["streams"] if item["codec_type"] == "video")
    validate_geometry(stream, profile)
    if not any(item["codec_type"] == "audio" for item in probe["streams"]):
        raise ValueError("Recovered video has no audio")
    if abs(float(stream["duration"]) - seconds) > 0.25:
        raise ValueError("Recovered duration differs from master by over 250ms")
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(video), "-f", "null", "-"],
        check=True,
    )
    recovery = {
        "status": "recovered_functional_output_checks",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": record["model"],
        "input_id": record["input_id"],
        "seed": record["seed"],
        "geometry_profile": profile,
        "original_result_sha256": digest(result_path),
        "original_worker_sha256": record["worker_sha256"],
        "output_video": video.name,
        "output_sha256": digest(video),
        "output_probe": probe,
        "note": "Failed launcher result preserved. Functional geometry/decode/audio gate only; not a successful timed run and never counted among the 108 matrix outputs.",
    }
    output = case / "recovery.json"
    with output.open("x") as file:
        file.write(json.dumps(recovery, indent=2) + "\n")
    return recovery


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    print(json.dumps(recover(parser.parse_args().case.resolve()), indent=2))
