"""Sequential 36-output FlashTalk matrix, reusing weights and resetting sessions."""

import argparse
import contextlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.run_flashtalk_baseline import REPO, digest, main as run_one


def frame_hash(video):
    return subprocess.check_output(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-c:v",
            "rawvideo",
            "-f",
            "hash",
            "-hash",
            "sha256",
            "-",
        ],
        text=True,
    ).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets", type=Path, default=REPO / "data/eval/facetime-zero-shot-v1"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.assets, args.output = args.assets.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    suite = json.loads((args.assets / "suite.json").read_text())
    rows = [
        json.loads(line)
        for line in (args.assets / "inputs.jsonl").read_text().splitlines()
    ]
    binding = {
        "input_manifest_sha256": digest(args.assets / "inputs.jsonl"),
        "suite_sha256": digest(args.assets / "suite.json"),
        "worker_sha256": digest(REPO / "scripts/run_flashtalk_baseline.py"),
        "batch_sha256": digest(Path(__file__)),
        "geometry_profile_sha256": digest(REPO / "configs/baseline_geometry.json"),
        "geometry_helper_sha256": digest(REPO / "scripts/baseline_geometry.py"),
        "checkpoints_sha256": digest(REPO / "checkpoints/flashtalk-revisions.json"),
        "environment_freeze": subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        ),
    }
    binding_path = args.output / "binding.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError(
            "Resume rejected: inputs, worker, environment or checkpoints changed"
        )
    binding_path.write_text(json.dumps(binding, indent=2) + "\n")
    jobs = [(row["id"], seed) for seed in suite["seeds"] for row in rows]
    if len(jobs) != 36 or len(set(jobs)) != 36:
        raise ValueError("Expected exactly 18 inputs x 2 seeds")
    # Check session isolation on different identities before the full matrix.
    priorities = [
        ("ref01__s01_greeting", 0),
        ("ref02__s01_greeting", 0),
        ("ref01__s05_pause_resume", 0),
    ]
    jobs = priorities + [job for job in jobs if job not in priorities]
    state, completed = {}, []
    started = time.monotonic()

    def alarm(signum, frame):
        raise TimeoutError("FlashTalk request exceeded its 20-minute limit")

    signal.signal(signal.SIGALRM, alarm)

    def execute(input_id, seed, directory):
        signal.alarm(1200)
        try:
            with (args.output / f"{directory.name}.log").open("w") as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    code = run_one(
                        [
                            "--assets",
                            str(args.assets),
                            "--input-id",
                            input_id,
                            "--seed",
                            str(seed),
                            "--output",
                            str(directory),
                        ],
                        shared_state=state,
                    )
            if code:
                raise RuntimeError(f"FlashTalk failed: {directory}; full suite stopped")
        finally:
            signal.alarm(0)

    for index, (input_id, seed) in enumerate(jobs):
        directory = args.output / f"{input_id}__seed{seed}"
        if directory.exists():
            metadata = json.loads((directory / "metadata.json").read_text())
            if (
                metadata["status"] != "passed_smoke_output_checks"
                or digest(directory / "res_video.mp4") != metadata["output_sha256"]
            ):
                raise ValueError(
                    f"Incomplete/changed output; retain it and use a separate attempt: {directory}"
                )
        else:
            execute(input_id, seed, directory)
        completed.append(directory.name)
        if index == 1 and not (args.output / "session-reset-check.json").exists():
            repeated = args.output / "session-reset-repeat-a"
            execute("ref01__s01_greeting", 0, repeated)
            first = args.output / "ref01__s01_greeting__seed0/res_video.mp4"
            check = {
                "method": "A(ref01)-B(ref02)-A(ref01), seed=0, decoded-video SHA256",
                "first": frame_hash(first),
                "repeated": frame_hash(repeated / "res_video.mp4"),
            }
            check["passed"] = check["first"] == check["repeated"]
            (args.output / "session-reset-check.json").write_text(
                json.dumps(check, indent=2) + "\n"
            )
            if not check["passed"]:
                raise RuntimeError(
                    "Session reset check differed; stopped before full matrix"
                )
        progress = {
            "model": "flashtalk",
            "completed": len(completed),
            "expected": 36,
            "status": "complete" if len(completed) == 36 else "running",
            "elapsed_seconds": time.monotonic() - started,
            "outputs": completed,
        }
        (args.output / "progress.json").write_text(
            json.dumps(progress, indent=2) + "\n"
        )
        print(json.dumps(progress), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
