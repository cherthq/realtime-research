"""Run the remaining 72 file-input outputs after the FlashTalk matrix succeeds."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.run_flashtalk_baseline import REPO, digest
from scripts.summarize_baseline import summarize


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def wait_for_flashtalk(root, deadline, matrix_progress):
    while time.monotonic() < deadline:
        path = root / "flashtalk/progress.json"
        if path.exists():
            try:
                progress = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                # An older producer may be in the middle of a non-atomic write.
                time.sleep(1)
                continue
            if matrix_progress["completed"] != progress["completed"]:
                matrix_progress["completed"] = progress["completed"]
                write(root / "matrix-progress.json", matrix_progress)
            if progress["status"] == "complete" and progress["completed"] == 36:
                return
        alive = (
            subprocess.run(
                ["tmux", "has-session", "-t", "chert-flashtalk-full"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
        if not alive:
            raise RuntimeError(
                "FlashTalk stopped without a complete 36-output matrix; queue stopped"
            )
        time.sleep(10)
    raise TimeoutError("Matrix prerequisite exceeded its wait limit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    progress = {
        "status": "waiting_for_flashtalk",
        "expected": 108,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed": 0,
    }
    try:
        write(root / "matrix-progress.json", progress)
        wait_for_flashtalk(root, time.monotonic() + 4 * 3600, progress)
        idle_deadline = time.monotonic() + 120
        while subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip():
            if time.monotonic() >= idle_deadline:
                raise RuntimeError("GPU did not become idle after FlashTalk completion")
            time.sleep(1)
        if not (args.checkpoints / "READY").exists():
            raise RuntimeError("Local SSD cache is not complete; queue stopped")
        assets = REPO / "data/eval/facetime-zero-shot-v1"
        suite = json.loads((assets / "suite.json").read_text())
        inputs = [
            json.loads(line)
            for line in (assets / "inputs.jsonl").read_text().splitlines()
        ]
        jobs = [(row["id"], seed) for seed in suite["seeds"] for row in inputs]
        priorities = [("ref01__s01_greeting", 0), ("ref01__s05_pause_resume", 0)]
        jobs = priorities + [job for job in jobs if job not in priorities]
        if len(jobs) != 36 or len(set(jobs)) != 36:
            raise ValueError("Expected 36 unique cases per model")
        completed = 36
        for model in ("liveavatar", "liveact"):
            directory = root / model
            directory.mkdir(exist_ok=True)
            python = Path("/home/ubuntu/envs") / model / "bin/python"
            binding = {
                "model": model,
                "inputs_sha256": digest(assets / "inputs.jsonl"),
                "suite_sha256": digest(assets / "suite.json"),
                "worker_sha256": digest(REPO / "scripts/run_native_baseline.py"),
                "geometry_profile_sha256": digest(
                    REPO / "configs/baseline_geometry.json"
                ),
                "geometry_helper_sha256": digest(REPO / "scripts/baseline_geometry.py"),
                "liveavatar_entry_sha256": digest(
                    REPO / "scripts/liveavatar_geometry_entry.py"
                ),
                "liveavatar_cache_sha256": digest(REPO / "scripts/liveavatar_cache.py"),
                "model_registry_sha256": digest(REPO / "configs/models.json"),
                "checkpoints_sha256": digest(
                    args.checkpoints / f"{model}-revisions.json"
                ),
                "environment": subprocess.check_output(
                    [str(python), "-m", "pip", "freeze"], text=True
                ),
            }
            bound = directory / "binding.json"
            if bound.exists() and json.loads(bound.read_text()) != binding:
                raise ValueError(
                    "Resume rejected: environment/checkpoint/config changed"
                )
            write(bound, binding)
            for input_id, seed in jobs:
                job = directory / f"{input_id}__seed{seed}"
                progress.update(
                    status="running",
                    model=model,
                    input_id=input_id,
                    seed=seed,
                    completed=completed,
                )
                write(root / "matrix-progress.json", progress)
                print(json.dumps(progress), flush=True)
                if job.exists():
                    result = json.loads((job / "result.json").read_text())
                    if (
                        result["status"] != "passed_smoke_output_checks"
                        or digest(job / result["output_video"])
                        != result["output_sha256"]
                    ):
                        raise RuntimeError(
                            f"Failed/changed existing output retained; choose a separate attempt: {job}"
                        )
                else:
                    command = [
                        sys.executable,
                        "-u",
                        "-m",
                        "scripts.run_native_baseline",
                        model,
                        "--python",
                        str(python),
                        "--assets",
                        str(assets),
                        "--input-id",
                        input_id,
                        "--seed",
                        str(seed),
                        "--output",
                        str(job),
                        "--checkpoints",
                        str(args.checkpoints),
                    ]
                    with (directory / f"{job.name}.launcher.log").open("w") as log:
                        worker = subprocess.Popen(
                            command,
                            cwd=REPO,
                            stdin=subprocess.DEVNULL,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        try:
                            returncode = worker.wait(timeout=1900)
                        except subprocess.TimeoutExpired:
                            # Let the wrapper record failure and reap its separate
                            # GPU process group before escalating the launcher.
                            os.killpg(worker.pid, signal.SIGTERM)
                            try:
                                worker.wait(timeout=40)
                            except subprocess.TimeoutExpired:
                                os.killpg(worker.pid, signal.SIGKILL)
                                worker.wait(timeout=10)
                            raise
                        if returncode:
                            raise subprocess.CalledProcessError(returncode, command)
                completed += 1
                progress["completed"] = completed
                write(root / "matrix-progress.json", progress)
        progress.update(status="complete", completed=108)
        write(root / "matrix-progress.json", progress)
        return 0
    except Exception as error:
        progress.update(status="stopped_on_failure", error=str(error))
        write(root / "matrix-progress.json", progress)
        raise
    finally:
        try:
            summarize(root, REPO / "data/eval/facetime-zero-shot-v1")
        except Exception as report_error:
            print(f"Result audit failed: {report_error}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
