"""Gate the paid 108-output matrix on three direct-4:3 generation smoke tests."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from scripts.baseline_geometry import load_geometry, validate_geometry
from scripts.run_flashtalk_baseline import REPO, digest
from scripts.run_native_baseline import stop_child


def run(command, log, timeout):
    with log.open("w") as stream:
        child = subprocess.Popen(
            command,
            cwd=REPO,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "8",
                "TORCHINDUCTOR_COMPILE_THREADS": "8",
            },
        )
        try:
            try:
                code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Native wrappers reap their separate GPU groups on SIGTERM.
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    stop_child(child)
                raise
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            stop_child(child)


def check_smoke(case, model, profile):
    record = json.loads(
        (
            case / ("metadata.json" if model == "flashtalk" else "result.json")
        ).read_text()
    )
    if model == "liveavatar" and record["status"] == "failed":
        recovery = json.loads((case / "recovery.json").read_text())
        if recovery["status"] != "recovered_functional_output_checks" or recovery[
            "original_result_sha256"
        ] != digest(case / "result.json"):
            raise ValueError("Recovery is not bound to the preserved failed result")
        record = recovery
    if (
        record["status"]
        not in ("passed_smoke_output_checks", "recovered_functional_output_checks")
        or record["geometry_profile"] != profile
    ):
        raise ValueError(f"Unsuccessful or changed smoke test: {case}")
    stream = next(
        item
        for item in record["output_probe"]["streams"]
        if item["codec_type"] == "video"
    )
    validate_geometry(stream, profile)
    filename = "res_video.mp4" if model == "flashtalk" else record["output_video"]
    if digest(case / filename) != record["output_sha256"]:
        raise ValueError(f"Smoke output changed: {case}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    profile = load_geometry()
    bindings = {
        str(path.relative_to(REPO)): digest(path)
        for path in [
            REPO / "configs/baseline_geometry.json",
            REPO / "configs/models.json",
            *(
                REPO / "scripts" / name
                for name in (
                    "run_4x3_baselines.py",
                    "run_native_baseline.py",
                    "run_flashtalk_baseline.py",
                    "run_flashtalk_suite.py",
                    "run_remaining_matrix.py",
                    "baseline_geometry.py",
                    "liveavatar_geometry_entry.py",
                    "liveavatar_cache.py",
                )
            ),
        ]
    }
    binding = root / "binding.json"
    if binding.exists() and json.loads(binding.read_text()) != bindings:
        raise ValueError(
            "Run source/profile changed; preserve it and choose a new attempt"
        )
    binding.write_text(json.dumps(bindings, indent=2) + "\n")
    status = {
        "status": "smoke_tests",
        "geometry": profile,
        "passed_smokes": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    def save():
        temporary = root / "controller-progress.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(root / "controller-progress.json")

    try:
        for model in ("flashtalk", "liveavatar", "liveact"):
            case = root / f"smoke-{model}"
            status["model"] = model
            save()
            if not case.exists():
                python = f"/home/ubuntu/envs/{model}/bin/python"
                command = (
                    [python, "-u", "-m", "scripts.run_flashtalk_baseline"]
                    if model == "flashtalk"
                    else [
                        sys.executable,
                        "-u",
                        "-m",
                        "scripts.run_native_baseline",
                        model,
                        "--python",
                        python,
                        "--checkpoints",
                        str(args.checkpoints),
                    ]
                )
                command += [
                    "--input-id",
                    "ref01__s01_greeting",
                    "--seed",
                    "0",
                    "--output",
                    str(case),
                ]
                run(command, root / f"smoke-{model}.launcher.log", 1900)
            check_smoke(case, model, profile)
            status["passed_smokes"].append(model)
            save()
        status.update(status="full_matrix", model="flashtalk")
        save()
        matrix = root / "matrix"
        matrix.mkdir(exist_ok=True)
        run(
            [
                "/home/ubuntu/envs/flashtalk/bin/python",
                "-u",
                "-m",
                "scripts.run_flashtalk_suite",
                "--output",
                str(matrix / "flashtalk"),
            ],
            root / "flashtalk-matrix.log",
            4 * 3600,
        )
        status["model"] = "liveavatar_then_liveact"
        save()
        run(
            [
                sys.executable,
                "-u",
                "-m",
                "scripts.run_remaining_matrix",
                "--output",
                str(matrix),
                "--checkpoints",
                str(args.checkpoints),
            ],
            root / "remaining-matrix.log",
            26 * 3600,
        )
        status["status"] = "complete"
    except Exception as error:
        status.update(status="stopped_on_failure", error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
