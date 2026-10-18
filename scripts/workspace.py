"""Small, dependency-free workspace helper. No installs or weight downloads."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(["git", *map(str, args)], text=True).strip()


def sources(models, selected, dry_run):
    for name in selected:
        model = models[name]
        dest = ROOT / "vendor" / name
        print(f"{name}: {model['repo']} @ {model['commit']}", flush=True)
        if dest.exists():
            if not (dest / ".git").exists():
                raise RuntimeError(f"Existing non-repository directory: {dest}")
            remote = git("-C", dest, "remote", "get-url", "origin")
            head = git("-C", dest, "rev-parse", "HEAD")
            dirty = git("-C", dest, "status", "--porcelain")
            if remote != model["repo"] or head != model["commit"] or dirty:
                raise RuntimeError(f"{dest}: remote, revision or clean-state mismatch; left untouched")
            print("  Existing checkout verified")
        elif dry_run:
            print(f"  Would fetch into {dest}")
        else:
            dest.mkdir(parents=True)
            subprocess.run(["git", "init", str(dest)], check=True)
            subprocess.run(["git", "-C", str(dest), "remote", "add", "origin", model["repo"]], check=True)
            subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", model["commit"]], check=True)
            subprocess.run(["git", "-C", str(dest), "checkout", "--detach", "FETCH_HEAD"], check=True)


def doctor(models, require_gpu):
    failures = []
    print(f"Python: {sys.version.split()[0]} | workspace: {ROOT}")
    for tool in ("git", "ffmpeg", "ssh"):
        found = shutil.which(tool)
        print(f"{tool}: {found or 'MISSING'}")
        if not found:
            failures.append(tool)
    for name, model in models.items():
        dest = ROOT / "vendor" / name
        valid = (dest / ".git").exists()
        if valid:
            valid = git("-C", dest, "rev-parse", "HEAD") == model["commit"]
        print(f"{name}: {'pinned source present' if valid else 'missing or revision mismatch'}")
        if not valid:
            failures.append(name)
    gpu = shutil.which("nvidia-smi")
    if gpu:
        result = subprocess.run([gpu, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], check=False)
        if result.returncode and require_gpu:
            failures.append("GPU query")
    elif require_gpu:
        failures.append("nvidia-smi")
    else:
        print("GPU: unavailable (optional for workspace checks)")
    if require_gpu:
        # Run inside each model's environment, not the lightweight local Python.
        result = subprocess.run([sys.executable, "-c", "import torch; print('torch:', torch.__version__); assert torch.cuda.is_available(), 'CUDA unavailable'"], check=False)
        if result.returncode:
            failures.append("PyTorch CUDA")
    print("PASS" if not failures else "Missing/failed: " + ", ".join(failures))
    return int(bool(failures))


def main():
    models = json.loads((ROOT / "configs" / "models.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    fetch = commands.add_parser("sources")
    fetch.add_argument("model", choices=["all", *models])
    fetch.add_argument("--dry-run", action="store_true")
    check = commands.add_parser("doctor")
    check.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    if args.command == "list":
        for name, model in models.items():
            print(f"{name}: {model['name']}\n  {model['notes']}")
            for repo in model["weights"]:
                print(f"  https://huggingface.co/{repo}")
        return 0
    if args.command == "sources":
        sources(models, models if args.model == "all" else [args.model], args.dry_run)
        return 0
    return doctor(models, args.require_gpu)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
