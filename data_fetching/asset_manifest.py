"""Print local checkpoint SHA-256 inventory; no writes or network access."""

import argparse
import json
from pathlib import Path

from .core import file_hash


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("path", type=Path)
    args = cli.parse_args()
    path = args.path.resolve()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError("No model files found")
    print(json.dumps([{"path": str(p), "sha256": file_hash(p)} for p in files], indent=2))


if __name__ == "__main__":
    main()
