"""Check the tracked and non-ignored working files before publishing.

Reports paths and line numbers only, never the matching credential text.
This catches common mistakes; it is not a replacement for reviewing a diff.
"""

import argparse
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DIRS = {".cache", "data", "checkpoints", "runs", "vendor", "node_modules"}
PRIVATE_SUFFIXES = {
    ".pem",
    ".key",
    ".safetensors",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".mp4",
    ".mkv",
    ".mov",
    ".webm",
    ".wav",
    ".flac",
    ".zip",
    ".gz",
}
PATTERNS = {
    "OpenAI credential": re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}"),
    "Hugging Face credential": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private key": re.compile(
        r"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----"
    ),
}


def check(root: Path) -> list[str]:
    listing = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
        )
        .decode("utf-8")
        .split("\0")
    )
    failures = []
    for name in sorted(set(filter(None, listing))):
        path = root / name
        relative = Path(name)
        if path.is_symlink():
            failures.append(f"{name}: review symlink before publishing")
            continue
        if not path.exists():
            continue  # A tracked deletion is not part of the eventual tree.
        private = (
            any(
                part in PRIVATE_DIRS or part.startswith(".venv")
                for part in relative.parts
            )
            or relative.suffix.lower() in PRIVATE_SUFFIXES
            or (relative.name.startswith(".env") and relative.name != ".env.example")
        )
        if private:
            failures.append(f"{name}: generated/private file would be published")
            continue
        data = path.read_bytes()
        if len(data) > 5 * 1024 * 1024:
            failures.append(f"{name}: unexpectedly large source file")
        if b"\0" in data:
            continue
        for line_number, line in enumerate(
            data.decode("utf-8", errors="replace").splitlines(), 1
        ):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    failures.append(f"{name}:{line_number}: possible {label}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    failures = check(parser.parse_args().root)
    if failures:
        print("\n".join(failures))
        raise SystemExit(1)
    print("Public-tree check passed: no flagged files or credential patterns.")


if __name__ == "__main__":
    main()
