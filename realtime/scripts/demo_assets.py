"""Check, export or import the nine media inputs needed by the demo.

Only configured portraits and replay WAVs are exported. Runtime state, voice
assignments, credentials, recordings and model weights are never included.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import wave
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS = ROOT / "data/eval/facetime-zero-shot-v1"
MANIFEST = "demo-assets.json"
MAX_FILE_BYTES = 64 * 1024 * 1024


def config():
    portraits = json.loads(
        (ROOT / "configs/zero_shot_portraits.json").read_text(encoding="utf-8")
    )
    suite = json.loads(
        (ROOT / "configs/zero_shot_suite.json").read_text(encoding="utf-8")
    )
    names = [p["path"] for p in portraits["portraits"]]
    names += [f"audio/{s['id']}.wav" for s in suite["scenarios"]]
    for name in names:
        if Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name:
            raise ValueError("Invalid configured asset path")
    return sorted(names), {"portraits": portraits, "suite": suite}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(assets: Path) -> dict:
    assets = assets.resolve()
    names, provenance = config()
    records = {}
    for name in names:
        path = assets / name
        if not path.resolve().is_relative_to(assets) or path.is_symlink():
            raise ValueError(f"Asset must be a regular file inside the pack: {name}")
        if not path.is_file():
            raise ValueError(f"Missing asset: {name}; import the demo asset pack first")
        size = path.stat().st_size
        if not 0 < size <= MAX_FILE_BYTES:
            raise ValueError(f"Unexpected asset size: {name}")
        if path.suffix == ".png":
            with Image.open(path) as image:
                if image.format != "PNG" or image.width * 3 != image.height * 4:
                    raise ValueError(f"Expected a 4:3 PNG: {name}")
                image.verify()
        else:
            with wave.open(str(path), "rb") as audio:
                if (
                    audio.getnchannels(),
                    audio.getframerate(),
                    audio.getsampwidth(),
                    audio.getcomptype(),
                ) != (1, 48000, 2, "NONE"):
                    raise ValueError(f"Expected mono 48kHz PCM16 WAV: {name}")
                frames = audio.getnframes()
                if (
                    not 0 < frames <= 20 * 48000
                    or len(audio.readframes(frames)) != frames * 2
                ):
                    raise ValueError(
                        f"Invalid/truncated replay audio or duration above 20s: {name}"
                    )
        records[name] = {"sha256": digest(path), "bytes": size}
    result = {
        "schema": "chert-demo-assets-v1",
        "files": records,
        "provenance": provenance,
    }
    manifest = assets / MANIFEST
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != result:
        raise ValueError(
            "Asset manifest does not match media or this checkout's configuration"
        )
    return result


def export_assets(assets: Path, output: Path) -> dict:
    record = check(assets)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves an existing frozen handoff pack.
    with output.open("xb") as target:
        try:
            with zipfile.ZipFile(
                target, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                for name in record["files"]:
                    archive.write(assets / name, name)
                archive.writestr(MANIFEST, json.dumps(record, indent=2) + "\n")
        except Exception:
            target.close()
            output.unlink()
            raise
    return {
        "archive": str(output),
        "sha256": digest(output),
        "media_files": len(record["files"]),
    }


def import_assets(archive_path: Path, assets: Path) -> dict:
    assets = assets.absolute()
    if assets.exists() or assets.is_symlink():
        raise FileExistsError("Destination already exists; use a new asset directory")
    names, _ = config()
    allowed = {*names, MANIFEST}
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        if len(entries) != len(allowed) or {x.filename for x in entries} != allowed:
            raise ValueError(
                "Archive must contain exactly the configured media and demo-assets.json"
            )
        if any(x.is_dir() or not 0 < x.file_size <= MAX_FILE_BYTES for x in entries):
            raise ValueError("Invalid or oversized archive entry")
        assets.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".demo-assets-", dir=assets.parent
        ) as staging:
            staged = Path(staging) / "pack"
            for entry in entries:
                target = staged / entry.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                # Copy allowlisted content; do not restore symlinks or permissions.
                with archive.open(entry) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
            record = check(staged)
            if assets.exists() or assets.is_symlink():
                raise FileExistsError("Destination was created during import")
            staged.rename(assets)
    return {
        "assets": str(assets),
        "media_files": len(record["files"]),
        "verified": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "export", "import"):
        command = commands.add_parser(name)
        command.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
        if name == "export":
            command.add_argument("--output", type=Path, required=True)
        elif name == "import":
            command.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "export":
            result = export_assets(args.assets, args.output)
        elif args.command == "import":
            result = import_assets(args.archive, args.assets)
        else:
            result = {"verified": True, "media_files": len(check(args.assets)["files"])}
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, zipfile.BadZipFile, wave.Error) as exc:
        parser.exit(1, f"Asset check failed: {exc}\n")


if __name__ == "__main__":
    main()
