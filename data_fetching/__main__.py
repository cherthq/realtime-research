"""Run from the repository root: python -m data_fetching --help."""

import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

from .core import DEFAULT_ROOT, load_config, load_sources, write_jsonl
from .media import require_tools
from .pipeline import (MODEL_URLS, build_export, dataset_lock, fetch_sources,
                       import_talkinghead, install_models, process_sources,
                       review_template, summary)


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Dataset directory (default: repo/data/facetime)")
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check local prerequisites; no installs or downloads")
    models = commands.add_parser("models", help="Explicitly download the two detector artifacts")
    models.add_argument("--dry-run", action="store_true")
    imported = commands.add_parser("import-talkinghead", help="Convert upstream *_video_ids.txt; no media download")
    imported.add_argument("--ids-file", type=Path, required=True)
    imported.add_argument("--split", choices=["train", "val", "test"])
    imported.add_argument("--limit", type=positive_int, default=5)
    imported.add_argument("--output", type=Path, required=True)
    for name in ("fetch", "process"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--limit", type=positive_int, default=5, help="Maximum source records, in manifest order")
        command.add_argument("--source-id", help="Select one manifest source ID before applying the limit")
        command.add_argument("--dry-run", action="store_true", help="List work without writing or downloading")
        if name == "process":
            command.add_argument("--config", type=Path)
            command.add_argument("--max-clips", type=positive_int, default=20, help="Maximum candidates attempted per source")
            command.add_argument("--scan-seconds", type=positive_int, default=300,
                                 help="Scan at most this many seconds from each source's start")
    reviews = commands.add_parser("review-template", help="Create a pending-review JSONL file; never overwrite")
    reviews.add_argument("--output", type=Path, required=True)
    export = commands.add_parser("export", help="Validate approvals and write the training JSONL manifest")
    export.add_argument("--reviews", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--dry-run", action="store_true")
    commands.add_parser("report", help="Read candidate counts/rejection reasons; no mutations")
    return cli


def doctor(root):
    failed = False
    print(f"Python {sys.version.split()[0]} | dataset: {root}")
    for name in ("ffmpeg", "ffprobe"):
        path = shutil.which(name)
        print(f"{name}: {path or 'MISSING'}")
        failed |= path is None
    for name in ("numpy", "opencv-contrib-python", "mediapipe", "scenedetect", "yt-dlp", "yt-dlp-ejs"):
        try:
            print(f"{name}: {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{name}: MISSING")
            failed = True
    runtime = shutil.which("deno") or shutil.which("node")
    print(f"YouTube JS runtime (Deno preferred, Node >=22 supported): {runtime or 'MISSING; local-file processing still works'}")
    try:
        from .vision import model_paths
        _, hashes = model_paths(root)
        print("Detector checksums: " + json.dumps(hashes))
    except (OSError, ValueError, KeyError) as exc:
        print(f"Detector models: MISSING/INVALID ({exc}); use the explicit models command later")
        failed = True
    return int(failed)


def write_new(path, rows):
    if path.exists():
        raise ValueError(f"Refusing to overwrite {path}; choose a new output filename")
    write_jsonl(path, rows)


def main(argv=None):
    args = parser().parse_args(argv)
    root = args.root.resolve()
    if args.command == "doctor":
        return doctor(root)
    if args.command == "report":
        print(json.dumps(summary(root), indent=2))
        return 0
    if args.command == "models":
        if args.dry_run:
            print(json.dumps(MODEL_URLS, indent=2))
            return 0
        with dataset_lock(root):
            print(json.dumps(install_models(root), indent=2))
        return 0
    if args.command == "import-talkinghead":
        rows = import_talkinghead(args.ids_file, args.split, args.limit)
        write_new(args.output, rows)
        print(f"Wrote {len(rows)} source records to {args.output}; no media downloaded")
        return 0
    if args.command in {"fetch", "process"}:
        sources = load_sources(args.manifest)
        if args.source_id:
            sources = [source for source in sources if source.id == args.source_id]
            if not sources:
                raise ValueError(f"Source ID not in manifest: {args.source_id}")
        sources = sources[:args.limit]
        config = load_config(args.config) if args.command == "process" else None
        if args.dry_run:
            planned = {"command": args.command, "root": str(root),
                       "sources": [s.to_dict() for s in sources],
                       "note": "No downloads, detector loading, or filesystem writes"}
            if config:
                planned.update(config=asdict(config), max_clips_per_source=args.max_clips,
                               scan_seconds_per_source=args.scan_seconds)
            print(json.dumps(planned, indent=2))
            return 0
        require_tools()
        with dataset_lock(root):
            if args.command == "fetch":
                result = fetch_sources(root, sources)
                return int(any(row["status"] == "failed" for row in result))
            result = process_sources(root, sources, config, max_clips=args.max_clips, scan_seconds=args.scan_seconds)
            return int(any(row["state"] == "failed" for row in result))
    if args.command == "review-template":
        with dataset_lock(root):
            rows = review_template(root)
            write_new(args.output, rows)
        print(f"Wrote {len(rows)} pending reviews; no approvals granted")
        return 0
    if args.command == "export":
        require_tools()
        if args.dry_run:
            rows = build_export(root, args.reviews)
        else:
            with dataset_lock(root):
                rows = build_export(root, args.reviews)
                write_new(args.output, rows)
        print(f"{'Validated' if args.dry_run else 'Exported'} {len(rows)} approved samples")
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
