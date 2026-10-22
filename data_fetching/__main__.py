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
from .pipeline import (
    MODEL_URLS,
    build_export,
    dataset_lock,
    fetch_sources,
    import_talkinghead,
    install_models,
    process_sources,
    review_template,
    summary,
)


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Dataset directory (default: repo/data/facetime)",
    )
    commands = cli.add_subparsers(dest="command", required=True)
    diagnostics = commands.add_parser(
        "doctor", help="Check local prerequisites; no installs or downloads"
    )
    diagnostics.add_argument(
        "--profile", choices=["smoke-test", "quality-long"], default="smoke-test"
    )
    diagnostics.add_argument("--workers", type=Path)
    models = commands.add_parser(
        "models", help="Explicitly download the two detector artifacts"
    )
    models.add_argument("--dry-run", action="store_true")
    models.add_argument(
        "--face-backend",
        choices=["mediapipe", "yunet"],
        default="mediapipe",
        help="Use a separate dataset root when comparing detector models",
    )
    imported = commands.add_parser(
        "import-talkinghead", help="Convert upstream *_video_ids.txt; no media download"
    )
    imported.add_argument("--ids-file", type=Path, required=True)
    imported.add_argument("--split", choices=["train", "val", "test"])
    imported.add_argument("--limit", type=positive_int, default=5)
    imported.add_argument("--output", type=Path, required=True)
    for name in ("fetch", "process"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument(
            "--limit",
            type=positive_int,
            default=5,
            help="Maximum source records, in manifest order",
        )
        command.add_argument(
            "--source-id",
            help="Select one manifest source ID before applying the limit",
        )
        command.add_argument(
            "--dry-run",
            action="store_true",
            help="List work without writing or downloading",
        )
        if name == "process":
            command.add_argument(
                "--profile",
                choices=["smoke-test", "quality-long"],
                default="smoke-test",
            )
            command.add_argument(
                "--workers",
                type=Path,
                help="Explicit local backend registry (quality-long only)",
            )
            command.add_argument(
                "--exclusions",
                type=Path,
                help="Reviewed bad source intervals JSONL (quality-long only)",
            )
            command.add_argument("--config", type=Path)
            command.add_argument("--jobs", type=positive_int, default=1,
                                 help="Concurrent source processes, quality-long only (1..64)")
            command.add_argument("--worker-mode", choices=["persistent", "oneshot"], default="persistent",
                                 help="Reuse OCR/VAD models across clips; oneshot uses less idle RAM")
            command.add_argument("--detector-device", choices=["cpu", "gpu"],
                                 help="Override quality config detector_device; GPU requires a supported MediaPipe runtime")
            command.add_argument(
                "--max-clips",
                type=positive_int,
                default=20,
                help="Maximum candidates attempted per source",
            )
            command.add_argument(
                "--scan-seconds",
                type=positive_int,
                default=300,
                help="Scan at most this many seconds from each source's start",
            )
    reviews = commands.add_parser(
        "review-template", help="Create a pending-review JSONL file; never overwrite"
    )
    reviews.add_argument("--output", type=Path, required=True)
    export = commands.add_parser(
        "export", help="Validate approvals and write the training JSONL manifest"
    )
    export.add_argument("--reviews", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--dry-run", action="store_true")
    report = commands.add_parser(
        "report", help="Read candidate counts/rejection reasons; no mutations"
    )
    report.add_argument(
        "--reviews", type=Path, help="Also validate and count reviewed approved footage"
    )
    imported_quality = commands.add_parser(
        "import-discovery",
        help="Import legacy/discovery JSONL without granting approvals",
    )
    imported_quality.add_argument("--input", type=Path, required=True)
    imported_quality.add_argument("--output", type=Path, required=True)
    imported_quality.add_argument("--provenance", type=Path, required=True)
    imported_quality.add_argument("--dry-run", action="store_true")
    package = commands.add_parser(
        "package-hub", help="Build a local approved-only package; never upload"
    )
    package.add_argument("--reviews", type=Path, required=True)
    package.add_argument("--output", type=Path, required=True)
    package.add_argument("--dry-run", action="store_true")
    duplicates = commands.add_parser(
        "duplicates", help="Read local temporal hashes and propose possible reuploads"
    )
    duplicates.add_argument("--output", type=Path, required=True)
    duplicates.add_argument("--max-pairs", type=positive_int, default=100000)
    benchmark = commands.add_parser(
        "benchmark", help="Read-only interval evaluation against reviewed labels"
    )
    benchmark.add_argument("--labels", type=Path, required=True)
    benchmark.add_argument(
        "--stage", choices=["candidates", "eligible", "approved"], default="candidates"
    )
    benchmark.add_argument("--reviews", type=Path)
    gallery = commands.add_parser(
        "gallery",
        help="Create a local evidence gallery inside the ignored dataset root",
    )
    gallery.add_argument("--output", type=Path, required=True)
    return cli


def doctor(root):
    failed = False
    print(f"Python {sys.version.split()[0]} | dataset: {root}")
    for name in ("ffmpeg", "ffprobe"):
        path = shutil.which(name)
        print(f"{name}: {path or 'MISSING'}")
        failed |= path is None
    for name in (
        "numpy",
        "opencv-contrib-python",
        "mediapipe",
        "scenedetect",
        "yt-dlp",
        "yt-dlp-ejs",
    ):
        try:
            print(f"{name}: {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{name}: MISSING")
            failed = True
    runtime = shutil.which("deno") or shutil.which("node")
    print(
        f"YouTube JS runtime (Deno preferred, Node >=22 supported): {runtime or 'MISSING; local-file processing still works'}"
    )
    try:
        from .vision import model_paths

        _, hashes = model_paths(root)
        print("Detector checksums: " + json.dumps(hashes))
    except (OSError, ValueError, KeyError) as exc:
        print(
            f"Detector models: MISSING/INVALID ({exc}); use the explicit models command later"
        )
        failed = True
    return int(failed)


def write_new(path, rows):
    if path.exists():
        raise ValueError(f"Refusing to overwrite {path}; choose a new output filename")
    write_jsonl(path, rows)


def main(argv=None):
    args = parser().parse_args(argv)
    root = args.root.resolve()
    if args.command == "gallery":
        from .quality_gallery import build_gallery

        print(build_gallery(root, args.output))
        return 0
    if args.command == "benchmark":
        from .quality_benchmark import benchmark

        print(
            json.dumps(benchmark(root, args.labels, args.stage, args.reviews), indent=2)
        )
        return 0
    if args.command == "doctor":
        status = doctor(root)
        if args.profile == "quality-long":
            from .quality_backends import load_registry, BACKEND_CHECKS

            try:
                print("av: " + importlib.metadata.version("av"))
                registry = load_registry(args.workers)
                for name in BACKEND_CHECKS:
                    print(
                        f"{name}: {'CONFIGURED (inference smoke test still required)' if name in registry else 'UNVERIFIED / NOT CONFIGURED'}"
                    )
                status |= int(set(registry) != set(BACKEND_CHECKS))
            except (
                ValueError,
                OSError,
                importlib.metadata.PackageNotFoundError,
            ) as exc:
                print(f"Quality prerequisites: {exc}")
                status = 1
        return status
    if args.command == "report":
        result = summary(root)
        if args.reviews:
            rows = build_export(root, args.reviews)
            from .pipeline import candidate_records

            approved = {row["id"] for row in rows}
            from .metadata import unique_hours

            result.update(
                approved_unique_hours=unique_hours(
                    [r for r in candidate_records(root) if r["id"] in approved]
                ),
                approved_samples=len(rows),
                distinct_reviewed_speakers=len({r["speaker_id"] for r in rows}),
            )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "import-discovery":
        from .quality_export import import_discovery

        rows, provenance = import_discovery(args.input)
        if not args.dry_run:
            if (
                args.output.resolve() == args.provenance.resolve()
                or args.output.exists()
                or args.provenance.exists()
            ):
                raise ValueError("Choose distinct new output/provenance paths")
            write_new(args.output, rows)
            write_new(args.provenance, provenance)
        print(
            f"{'Would import' if args.dry_run else 'Imported'} {len(rows)} sources; no approvals or downloads"
        )
        return 0
    if args.command == "package-hub":
        from .quality_export import package_hub

        if args.dry_run:
            rows = build_export(root, args.reviews)
            if any(
                r.get("schema_version") != 2 or not r.get("redistribution_approved")
                for r in rows
            ):
                raise ValueError("Hub package requires v2 and redistribution approvals")
            print(f"Would package {len(rows)} rows; no files written or uploaded")
        else:
            with dataset_lock(root):
                print(
                    f"Packaged {package_hub(root, args.reviews, args.output)} approved rows; no upload"
                )
        return 0
    if args.command == "duplicates":
        from .quality_export import duplicate_proposals
        from .core import write_json

        if args.output.exists():
            raise ValueError("Duplicate report exists; choose a new path")
        write_json(args.output, duplicate_proposals(root, args.max_pairs))
        return 0
    if args.command == "models":
        if args.dry_run:
            from .pipeline import YUNET_URL

            urls = {
                **MODEL_URLS,
                **({"face": YUNET_URL} if args.face_backend == "yunet" else {}),
            }
            print(json.dumps(urls, indent=2))
            return 0
        with dataset_lock(root):
            print(json.dumps(install_models(root, args.face_backend), indent=2))
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
        sources = sources[: args.limit]
        quality = args.command == "process" and args.profile == "quality-long"
        if (
            args.command == "process"
            and not quality
            and (args.workers or args.exclusions)
        ):
            raise ValueError("--workers/--exclusions require --profile quality-long")
        if quality:
            from .quality_config import load_quality_config
            from dataclasses import replace

            config = load_quality_config(args.config)
            if args.detector_device:
                config = replace(config, detector_device=args.detector_device)
            if args.jobs > 64:
                raise ValueError("--jobs must be in 1..64")
        else:
            if args.command == "process" and (args.jobs != 1 or args.worker_mode != "persistent" or args.detector_device):
                raise ValueError("--jobs/--worker-mode/--detector-device require --profile quality-long")
            config = load_config(args.config) if args.command == "process" else None
        if args.dry_run:
            planned = {
                "command": args.command,
                "root": str(root),
                "sources": [s.to_dict() for s in sources],
                "note": "No downloads, detector loading, or filesystem writes",
            }
            if config:
                planned.update(
                    config=asdict(config),
                    max_clips_per_source=args.max_clips,
                    scan_seconds_per_source=args.scan_seconds,
                )
            if quality:
                planned.update(jobs=args.jobs, worker_mode=args.worker_mode)
            print(json.dumps(planned, indent=2))
            return 0
        require_tools()
        with dataset_lock(root):
            if args.command == "fetch":
                result = fetch_sources(root, sources)
                return int(any(row["status"] == "failed" for row in result))
            if quality:
                from .quality_backends import load_registry
                from .quality_pipeline import process_quality
                from .core import read_jsonl

                result = process_quality(
                    root,
                    sources,
                    config,
                    load_registry(args.workers),
                    max_clips=args.max_clips,
                    scan_seconds=args.scan_seconds,
                    exclusions=read_jsonl(args.exclusions) if args.exclusions else [],
                    jobs=args.jobs,
                    persistent_workers=args.worker_mode == "persistent",
                )
            else:
                result = process_sources(
                    root,
                    sources,
                    config,
                    max_clips=args.max_clips,
                    scan_seconds=args.scan_seconds,
                )
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
        print(
            f"{'Validated' if args.dry_run else 'Exported'} {len(rows)} approved samples"
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
