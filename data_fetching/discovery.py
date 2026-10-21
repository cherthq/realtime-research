"""Persistent long-footage queue: python -m data_fetching.discovery --help.

No collection on import. A scheduled wakeup should invoke `run`; the pilot's
cumulative acquisition caps persist across invocations. Scheduling is separate.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import time

from .core import ROOT, Source, file_hash, read_json, read_jsonl, write_json, write_jsonl
from .pipeline import dataset_lock, fetch_one, source_record_path
from .queue import Queue


DEFAULT_QUEUE = ROOT / "data" / "discovery"
DEFAULT_QUERIES = [
    'Wikitongues speaking monologue',
    '"Creative Commons" "webcam" interview',
    '"creative commons" "personal update"',
    '"CC BY" "talk" "camera"',
    '"CC BY" "interview"',
    '"creative commons" "Q&A"',
    '"creative commons" "educational" talk',
]


def ydl_options():
    options = {"quiet": True, "no_warnings": False, "socket_timeout": 30, "retries": 2,
               "noplaylist": True, "cachedir": False}
    if shutil.which("deno"):
        options["js_runtimes"] = {"deno": {}}
    elif shutil.which("node"):
        options["js_runtimes"] = {"node": {}}
    return options


def discover(queue, query, batch=10):
    from yt_dlp import YoutubeDL

    row = queue.db.execute("SELECT * FROM searches WHERE query=?", (query,)).fetchone()
    cursor = row["cursor"] if row else 0
    # Search result ordering can change; source upserts make repetition harmless.
    # Rotate a query after 200 results; a new exploration cycle is timestamped.
    if cursor >= 200:
        cursor = 0
    options = ydl_options() | {"extract_flat": True, "playliststart": cursor + 1, "playlistend": cursor + batch}
    with YoutubeDL(options) as downloader:
        found = downloader.extract_info(f"ytsearch{cursor + batch}:{query}", download=False)
    identities = []
    for entry in found.get("entries", []):
        if not entry:
            continue
        url = entry.get("webpage_url") or entry.get("url")
        if not url or not url.startswith("https://"):
            continue
        identities.append(queue.add({"url": url, "title": entry.get("title"), "duration": entry.get("duration"),
              "uploader": entry.get("uploader") or entry.get("channel"), "channel_id": entry.get("channel_id"),
              "search_query": query, "inspection": "metadata_only", "license": "unknown"}))
    with queue.transaction():
        queue.db.execute("INSERT OR REPLACE INTO searches(query,cursor,last_run,exhausted) VALUES(?,?,?,?)",
                         (query, cursor + batch, time.time(), int(not identities)))
        queue.event("search", {"query": query, "cursor": cursor, "results": len(identities)})
    return identities


def budgets(queue, raw_hours, max_bytes):
    raw = sum(json.loads(s["media"])["duration"] for s in queue.sources() if s["acquired"])
    storage = sum(p.stat().st_size for p in (queue.root.parent / "facetime" / "originals").rglob("*") if p.is_file())
    return max(0, raw_hours * 3600 - raw), max(0, max_bytes - storage)


def acquire(queue, source, raw_hours=10, max_bytes=50_000_000_000):
    from yt_dlp import YoutubeDL

    if source["acquired"]:
        if file_hash(source["original"]) != source["sha256"]:
            raise ValueError("Acquired original changed")
        return {"id": source["id"], "state": "cached"}
    remaining_seconds, remaining_bytes = budgets(queue, raw_hours, max_bytes)
    metadata = json.loads(source["metadata"])
    if source["url"].startswith("local:"):
        from .media import probe
        path = Path(source["url"][6:])
        duration, estimated_bytes = probe(path)["duration"], path.stat().st_size
        definition = Source(id=source["id"], path=str(path), source=metadata.get("source_page", "local source"))
    else:
        with YoutubeDL(ydl_options()) as downloader:
            details = downloader.extract_info(source["url"], download=False)
        duration = details.get("duration")
        if not duration or details.get("is_live"):
            raise ValueError("Cannot budget a source with unknown duration or a live stream")
        metadata.update({key: details.get(key) for key in ("title", "uploader", "uploader_id", "channel_url", "description", "license", "duration", "language", "upload_date") if details.get(key) is not None})
        queue.add(metadata | {"url": source["url"]})
        # Conservative preflight estimate; actual transfer is capped below.
        estimated_bytes = duration * 8_000_000 / 8
        definition = Source(id=source["id"], url=source["url"], source=metadata.get("uploader", "original uploader unverified"),
                            license=metadata.get("license", "unknown"), license_evidence=metadata.get("license_evidence", ""))
    if duration > remaining_seconds or estimated_bytes > remaining_bytes:
        return {"id": source["id"], "state": "budget_deferred", "duration_seconds": duration}
    if shutil.disk_usage(queue.root).free < estimated_bytes * 2 + 2_000_000_000:
        raise ValueError("Insufficient local disk headroom for original plus merge")
    media_root = queue.root.parent / "facetime"
    fetch_one(media_root, definition, max_bytes=remaining_bytes)
    record = read_json(source_record_path(media_root, definition))
    queue.acquired(source["id"], media_root / record["original"], record["sha256"], record["media"])
    return {"id": source["id"], "state": "acquired", "duration_seconds": record["media"]["duration"]}


def rank(queue, source):
    metadata = json.loads(source["metadata"])
    duration = metadata.get("duration") or 0
    longness = 5 if 600 <= duration <= 3600 else 3 if duration > 3600 else 0
    uploader = metadata.get("channel_id") or metadata.get("uploader")
    existing = sum(json.loads(s["metadata"]).get("uploader") == metadata.get("uploader") and bool(s["acquired"]) for s in queue.sources()) if uploader else 0
    rights = 2 if "creative commons" in str(metadata.get("license", "")).lower() or "cc by" in str(metadata.get("license", "")).lower() else 0
    return longness + rights - min(2, existing * 0.25)


def run(queue, *, seconds=1800, raw_hours=10, max_bytes=50_000_000_000, search=True):
    from .long_scan import scan_source

    deadline = time.monotonic() + seconds
    report = queue.report()
    if report["approved_preferred_hours_at_least_60s"] >= 250:
        return {"state": "target_reached", "report": report}
    if search:
        previous = {r["query"]: r["last_run"] or 0 for r in queue.db.execute("SELECT * FROM searches")}
        query = min(DEFAULT_QUERIES, key=lambda q: previous.get(q, 0))
        try:
            discover(queue, query)
        except Exception as exc:
            with queue.transaction():
                queue.event("search_failure", {"query": query, "error": str(exc)[:1000]})
    results = []
    # Existing acquired sources are processed first; fairness is by last activity.
    sources = sorted(queue.sources(), key=lambda s: (not bool(s["acquired"]), -rank(queue, s), s["updated"]))
    for source in sources:
        if time.monotonic() >= deadline:
            break
        if source["state"] in {"screened", "source_rejected"} or source["next_attempt"] > time.time():
            continue
        if not source["acquired"] and not json.loads(source["metadata"]).get("acquisition_selected"):
            continue
        try:
            result = acquire(queue, source, raw_hours=raw_hours, max_bytes=max_bytes)
            results.append(result)
            if result["state"] == "budget_deferred":
                continue
            source = queue.source(source["id"])
            from .dedup import screen
            proposals = screen(queue, source)
            if proposals:
                results.append({"id": source["id"], "duplicate_proposals": proposals})
            results.append(scan_source(queue, source, deadline=deadline))
        except Exception as exc:
            queue.fail(source["id"], str(exc))
            results.append({"id": source["id"], "state": "failed", "error": str(exc)[:1000]})
        finally:
            with queue.transaction():
                queue.db.execute("UPDATE sources SET updated=? WHERE id=?", (time.time(), source["id"]))
    return {"state": "checkpointed", "work": results, "report": queue.snapshot()}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE)
    commands = cli.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="Import discovery metadata; no downloads")
    add.add_argument("manifest", type=Path)
    search = commands.add_parser("search", help="Discover metadata only")
    search.add_argument("query")
    search.add_argument("--batch", type=int, default=10)
    runner = commands.add_parser("run", help="One bounded, resumable acquisition/screening run")
    runner.add_argument("--seconds", type=int, default=1800)
    runner.add_argument("--max-raw-hours", type=float, default=10)
    runner.add_argument("--max-bytes", type=int, default=50_000_000_000)
    runner.add_argument("--no-search", action="store_true")
    commands.add_parser("report")
    commands.add_parser("review-template")
    review = commands.add_parser("review")
    review.add_argument("manifest", type=Path)
    source_review = commands.add_parser("source-review")
    source_review.add_argument("manifest", type=Path)
    publish_cmd = commands.add_parser("publish")
    publish_cmd.add_argument("--repo-id", default="garyzgao/facetime-video")
    publish_cmd.add_argument("--dry-run", action="store_true")
    args = cli.parse_args(argv)
    for name in ("seconds", "max_raw_hours", "max_bytes", "batch"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            cli.error(f"{name} must be positive")
    # Every mutation/media command shares the original pipeline's writer lock.
    # Queue readers use SQLite snapshots and do not remove or acquire this lock.
    queue = Queue(args.queue_root)
    try:
        if args.command == "report":
            result = queue.report()
        else:
            with dataset_lock(queue.root.parent / "facetime"):
                if args.command == "add":
                    result = [queue.add(row) for row in read_jsonl(args.manifest)]
                elif args.command == "search":
                    result = discover(queue, args.query, args.batch)
                elif args.command == "run":
                    result = run(queue, seconds=args.seconds, raw_hours=args.max_raw_hours,
                                 max_bytes=args.max_bytes, search=not args.no_search)
                elif args.command == "source-review":
                    for row in read_jsonl(args.manifest):
                        identity = row.pop("id")
                        queue.source_review(identity, **row)
                    result = queue.snapshot()
                elif args.command == "review":
                    for row in read_jsonl(args.manifest):
                        queue.review(row)
                    result = queue.snapshot()
                elif args.command == "review-template":
                    result = [dict(r) | {"reviewer": "", "notes": "", "inspection": "uninspected"}
                              for r in queue.db.execute("SELECT * FROM intervals WHERE quality='pending'")]
                elif args.command == "publish":
                    from .hub_storage import publish
                    result = publish(queue, args.repo_id, args.dry_run)
                else:
                    raise AssertionError(args.command)
        print(json.dumps(result, indent=2))
        return int(args.command == "run" and any(r.get("state") == "failed" for r in result.get("work", [])))
    finally:
        queue.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
