"""Persistent discovery ledger. Network/media work lives outside transactions.

All times are seconds on a canonical original's timeline. Quality, rights,
deduplication and publication are independent decisions, never inferred from
successful processing. SQLite is authoritative; JSONL is an interchange format.
"""

from contextlib import contextmanager
import json
import math
from pathlib import Path
import sqlite3
import time

from .core import file_hash, fingerprint, read_jsonl, write_json, write_jsonl, youtube_url


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1] + 1e-6:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def seconds(intervals):
    return sum(b - a for a, b in union(intervals))


class Queue:
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "queue.sqlite3", timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS sources (
          id TEXT PRIMARY KEY, url TEXT NOT NULL UNIQUE, metadata TEXT NOT NULL,
          created REAL NOT NULL, updated REAL NOT NULL,
          state TEXT NOT NULL DEFAULT 'discovered',
          canonical TEXT NOT NULL, offset REAL NOT NULL DEFAULT 0,
          duplicate_status TEXT NOT NULL DEFAULT 'unverified',
          permission_status TEXT NOT NULL DEFAULT 'pending',
          redistribution_status TEXT NOT NULL DEFAULT 'pending',
          evidence TEXT NOT NULL DEFAULT '', reviewer TEXT NOT NULL DEFAULT '',
          original TEXT, sha256 TEXT, media TEXT,
          attempts INTEGER NOT NULL DEFAULT 0, next_attempt REAL NOT NULL DEFAULT 0,
          error TEXT, acquired REAL);
        CREATE TABLE IF NOT EXISTS windows (
          source_id TEXT NOT NULL REFERENCES sources(id), revision TEXT NOT NULL,
          start REAL NOT NULL, end REAL NOT NULL, completed REAL NOT NULL,
          PRIMARY KEY(source_id, revision, start, end));
        CREATE TABLE IF NOT EXISTS intervals (
          id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id),
          revision TEXT NOT NULL, start REAL NOT NULL, end REAL NOT NULL,
          framing TEXT NOT NULL, issues TEXT NOT NULL, method TEXT NOT NULL,
          quality TEXT NOT NULL DEFAULT 'pending', speaker_id TEXT,
          conversation_id TEXT, split TEXT, reviewer TEXT, reviewed REAL,
          inspection TEXT, notes TEXT, artifact TEXT,
          UNIQUE(source_id,revision,start,end));
        CREATE TABLE IF NOT EXISTS events (
          id INTEGER PRIMARY KEY, at REAL NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS searches (
          query TEXT PRIMARY KEY, cursor INTEGER NOT NULL DEFAULT 0,
          last_run REAL, exhausted INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS uploads (
          path TEXT PRIMARY KEY, sha256 TEXT NOT NULL, commit_id TEXT NOT NULL, uploaded REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        ''')

    def close(self):
        self.db.close()

    @contextmanager
    def transaction(self):
        try:
            self.db.execute("BEGIN IMMEDIATE")
            yield
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def event(self, kind, payload):
        self.db.execute("INSERT INTO events(at,kind,payload) VALUES(?,?,?)",
                        (time.time(), kind, json.dumps(payload, allow_nan=False)))

    def add(self, row):
        url = youtube_url(row["url"]) if row.get("url") else "local:" + str(Path(row["path"]).resolve())
        identity = fingerprint(url)[:20]
        duration = row.get("duration")
        if duration is not None and number(duration, "duration") <= 0:
            raise ValueError("duration must be positive")
        now = time.time()
        # Existing rights/review state is never reset by refreshed search metadata.
        with self.transaction():
            existing = self.db.execute("SELECT * FROM sources WHERE url=?", (url,)).fetchone()
            if existing:
                metadata = json.loads(existing["metadata"])
                metadata.update({k: v for k, v in row.items() if v is not None})
                self.db.execute("UPDATE sources SET metadata=?,updated=? WHERE id=?",
                                (json.dumps(metadata, allow_nan=False), now, existing["id"]))
                return existing["id"]
            self.db.execute("INSERT INTO sources(id,url,metadata,created,updated,canonical) VALUES(?,?,?,?,?,?)",
                            (identity, url, json.dumps(row, allow_nan=False), now, now, identity))
            self.event("discovered", {"id": identity, "url": url})
        return identity

    def source(self, identity):
        row = self.db.execute("SELECT * FROM sources WHERE id=?", (identity,)).fetchone()
        if not row:
            raise ValueError(f"Unknown source {identity}")
        return dict(row)

    def sources(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM sources ORDER BY created,id")]

    def fail(self, identity, message):
        source = self.source(identity)
        attempts = source["attempts"] + 1
        with self.transaction():
            self.db.execute("UPDATE sources SET attempts=?,next_attempt=?,error=? WHERE id=?",
                            (attempts, time.time() + min(86400, 300 * 2 ** min(attempts, 8)), str(message)[:2000], identity))
            self.event("failure", {"id": identity, "error": str(message)[:2000]})

    def acquired(self, identity, original, digest, media):
        source = self.source(identity)
        if source["sha256"] and source["sha256"] != digest:
            raise ValueError("Original changed; existing acquisition left untouched")
        with self.transaction():
            identical = self.db.execute("SELECT * FROM sources WHERE sha256=? AND id<>? ORDER BY acquired LIMIT 1",
                                        (digest, identity)).fetchone()
            canonical = identical["canonical"] if identical else source["canonical"]
            offset = identical["offset"] if identical else source["offset"]
            status = "exact_duplicate" if identical else source["duplicate_status"]
            self.db.execute('''UPDATE sources SET state='acquired',original=?,sha256=?,media=?,
              acquired=COALESCE(acquired,?),error=NULL,attempts=0,next_attempt=0,
              canonical=?,offset=?,duplicate_status=? WHERE id=?''',
              (str(original), digest, json.dumps(media), time.time(), canonical, offset, status, identity))
            self.event("acquired", {"id": identity, "sha256": digest, "duplicate": bool(identical)})

    def source_review(self, identity, *, reviewer, evidence, permission="pending",
                      redistribution="pending", unique=False, canonical=None, offset=0,
                      quality_rejection=None):
        self.source(identity)
        if not reviewer.strip() or not evidence.strip():
            raise ValueError("Reviewer and evidence are required")
        if permission not in {"pending", "approved", "rejected"} or redistribution not in {"pending", "approved", "rejected"}:
            raise ValueError("Invalid permission decision")
        offset = number(offset, "offset")
        if unique and canonical:
            raise ValueError("Choose unique or a duplicate canonical source")
        status = None
        if canonical:
            parent = self.source(canonical)
            if parent["canonical"] != canonical or canonical == identity:
                raise ValueError("Choose a different canonical root; duplicate chains are forbidden")
            # Existing child mappings must not silently inherit a changed offset.
            if self.db.execute("SELECT 1 FROM sources WHERE canonical=? AND id<>?", (identity, identity)).fetchone():
                raise ValueError("Reconcile dependent duplicate sources before remapping this root")
            status = "reviewed_duplicate"
        elif unique:
            canonical, offset, status = identity, 0, "reviewed_unique"
        with self.transaction():
            self.db.execute("UPDATE sources SET permission_status=?,redistribution_status=?,evidence=?,reviewer=? WHERE id=?",
                            (permission, redistribution, evidence, reviewer, identity))
            if status:
                self.db.execute("UPDATE sources SET canonical=?,offset=?,duplicate_status=? WHERE id=?",
                                (canonical, offset, status, identity))
            if quality_rejection:
                self.db.execute("UPDATE sources SET state='source_rejected',error=NULL WHERE id=?", (identity,))
                self.event("source_quality_rejection", {"id": identity, "reviewer": reviewer, "reason": quality_rejection})
            self.event("source_review", {"id": identity, "reviewer": reviewer, "evidence": evidence,
                        "permission": permission, "redistribution": redistribution, "canonical": canonical, "offset": offset})

    def complete_window(self, identity, revision, start, end, candidates):
        source = self.source(identity)
        duration = json.loads(source["media"])["duration"]
        if not 0 <= start < end <= duration + 0.05:
            raise ValueError("Invalid scan coverage")
        with self.transaction():
            for item in candidates:
                a, b = item["start"], item["end"]
                if not start <= a < b <= end + 1e-6:
                    raise ValueError("Candidate outside completed window")
                clip_id = fingerprint([identity, revision, a, b])[:24]
                self.db.execute('''INSERT OR IGNORE INTO intervals
                 (id,source_id,revision,start,end,framing,issues,method)
                 VALUES(?,?,?,?,?,?,?,?)''', (clip_id, identity, revision, a, b,
                    item["framing"], json.dumps(item.get("issues", [])), item["method"]))
            self.db.execute("INSERT OR IGNORE INTO windows VALUES(?,?,?,?,?)",
                            (identity, revision, start, end, time.time()))
            self.event("scan_completed", {"id": identity, "revision": revision, "start": start, "end": end})

    def review(self, row):
        clip = self.db.execute("SELECT * FROM intervals WHERE id=?", (row["id"],)).fetchone()
        if not clip:
            raise ValueError("Unknown interval")
        quality = row["quality"]
        if quality not in {"pending", "approved", "rejected"}:
            raise ValueError("Invalid quality decision")
        if not row.get("reviewer") or not row.get("notes"):
            raise ValueError("Reviewer and decision notes are required")
        if quality == "approved":
            if row.get("inspection") != "full_video_audio":
                raise ValueError("Full interval video and audio inspection is required")
            if not row.get("speaker_id") or not row.get("conversation_id"):
                raise ValueError("Reviewed speaker and recording groups are required")
            if row.get("split") not in {"train", "val", "test"}:
                raise ValueError("Choose train, val or test")
            source = self.source(clip["source_id"])
            if source["permission_status"] != "approved" or not self.unique_confirmed(source):
                raise ValueError("Source permission and duplicate review must precede approval")
            if not source["original"] or file_hash(source["original"]) != source["sha256"]:
                raise ValueError("Original is missing or changed; approval refused")
            # Include canonical recording groups, even for reencoded reuploads.
            for other in self.approved():
                if (other["canonical"] == source["canonical"] or
                    other["speaker_id"] == row["speaker_id"] or
                    other["conversation_id"] == row["conversation_id"]):
                    if other["split"] != row["split"] and other["id"] != row["id"]:
                        raise ValueError("Cross-split speaker/recording leakage")
        with self.transaction():
            self.db.execute('''UPDATE intervals SET quality=?,speaker_id=?,conversation_id=?,split=?,
              reviewer=?,reviewed=?,inspection=?,notes=? WHERE id=?''',
              (quality, row.get("speaker_id"), row.get("conversation_id"), row.get("split"),
               row["reviewer"], time.time(), row.get("inspection"), row["notes"], row["id"]))
            self.event("interval_review", row)

    def unique_confirmed(self, source):
        root = self.source(source["canonical"])
        return root["duplicate_status"] == "reviewed_unique" and source["duplicate_status"] in {
            "reviewed_unique", "reviewed_duplicate", "exact_duplicate"}

    def approved(self):
        rows = self.db.execute('''SELECT i.*,s.canonical,s.offset,s.permission_status,s.duplicate_status
          FROM intervals i JOIN sources s ON s.id=i.source_id WHERE i.quality='approved' ''')
        return [dict(r) for r in rows if r["permission_status"] == "approved" and self.unique_confirmed(self.source(r["source_id"]))]

    def report(self):
        sources = self.sources()
        candidates = [dict(r) for r in self.db.execute("SELECT * FROM intervals WHERE quality<>'rejected'")]
        approved = self.approved()

        def grouped(rows, minimum=None, short=False):
            groups = {}
            for r in rows:
                source = self.source(r["source_id"])
                length = r["end"] - r["start"]
                if minimum is not None and length <= minimum:
                    continue
                if short and length >= 60:
                    continue
                groups.setdefault(source["canonical"], []).append((r["start"] + source["offset"], r["end"] + source["offset"]))
            return sum(seconds(v) for v in groups.values()) / 3600

        raw_found, raw_acquired, confirmed_found, confirmed_acquired = {}, {}, {}, {}
        for s in sources:
            duration = json.loads(s["media"])["duration"] if s["media"] else json.loads(s["metadata"]).get("duration")
            if duration is None:
                continue
            interval = (s["offset"], s["offset"] + duration)
            raw_found.setdefault(s["canonical"], []).append(interval)
            if s["acquired"]:
                raw_acquired.setdefault(s["canonical"], []).append(interval)
            if self.unique_confirmed(s):
                confirmed_found.setdefault(s["canonical"], []).append(interval)
                if s["acquired"]:
                    confirmed_acquired.setdefault(s["canonical"], []).append(interval)
        hours = lambda groups: sum(seconds(v) for v in groups.values()) / 3600
        rejections = {}
        for r in self.db.execute("SELECT notes FROM intervals WHERE quality='rejected'"):
            rejections[r[0]] = rejections.get(r[0], 0) + 1
        for r in self.db.execute("SELECT payload FROM events WHERE kind='source_quality_rejection'"):
            reason = json.loads(r[0])["reason"]
            rejections[reason] = rejections.get(reason, 0) + 1
        issues = {}
        for r in self.db.execute("SELECT issues FROM intervals"):
            for issue in json.loads(r[0]):
                issues[issue] = issues.get(issue, 0) + 1
        speakers = {}
        for speaker in {r["speaker_id"] for r in approved}:
            speakers[speaker] = grouped([r for r in approved if r["speaker_id"] == speaker])
        return {"at": time.time(), "sources": len(sources),
          "raw_discovered_hours_provisional": hours(raw_found),
          "raw_acquired_hours_provisional": hours(raw_acquired),
          "unique_raw_discovered_hours_confirmed": hours(confirmed_found),
          "unique_raw_acquired_hours_confirmed": hours(confirmed_acquired),
          "candidate_clean_hours_provisional": grouped(candidates),
          "candidate_clean_hours_dedup_reviewed": grouped([r for r in candidates if self.unique_confirmed(self.source(r["source_id"]))]),
          "approved_usable_hours_all_lengths": grouped(approved),
          "approved_preferred_hours_at_least_60s": grouped([r for r in approved if r["end"] - r["start"] >= 60]),
          "approved_continuous_hours_gt_60s": grouped(approved, 60),
          "approved_continuous_hours_gt_120s": grouped(approved, 120),
          "approved_continuous_hours_gt_300s": grouped(approved, 300),
          "approved_short_hours_lt_60s": grouped(approved, short=True),
          "reviewed_speaker_count": len(speakers), "approved_hours_by_speaker": speakers,
          "source_states": {state: sum(s["state"] == state for s in sources) for state in {s["state"] for s in sources}},
          "rejection_reasons": rejections, "automatic_issue_flags": issues,
          "pending_intervals": sum(r["quality"] == "pending" for r in candidates),
          "blocked_sources": [{"id": s["id"], "error": s["error"]} for s in sources if s["error"]],
          "note": "Sampled checks are provisional. Raw/clean uniqueness requires duplicate review. Long-duration metrics overlap; do not add them."}

    def snapshot(self):
        rows = []
        for s in self.sources():
            row = {**json.loads(s["metadata"]), **{k: v for k, v in s.items() if k not in {"metadata", "media"}}}
            row["media"] = json.loads(s["media"]) if s["media"] else None
            row["intervals"] = [dict(r) for r in self.db.execute("SELECT * FROM intervals WHERE source_id=? ORDER BY start", (s["id"],))]
            rows.append(row)
        write_jsonl(self.root / "candidates.jsonl", rows)
        report = self.report()
        previous = self.root / "report.json"
        if previous.exists():
            old = json.loads(previous.read_text())
            report["change_since_previous_snapshot"] = {k: v - old.get(k, 0) for k, v in report.items()
                                                        if isinstance(v, (int, float)) and k != "at"}
        write_json(previous, report)
        return report
