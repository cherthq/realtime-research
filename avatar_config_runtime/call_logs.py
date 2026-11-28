"""Private call ledger and short-lived, file-scoped download authorization."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time
from datetime import datetime, timezone

FILES = {
    "audio": ("audio.wav", "audio/wav"),
    "caller": ("caller.wav", "audio/wav"),
    "agent": ("agent.wav", "audio/wav"),
    "video": ("video.mp4", "video/mp4"),
    "transcript": ("transcript.txt", "text/plain; charset=utf-8"),
}
ID = re.compile(r"^[a-f0-9]{24}$")
RESERVE = 5 * 1024**3


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    with temp.open("w") as out:
        os.chmod(temp, 0o600)
        json.dump(value, out, ensure_ascii=False)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp, path)


class CallLogs:
    def __init__(self, root: Path, secret: str):
        self.root, self.secret = root, secret
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = root / "calls.sqlite3"
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, started TEXT, source TEXT, status TEXT, duration REAL, version TEXT, error TEXT)"
            )
            db.execute(
                "UPDATE calls SET status='interrupted', error='Service restarted before recording finished' WHERE status IN ('recording','processing')"
            )
        os.chmod(self.db, 0o600)

    def connect(self):
        return sqlite3.connect(self.db, timeout=10)

    def directory(self, ident):
        if not isinstance(ident, str) or not ID.fullmatch(ident):
            raise ValueError("Invalid call ID")
        return self.root / ident

    def begin(self, ident, source, version):
        folder = self.directory(ident)
        folder.mkdir(mode=0o700)
        with self.connect() as db:
            db.execute(
                "INSERT INTO calls VALUES (?,?,?,?,?,?,?)",
                (ident, utc(), source, "recording", 0, version, None),
            )
        return folder

    def update(self, ident, status, duration, error=None, source=None):
        with self.connect() as db:
            db.execute(
                "UPDATE calls SET status=?,duration=?,error=?,source=COALESCE(?,source) WHERE id=?",
                (status, round(duration, 3), error, source, ident),
            )

    def transcript(self, ident, rows):
        folder = self.directory(ident)
        atomic_json(folder / "transcript.json", rows)
        text = "\n\n".join(
            f"[{r['seconds']:.1f}s] {'Agent' if r['speaker'] == 'assistant' else 'Caller'}: {r['text']}"
            for r in rows
            if r["text"]
        )
        temp = folder / "transcript.txt.tmp"
        temp.write_text(text, encoding="utf8")
        temp.chmod(0o600)
        os.replace(temp, folder / "transcript.txt")

    def item(self, row, detail=False):
        ident, started, source, status, duration, version, error = row
        folder = self.directory(ident)
        files = {
            key: {"bytes": (folder / name).stat().st_size}
            for key, (name, _) in FILES.items()
            if (folder / name).is_file()
            and not (folder / name).is_symlink()
            and (status not in ("recording", "processing") or key == "transcript")
        }
        item = dict(
            id=ident,
            startedAt=started,
            source=source,
            status=status,
            duration=duration,
            deploymentId=version,
            error=error,
            files=files,
        )
        if detail:
            path = folder / "transcript.json"
            item["transcript"] = json.loads(path.read_text()) if path.exists() else []
        return item

    def listing(self, offset=0):
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM calls ORDER BY started DESC,id DESC LIMIT 51 OFFSET ?",
                (offset,),
            ).fetchall()
        return dict(
            calls=[self.item(row) for row in rows[:50]],
            nextOffset=offset + 50 if len(rows) > 50 else None,
            storageFreeBytes=shutil.disk_usage(self.root).free,
            storageReserveBytes=RESERVE,
            recordingEnabled=True,
            historyNote="Recordings begin when call logging is enabled. Earlier calls were not saved.",
        )

    def get(self, ident):
        self.directory(ident)
        with self.connect() as db:
            row = db.execute("SELECT * FROM calls WHERE id=?", (ident,)).fetchone()
        if row is None:
            raise KeyError("Call not found")
        return self.item(row, True)

    def file(self, ident, kind):
        if kind not in FILES or kind not in self.get(ident)["files"]:
            raise KeyError("Recording not available")
        name, media = FILES[kind]
        return self.directory(ident) / name, media

    def signature(self, ident, kind, expiry):
        return hmac.new(
            self.secret.encode(),
            f"call-download\n{ident}\n{kind}\n{expiry}".encode(),
            hashlib.sha256,
        ).hexdigest()

    def ticket(self, ident, kind):
        self.file(ident, kind)
        expiry = int(time.time()) + 300
        return dict(
            id=ident,
            file=kind,
            expires=str(expiry),
            signature=self.signature(ident, kind, expiry),
        )

    def verify(self, ident, kind, expiry, signature):
        if (
            not ID.fullmatch(ident or "")
            or kind not in FILES
            or not str(expiry).isdigit()
        ):
            return False
        expires = int(expiry)
        return time.time() < expires <= time.time() + 301 and hmac.compare_digest(
            self.signature(ident, kind, expires), signature
        )
