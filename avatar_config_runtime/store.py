"""Immutable prompt/image versions with atomic activation and private assets."""
from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import os
from pathlib import Path
import re
import secrets
import sqlite3
import warnings

from PIL import Image, ImageOps

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 24_000_000


class InvalidConfiguration(ValueError):
    pass


class RevisionConflict(ValueError):
    pass


@dataclass(frozen=True)
class Version:
    id: str
    prompt: str
    image_sha256: str
    created_at: str
    source: str
    restored_from: str | None = None


def validate_prompt(prompt):
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 12000:
        raise InvalidConfiguration('Enter a prompt between 1 and 12,000 characters.')
    return prompt


def normalize_image(encoded):
    if not isinstance(encoded, str) or len(encoded) > MAX_BYTES * 4 // 3 + 100:
        raise InvalidConfiguration('Choose a PNG or JPG smaller than 10 MB.')
    match = re.fullmatch(r'data:image/(png|jpeg);base64,([A-Za-z0-9+/]+={0,2})', encoded)
    if not match:
        raise InvalidConfiguration('Choose a PNG or JPG image.')
    try:
        raw = base64.b64decode(match[2], validate=True)
        if len(raw) > MAX_BYTES:
            raise ValueError()
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as image:
                expected = 'PNG' if match[1] == 'png' else 'JPEG'
                if image.format != expected or image.width * image.height > MAX_PIXELS or getattr(image, 'n_frames', 1) != 1:
                    raise ValueError()
                image.load()
                output = ImageOps.fit(ImageOps.exif_transpose(image).convert('RGB'), (640, 480), method=Image.Resampling.LANCZOS)
                buffer = BytesIO()
                output.save(buffer, format='PNG')
                return buffer.getvalue()
    except (OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise InvalidConfiguration('Could not read this image. Use a still PNG or JPG under 24 megapixels.') from None


class ConfigurationStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.assets = self.root / 'images'
        self.assets.mkdir(exist_ok=True, mode=0o700)
        self.database = self.root / 'versions.sqlite3'
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS versions (
                    id TEXT PRIMARY KEY, prompt TEXT NOT NULL, image_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL, source TEXT NOT NULL, restored_from TEXT
                );
                CREATE TABLE IF NOT EXISTS active (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    version_id TEXT NOT NULL REFERENCES versions(id), revision INTEGER NOT NULL
                );
            ''')
        os.chmod(self.database, 0o600)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    def put_image(self, raw: bytes):
        sha = hashlib.sha256(raw).hexdigest()
        target = self.assets / f'{sha}.png'
        if target.exists():
            self.image_path(sha)
            return sha
        temporary = self.assets / f'.{secrets.token_hex(16)}.tmp'
        try:
            with temporary.open('xb') as output:
                os.chmod(temporary, 0o600)
                output.write(raw)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return sha

    def image_path(self, sha):
        if not isinstance(sha, str) or not re.fullmatch(r'[a-f0-9]{64}', sha):
            raise InvalidConfiguration('Invalid image reference.')
        path = self.assets / f'{sha}.png'
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise InvalidConfiguration('The saved image is unavailable or damaged.')
        return path

    @staticmethod
    def version(row):
        return Version(**dict(row))

    def current(self):
        with self.connect() as db:
            row = db.execute('SELECT versions.*, active.revision FROM active JOIN versions ON versions.id=active.version_id').fetchone()
        if row is None:
            raise InvalidConfiguration('No deployment has been initialized.')
        data = dict(row)
        revision = data.pop('revision')
        version = Version(**data)
        self.image_path(version.image_sha256)
        return revision, version

    def get(self, ident):
        with self.connect() as db:
            row = db.execute('SELECT * FROM versions WHERE id=?', (ident,)).fetchone()
        if row is None:
            raise InvalidConfiguration('Deployment not found.')
        version = self.version(row)
        self.image_path(version.image_sha256)
        return version

    def bootstrap(self, prompt, image):
        validate_prompt(prompt)
        # Preserve the verified live reference byte-for-byte, including its geometry.
        with self.connect() as db:
            if db.execute('SELECT 1 FROM active').fetchone():
                return self.current()
        sha = self.put_image(Path(image).read_bytes())
        try:
            self.activate(prompt, sha, expected_revision=0, source='baseline')
        except RevisionConflict:
            pass  # Another process initialized the same store first.
        return self.current()

    def activate(self, prompt, image_sha256, *, expected_revision, source='publish', restored_from=None):
        validate_prompt(prompt)
        self.image_path(image_sha256)
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 0:
            raise InvalidConfiguration('A valid deployment revision is required.')
        version = Version(secrets.token_hex(16), prompt, image_sha256, datetime.now(timezone.utc).isoformat(), source, restored_from)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            active = db.execute('SELECT revision FROM active').fetchone()
            current_revision = active['revision'] if active else 0
            if current_revision != expected_revision:
                raise RevisionConflict('Deployment changed in another window. Reload before publishing.')
            db.execute('INSERT INTO versions VALUES(?,?,?,?,?,?)', tuple(version.__dict__.values()))
            db.execute('INSERT INTO active VALUES(1,?,?) ON CONFLICT(singleton) DO UPDATE SET version_id=excluded.version_id, revision=excluded.revision', (version.id, current_revision + 1))
        return current_revision + 1, version

    def history(self, limit=20):
        with self.connect() as db:
            rows = db.execute('SELECT versions.* FROM versions WHERE id != (SELECT version_id FROM active) ORDER BY rowid DESC LIMIT ?', (limit,)).fetchall()
        return [self.version(row) for row in rows]
