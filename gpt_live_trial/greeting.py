"""Private, bounded PCM cache. Video always uses the live continuous renderer."""

from dataclasses import dataclass
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from realtime.contracts import SAMPLE_RATE

MAX_SECONDS = 12
MAX_FILE_BYTES = 2_000_000


def greeting_text(recording):
    notice = "This call is recorded. " if recording else ""
    return f"Hi, I'm Chert. {notice}How can I help you?"


def normalized(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def identity(model, voice, instructions, text):
    # Image changes need no invalidation: every frame is rendered for this call's
    # actual image/history. Prompt/voice/notice changes cannot reuse stale speech.
    return dict(version=1, model=model, voice=voice, text=text, rate=SAMPLE_RATE,
                instructions_sha256=hashlib.sha256(instructions.encode()).hexdigest())


def key(descriptor):
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class GreetingClip:
    pcm: np.ndarray
    cues: tuple


def validate(pcm, cues, text):
    if pcm.dtype != np.dtype("int16") or pcm.ndim != 1 or not SAMPLE_RATE // 2 <= len(pcm) <= SAMPLE_RATE * MAX_SECONDS:
        raise ValueError("Invalid greeting PCM")
    if not np.any(pcm) or not cues or len(cues) > 256:
        raise ValueError("Missing greeting audio or transcript")
    previous = -1
    for sample, words in cues:
        if type(sample) is not int or not 0 <= sample <= len(pcm) or sample < previous or not isinstance(words, str) or len(words) > 1000:
            raise ValueError("Invalid greeting caption")
        previous = sample
    if normalized(cues[-1][1]) != normalized(text):
        raise ValueError("Greeting transcript does not match the approved words")


class GreetingCache:
    def __init__(self, folder: Path):
        self.folder = folder

    def load(self, descriptor):
        """A missing/corrupt/stale cache uses the ordinary live greeting."""
        try:
            with (self.folder / (key(descriptor) + ".json")).open("rb") as handle:
                raw = handle.read(MAX_FILE_BYTES + 1)
            if len(raw) > MAX_FILE_BYTES:
                return None
            payload = json.loads(raw)
            if payload["identity"] != descriptor:
                return None
            pcm_bytes = base64.b64decode(payload["pcm"], validate=True)
            if hashlib.sha256(pcm_bytes).hexdigest() != payload["sha256"]:
                return None
            pcm = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.int16)
            cues = tuple(tuple(cue) for cue in payload["cues"])
            validate(pcm, cues, descriptor["text"])
            pcm.flags.writeable = False
            return GreetingClip(pcm, cues)
        except (OSError, ValueError, KeyError, TypeError, OverflowError):
            return None

    def save(self, descriptor, pcm, cues):
        validate(pcm, cues, descriptor["text"])
        raw = pcm.astype("<i2").tobytes()
        payload = json.dumps(dict(identity=descriptor, pcm=base64.b64encode(raw).decode(),
                                  sha256=hashlib.sha256(raw).hexdigest(), cues=cues)).encode()
        if len(payload) > MAX_FILE_BYTES:
            raise ValueError("Greeting cache is too large")
        self.folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        target = self.folder / (key(descriptor) + ".json")
        fd, temporary = tempfile.mkstemp(dir=self.folder)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return target
