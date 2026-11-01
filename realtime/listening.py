"""Experimental, chunk-level listening cues for clean synthesized PCM.

This selects text conditioning, not a mouth-closure guarantee or microphone VAD.
Preview is side-effect free so cancelled generation cannot consume an onset cue.
"""

from dataclasses import dataclass

import numpy as np

from realtime.contracts import AudioChunk

QUIET_PROMPT = (
    "A single continuous upper-body video call. The person alternates between "
    "speaking in sync with audible speech and quietly listening during silence. "
    "During pauses the lips rest naturally, with gentle blinking and subtle head motion. "
    "During speech the lips and jaw articulate the supplied words clearly. "
    "Stable identity, clothing, body scale, lighting and background throughout the call."
)
ONSET_STRENGTH = 1.5
ONSET_CHUNKS = 2


@dataclass(frozen=True)
class ListeningCue:
    profile: str
    speech_chunks: int
    epoch: int


class ListeningControl:
    def __init__(self, profile: str = "off"):
        if profile not in {"off", "onset_boost"}:
            raise ValueError("Unknown AvatarForever listening profile")
        self.profile = profile
        self.reset()

    @property
    def profiles(self):
        return ("default",) if self.profile == "off" else ("quiet", "onset", "speaking")

    def reset(self):
        self.current = None

    def preview(self, audio: AudioChunk) -> ListeningCue:
        if self.profile == "off":
            return ListeningCue("default", 0, audio.epoch)
        # Ignore padding and future lookahead. Widen before abs(-32768).
        pcm = audio.pcm[: audio.valid_samples].astype(np.int32)
        audible = bool(np.any(np.abs(pcm) > 2))
        previous = self.current
        count = (
            previous.speech_chunks
            if previous is not None and previous.epoch == audio.epoch
            else 0
        )
        count = count + 1 if audible else 0
        mode = (
            "quiet" if not audible else "onset" if count <= ONSET_CHUNKS else "speaking"
        )
        return ListeningCue(mode, count, audio.epoch)

    def commit(self, cue: ListeningCue):
        self.current = cue

    def info(self):
        return {
            "profile": self.profile,
            "qualification": "experimental" if self.profile != "off" else "disabled",
            "mode": self.current.profile if self.current is not None else "unstarted",
            "speech_chunks": self.current.speech_chunks
            if self.current is not None
            else 0,
            "onset_chunks": ONSET_CHUNKS,
            "onset_strength": ONSET_STRENGTH,
        }
