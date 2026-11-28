"""Prepare one greeting using the existing voice session, without a call or GPU.

This is an explicit, billable operator command, never an idle background session.
"""

import argparse
import asyncio
from pathlib import Path
import time

import numpy as np

from realtime.config import Settings
from realtime.contracts import ResponseBuffer
from realtime.transcripts import TranscriptLedger
from .agent import LiveAgent, MODEL, VOICE
from .greeting import GreetingCache, greeting_text, identity


class Capture:
    def __init__(self):
        self.buffer = ResponseBuffer()
        self.transcripts = TranscriptLedger()
        self.error = None

    @property
    def response_active(self):
        return self.buffer.active

    async def begin_voice_response(self, name):
        return self.buffer.begin(name)

    async def cancel_media(self, count=False):
        self.buffer.cancel()


async def prepare(settings, instructions, recording, folder):
    sink = Capture()
    voice = LiveAgent(settings, sink)
    voice.instructions = instructions
    voice.recording_enabled = recording
    # greeting_cache intentionally stays disabled: never cache another cache.
    try:
        await voice.start()
        await voice.greet()
        deadline = time.monotonic() + 30
        quiet_since = None
        while time.monotonic() < deadline:
            if voice.failure:
                raise RuntimeError(voice.failure)
            voice.microphone(bytes(1920))  # continuous provider's 20ms input clock
            await asyncio.sleep(0.02)
            if voice.turns and not voice.active and len(sink.buffer.samples):
                quiet_since = quiet_since or time.monotonic()
                if time.monotonic() - quiet_since >= 1:
                    break
            else:
                quiet_since = None
        else:
            raise TimeoutError("Greeting did not finish")
        if voice.turns != 1:
            raise ValueError("Greeting produced more than one response")
        row = sink.transcripts.rows["assistant:" + voice.segment]
        pcm = sink.buffer.samples.copy().astype(np.int16)
        descriptor = identity(MODEL, VOICE, instructions, greeting_text(recording))
        # Exact transcript validation rejects extra prose, truncated speech, or a
        # missing recording notice before anything replaces the current cache.
        path = GreetingCache(folder).save(descriptor, pcm, row["cues"])
        return dict(key=path.stem, duration_seconds=round(len(pcm) / 48000, 3))
    finally:
        await voice.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instructions-file", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/greeting-cache"))
    parser.add_argument("--recording", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(prepare(Settings(_env_file=args.env_file),
                                 args.instructions_file.read_text(), args.recording, args.cache_dir))
    print(f"Prepared greeting {result['key']} ({result['duration_seconds']}s)")


if __name__ == "__main__":
    main()
