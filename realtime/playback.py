"""Frame-associated browser feedback, separate from physical audio-device timing."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import time

from livekit import rtc

from .contracts import FRAME_SAMPLES, SAMPLE_RATE


def frame_id(epoch: int, sample_start: int, valid: bool) -> int:
    # Two uint16 fields fit LiveKit's uint32 frame ID. At 25 FPS the low half
    # covers over 43 minutes; the demo call is capped at ten minutes.
    index = sample_start // FRAME_SAMPLES + 1 if valid else 0
    if not 0 <= epoch < 65535 or not 0 <= index < 65536 or sample_start % FRAME_SAMPLES:
        raise ValueError("Playback frame position exceeds the session contract")
    return (epoch + 1) * 65536 + index  # zero is reserved for absent/default metadata


class PlaybackLedger:
    def __init__(self):
        self.sent = OrderedDict()
        self.positions = OrderedDict()
        self.browser_seen = False
        self.reports = 0
        self.cancel_epoch = 0
        self.cancel_started = None
        self.cancel_ack = asyncio.Event()
        self.cancel_roundtrip_ms = None
        self.browser_mute_ms = None
        self.last_frame_id = None

    def capture(self, ident: int, epoch: int, sample_start: int, valid: bool):
        self.sent[ident] = (epoch, sample_start, valid)
        self.sent.move_to_end(ident)
        while len(self.sent) > 4096:
            self.sent.popitem(last=False)

    def cancelled(self, epoch: int):
        self.cancel_epoch = epoch
        self.cancel_started = time.monotonic()
        self.cancel_roundtrip_ms = self.browser_mute_ms = None
        self.cancel_ack.clear()

    def feedback(
        self,
        ident: int | None,
        audio_playing: bool,
        cancel_epoch: int | None = None,
        mute_ms: float | None = None,
    ):
        accepted = False
        if ident is not None and ident in self.sent:
            epoch, position, valid = self.sent[ident]
            self.browser_seen = True
            self.reports += 1
            self.last_frame_id = ident
            if audio_playing and valid:
                self.positions[epoch] = max(position, self.positions.get(epoch, 0))
                while len(self.positions) > 32:
                    self.positions.popitem(last=False)
            accepted = True
        if (
            cancel_epoch == self.cancel_epoch
            and self.cancel_started is not None
            and not self.cancel_ack.is_set()
        ):
            self.cancel_roundtrip_ms = (time.monotonic() - self.cancel_started) * 1000
            self.browser_mute_ms = mute_ms
            self.cancel_ack.set()
            accepted = True
        return accepted

    def played_ms(self, epoch: int, server_estimate: int):
        if not self.browser_seen:
            return 0  # no browser evidence: do not retain potentially unheard speech
        # Confirmation is the START of a displayed frame with its audio output
        # graph enabled. This is an A/V playback proxy, not speaker capture.
        return min(server_estimate, self.positions.get(epoch, 0) * 1000 // SAMPLE_RATE)


class TaggedVideoSource(rtc.VideoSource):
    def __init__(self, width, height, ledger: PlaybackLedger, epoch):
        super().__init__(width, height)
        self.ledger, self.epoch = ledger, epoch

    def capture_frame(self, frame, **kwargs):
        current = self.epoch()
        epoch, position, valid = getattr(
            frame, "playback_position", (current, 0, False)
        )
        # AVSynchronizer may already have dequeued an old frame when cancellation
        # clears its queue. Drop that frame at the actual capture boundary too.
        if epoch != current:
            return
        ident = frame_id(epoch, position, valid)
        super().capture_frame(
            frame, metadata=rtc.FrameMetadata(frame_id=ident), **kwargs
        )
        self.ledger.capture(ident, epoch, position, valid)


def tag(frame, epoch: int, position: int, valid: bool):
    frame.playback_position = (epoch, position, valid)
    return frame
