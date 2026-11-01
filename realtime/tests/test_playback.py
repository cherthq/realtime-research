import asyncio

import pytest

from realtime.contracts import FRAME_SAMPLES
from realtime.playback import PlaybackLedger, TaggedVideoSource, frame_id, tag
from livekit import rtc


def test_feedback_uses_only_captured_frames_with_audio_playing_and_never_advances_from_future():
    ledger = PlaybackLedger()
    ident = frame_id(5, 12 * FRAME_SAMPLES, True)
    assert ledger.played_ms(5, 1000) == 0
    assert not ledger.feedback(ident, True)
    ledger.capture(ident, 5, 12 * FRAME_SAMPLES, True)
    assert ledger.feedback(ident, False)
    assert ledger.played_ms(5, 1000) == 0  # blocked/muted audio cannot count as played
    assert ledger.feedback(ident, True)
    assert ledger.played_ms(5, 1000) == 480
    assert ledger.played_ms(5, 200) == 200  # never exceeds the publication bound
    assert ledger.played_ms(6, 1000) == 0  # a new turn has no prior playback
    earlier = frame_id(5, 6 * FRAME_SAMPLES, True)
    ledger.capture(earlier, 5, 6 * FRAME_SAMPLES, True)
    ledger.feedback(earlier, True)
    assert ledger.played_ms(5, 1000) == 480


def test_cancel_receipt_is_scoped_to_current_epoch_and_bounded():
    async def run():
        ledger = PlaybackLedger()
        ledger.cancelled(6)
        assert not ledger.feedback(None, False, 5, 0.1)
        assert not ledger.cancel_ack.is_set()
        assert ledger.feedback(None, False, 6, 0.2)
        assert ledger.cancel_ack.is_set() and ledger.browser_mute_ms == 0.2
        first = ledger.cancel_roundtrip_ms
        assert not ledger.feedback(None, False, 6, 500)
        assert ledger.cancel_roundtrip_ms == first
        ledger.cancelled(7)
        assert not ledger.cancel_ack.is_set()
        for index in range(5000):
            ident = frame_id(7, index * FRAME_SAMPLES, True)
            ledger.capture(ident, 7, index * FRAME_SAMPLES, True)
        assert len(ledger.sent) == 4096
        assert not ledger.feedback(frame_id(7, 0, True), True)

    asyncio.run(run())


def test_frame_ids_do_not_wrap_or_accept_unaligned_positions():
    assert frame_id(4, 0, True) == 5 * 65536 + 1
    assert frame_id(4, 0, False) == 5 * 65536
    with pytest.raises(ValueError):
        frame_id(65536, 0, True)
    with pytest.raises(ValueError):
        frame_id(1, 1, True)


def test_last_speech_receipt_requires_matching_idle_epoch_and_audio_output():
    ledger = PlaybackLedger()
    final, idle = frame_id(3, 48000, True), frame_id(3, 0, False)
    ledger.capture(final, 3, 48000, True)
    ledger.capture(idle, 3, 0, False)
    ledger.feedback(idle, False, completed_frame_id=final)
    assert ledger.played_ms(3, 2000) == 0
    ledger.feedback(idle, True, completed_frame_id=final)
    assert ledger.played_ms(3, 2000) == 1000
    other = frame_id(4, 0, False)
    ledger.capture(other, 4, 0, False)
    ledger.feedback(other, True, completed_frame_id=final)
    assert ledger.played_ms(4, 2000) == 0


def test_stale_speaking_pose_is_dropped_but_listening_can_continue(
    monkeypatch,
):
    frames = []
    monkeypatch.setattr(
        rtc.VideoSource, "capture_frame", lambda _self, frame, **kw: frames.append(kw)
    )
    ledger = PlaybackLedger()
    source = TaggedVideoSource(2, 2, ledger, lambda: 3, preserve_stale=True)
    frame = rtc.VideoFrame(2, 2, rtc.VideoBufferType.RGB24, bytes(12))
    source.capture_frame(tag(frame, 2, 1920, True))
    assert not frames
    source.capture_frame(tag(frame, 2, 0, False))
    assert len(frames) == 1
    ident = frames[0]["metadata"].frame_id
    assert ident == frame_id(3, 0, False)
    assert ledger.feedback(ident, True)
    assert ledger.played_ms(2, 1000) == ledger.played_ms(3, 1000) == 0


def test_rate_cap_preserves_native_audio_positions_across_turns(monkeypatch):
    captured = []
    monkeypatch.setattr(
        rtc.VideoSource,
        "capture_frame",
        lambda _self, frame, **kw: captured.append(kw["metadata"].frame_id),
    )
    ledger, epoch = PlaybackLedger(), [1]
    source = TaggedVideoSource(2, 2, ledger, lambda: epoch[0], preserve_stale=True)
    frame = rtc.VideoFrame(2, 2, rtc.VideoBufferType.RGB24, bytes(12))
    native = []
    due = []
    # Change response mid-second: the output cap must not reset with the turn.
    for index in range(250):
        epoch[0] = 1 if index < 103 else 2
        position = index * FRAME_SAMPLES
        native.append(frame_id(epoch[0], position, True))
        delay = source.presentation_delay_seconds
        assert 0 <= delay < 1 / 25
        before = len(captured)
        source.capture_frame(tag(frame, epoch[0], position, True))
        if len(captured) > before:
            due.append(index / 25 + delay)
    assert len(captured) == 240 and source.rate_limited_frames == 10
    assert captured[0] == native[0]
    assert due == pytest.approx([index / 24 for index in range(240)], abs=1e-12)
    assert captured == [ident for i, ident in enumerate(native) if (i + 1) % 25]
    # Omitted video cannot be credited as displayed. Kept IDs retain their
    # original PCM positions rather than becoming a slower 24 FPS audio clock.
    assert not ledger.feedback(native[124], True)
    assert ledger.feedback(native[125], True)
    assert ledger.played_ms(2, 10000) == 5000
