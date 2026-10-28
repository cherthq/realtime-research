import asyncio

import pytest

from realtime.contracts import FRAME_SAMPLES
from realtime.playback import PlaybackLedger, frame_id


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
