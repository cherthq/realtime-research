import numpy as np
import pytest

from realtime.contracts import AudioChunk
from realtime.listening import ListeningControl


def chunk(value=10, *, epoch=1, valid=8, lookahead=0):
    return AudioChunk(
        "reply",
        epoch,
        0,
        np.full(8, value, np.int16),
        valid,
        np.full(8, lookahead, np.int16),
    )


def advance(control, audio):
    cue = control.preview(audio)
    control.commit(cue)
    return cue.profile


def test_quiet_onset_speech_and_pause_transitions():
    control = ListeningControl("onset_boost")
    sequence = [0, 10, 10, 10, 10, 0, 10, 10, 10]
    assert [advance(control, chunk(v)) for v in sequence] == [
        "quiet",
        "onset",
        "onset",
        "speaking",
        "speaking",
        "quiet",
        "onset",
        "onset",
        "speaking",
    ]


def test_cancelled_preview_does_not_consume_onset_and_new_epoch_restarts_it():
    control = ListeningControl("onset_boost")
    assert advance(control, chunk()) == "onset"
    cancelled = control.preview(chunk())
    assert cancelled.speech_chunks == 2
    assert control.preview(chunk()) == cancelled
    assert control.current.speech_chunks == 1
    assert advance(control, chunk()) == "onset"
    assert advance(control, chunk()) == "speaking"
    assert advance(control, chunk(epoch=2)) == "onset"
    assert control.current.speech_chunks == 1
    control.reset()
    assert control.current is None
    assert advance(control, chunk()) == "onset"


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, "quiet"),
        (2, "quiet"),
        (-2, "quiet"),
        (3, "onset"),
        (-3, "onset"),
        (-32768, "onset"),
    ],
)
def test_exact_pcm_threshold_and_signed_overflow(value, expected):
    assert ListeningControl("onset_boost").preview(chunk(value)).profile == expected


def test_padding_and_lookahead_cannot_trigger_speaking():
    control = ListeningControl("onset_boost")
    assert control.preview(chunk(0, lookahead=100)).profile == "quiet"
    assert control.preview(chunk(100, valid=0)).profile == "quiet"


def test_default_off_and_invalid_profile():
    control = ListeningControl()
    assert control.profiles == ("default",)
    assert advance(control, chunk()) == "default"
    assert advance(control, chunk(0)) == "default"
    with pytest.raises(ValueError, match="Unknown"):
        ListeningControl("typo")
