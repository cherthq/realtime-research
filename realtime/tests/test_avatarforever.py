from pathlib import Path

import pytest

from realtime.avatarforever import build_command, valid_frame_count


@pytest.mark.parametrize("samples", [1, 48_000, 48_001, 671_999, 28_800_000])
def test_frame_budget_covers_audio_with_only_native_final_padding(samples):
    frames = valid_frame_count(samples, 48000)
    assert (frames - 1) % 8 == 0
    assert frames / 25 >= samples / 48000
    assert (frames - 8) / 25 < samples / 48000


def test_native_command_keeps_reference_and_geometry_explicit():
    command = build_command(
        "gpu-python",
        Path("portrait.png"),
        Path("reply.wav"),
        Path("weights"),
        Path("out.mp4"),
        161,
    )
    assert command[command.index("--width") + 1] == "640"
    assert command[command.index("--height") + 1] == "480"
    assert (
        command[command.index("--first-frame-condition-image-path") + 1]
        == "portrait.png"
    )
    assert "--ar-history-feature-cache" in command
    with pytest.raises(ValueError):
        build_command("python", Path("i"), Path("a"), Path("w"), Path("o"), 160)
