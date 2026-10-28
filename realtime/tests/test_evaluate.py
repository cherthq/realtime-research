import asyncio
import json
import math
import shutil
import wave
import os
import subprocess
import venv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from realtime.avatarforever import file_hash, load_frozen_case
from realtime.contracts import FRAME_SAMPLES, HEIGHT, WIDTH
from realtime.evaluate import ReplayFailure, VideoWriter, replay, run
from realtime.streaming import GeneratedBatch


class Renderer:
    def __init__(self):
        self.inputs = []

    async def generate(self, chunk):
        self.inputs.append(chunk)
        await asyncio.sleep(0.001)
        # Repeating output is deliberate: the report must not call all frames unique.
        frames = np.zeros(
            (math.ceil(chunk.valid_samples / FRAME_SAMPLES), HEIGHT, WIDTH, 3),
            dtype=np.uint8,
        )
        return GeneratedBatch(chunk, frames, 0.001)


def test_paced_replay_waits_for_capture_windows_and_preserves_partial_audio():
    async def run():
        renderer = Renderer()
        samples = np.arange(48_001, dtype=np.int16)
        result = await replay(
            renderer, samples, mode="paced", response_id="turn", epoch=1
        )
        assert result["first_decoded_seconds"] >= 0.98
        assert result["valid_samples"] == len(samples)
        assert result["generated_frames"] == 26
        assert (
            result["distinct_frame_hashes"] == 1
            and result["exact_duplicate_frames"] == 25
        )
        reconstructed = np.concatenate(
            [chunk.pcm[: chunk.valid_samples] for chunk in renderer.inputs]
        )
        assert np.array_equal(samples, reconstructed)
        assert renderer.inputs[-1].valid_samples == 1
        for event in result["events"]:
            assert (
                event["sample_start"] + event["valid_samples"]
                <= event["available_samples_at_dispatch"]
            )

    asyncio.run(run())


def test_throughput_and_paced_replay_pass_identical_audio_to_renderer():
    async def run():
        samples = np.arange(48_001, dtype=np.int16)
        paced, throughput = Renderer(), Renderer()
        a = await replay(paced, samples, mode="paced", response_id="r", epoch=1)
        b = await replay(
            throughput, samples, mode="throughput", response_id="r", epoch=1
        )
        assert len(paced.inputs) == len(throughput.inputs)
        assert all(
            np.array_equal(x.pcm, y.pcm)
            for x, y in zip(paced.inputs, throughput.inputs)
        )
        assert a["mode"] != b["mode"] and a["valid_samples"] == b["valid_samples"]
        assert b["replay_wall_seconds"] < a["replay_wall_seconds"]

    asyncio.run(run())


def test_failed_replay_retains_partial_timing_and_frame_evidence():
    class FailingRenderer(Renderer):
        async def generate(self, chunk):
            if self.inputs:
                raise RuntimeError("simulated generation failure")
            return await super().generate(chunk)

    async def run():
        with pytest.raises(ReplayFailure) as failure:
            await replay(
                FailingRenderer(),
                np.zeros(48001, dtype=np.int16),
                mode="throughput",
                response_id="failure",
                epoch=1,
            )
        record = failure.value.metrics
        assert record["completed"] is False
        assert record["generated_frames"] == 25 and record["valid_samples"] == 48000
        assert record["distinct_frame_hashes"] == 1 and len(record["events"]) == 1
        assert failure.value.cause_type == "RuntimeError"

    asyncio.run(run())


def make_pack(tmp_path):
    image, audio = tmp_path / "portrait.png", tmp_path / "audio.wav"
    image.write_bytes(b"reference fixture")
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 48000, 1001, "NONE", "not compressed"))
        stream.writeframes(np.zeros(1001, dtype=np.int16).tobytes())
    manifest = {
        "status": "frozen",
        "suite_id": "facetime-avatar-voices-v2",
        "portraits": {
            "ref01": {
                "path": image.name,
                "sha256": file_hash(image),
                "audio_ids": ["reply"],
                "voice_selection": {"voice": "test"},
            }
        },
        "audio": {
            "reply": {
                "path": audio.name,
                "sha256": file_hash(audio),
                "scenario_id": "s01_greeting",
                "samples": 1001,
            }
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_frozen_case_rejects_tampering_and_sample_mismatch(tmp_path):
    manifest = make_pack(tmp_path)
    assert load_frozen_case(tmp_path, "ref01", "s01_greeting")["samples"] == 1001
    manifest["audio"]["reply"]["samples"] = 1000
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="sample count"):
        load_frozen_case(tmp_path, "ref01", "s01_greeting")
    (tmp_path / "audio.wav").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_frozen_case(tmp_path, "ref01", "s01_greeting")


def test_frozen_case_cannot_escape_pack(tmp_path):
    manifest = make_pack(tmp_path)
    manifest["portraits"]["ref01"]["path"] = "../outside.png"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="leaves"):
        load_frozen_case(tmp_path, "ref01", "s01_greeting")


@pytest.mark.skipif(os.name != "posix", reason="Linux venv Python is a symlink")
def test_evaluation_launch_keeps_the_selected_virtual_environment(tmp_path, capsys):
    make_pack(tmp_path)
    environment = tmp_path / "gpu-env"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    args = SimpleNamespace(
        pack=tmp_path,
        portrait="ref01",
        scenario="s01_greeting",
        duration_seconds=None,
        gpu_python=environment / "bin/python",
        weights=tmp_path / "weights",
        dry_run=True,
        mode="paced",
        repeat=1,
        save_video=False,
        compile_blocks=False,
        load_timeout=300,
    )
    asyncio.run(run(args))
    command = json.loads(capsys.readouterr().out)["command"]
    prefix = subprocess.check_output(
        [command[0], "-c", "import sys; print(sys.prefix)"], text=True
    ).strip()
    assert Path(prefix) == environment


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg/FFprobe required",
)
def test_review_video_keeps_native_geometry_and_partial_audio(tmp_path):
    async def run():
        samples = np.arange(49_001, dtype=np.int16)
        audio = tmp_path / "input.wav"
        with wave.open(str(audio), "wb") as stream:
            stream.setparams((1, 2, 48000, len(samples), "NONE", "not compressed"))
            stream.writeframes(samples.tobytes())
        writer = VideoWriter(tmp_path, audio)
        try:
            await writer.start()
            metrics = await replay(
                Renderer(),
                samples,
                mode="throughput",
                response_id="review",
                epoch=1,
                on_batch=writer.append,
            )
            result = await writer.finish(len(samples))
            video = next(
                s for s in result["probe"]["streams"] if s["codec_type"] == "video"
            )
            assert (video["width"], video["height"]) == (640, 480)
            assert int(video["nb_frames"]) == metrics["generated_frames"] == 26
            assert len(result["sha256"]) == 64
        finally:
            await writer.close()

    asyncio.run(run())
