import asyncio
import json
import shutil
import subprocess
import time
from types import SimpleNamespace
import wave

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from livekit import rtc

from avatar_config_runtime.call_logs import CallLogs
from avatar_config_runtime.recording import MediaWriter, CallRecorder
from realtime.transcripts import TranscriptLedger
from avatar_config_runtime.tests.test_runtime import TOKEN

pytest_plugins = ["avatar_config_runtime.tests.test_runtime"]

IDENT = "a" * 24


def test_private_ledger_pagination_crash_recovery_and_signed_downloads(tmp_path):
    logs = CallLogs(tmp_path, TOKEN)
    folder = logs.begin(IDENT, "facetime", "version")
    logs.transcript(IDENT, [dict(speaker="user", seconds=1, text="Hello")])
    (folder / "audio.wav").write_bytes(b"fixture")
    assert "audio" not in logs.get(IDENT)["files"]
    logs.update(IDENT, "complete", 2)
    ticket = logs.ticket(IDENT, "audio")
    assert logs.verify(IDENT, "audio", ticket["expires"], ticket["signature"])
    assert not logs.verify(IDENT, "video", ticket["expires"], ticket["signature"])
    assert not logs.verify("b" * 24, "audio", ticket["expires"], ticket["signature"])
    assert not logs.verify(
        IDENT, "audio", str(int(time.time()) - 1), ticket["signature"]
    )
    assert not logs.verify(IDENT, "audio", ticket["expires"], "x")
    with pytest.raises(ValueError):
        logs.get("../secrets")
    with pytest.raises(KeyError):
        logs.file(IDENT, "../../secret")
    pending = "b" * 24
    logs.begin(pending, "web", None)
    reopened = CallLogs(tmp_path, TOKEN)
    assert reopened.get(pending)["status"] == "interrupted"
    assert reopened.get(IDENT)["source"] == "facetime"
    assert reopened.get(IDENT)["transcript"][0]["text"] == "Hello"
    assert reopened.listing()["calls"][0]["id"] == pending
    assert reopened.listing(2)["calls"] == []


def test_call_endpoints_authorization_download_range_and_traversal(runtime, tmp_path):
    async def run():
        runtime.logs = CallLogs(tmp_path, TOKEN)
        folder = runtime.logs.begin(IDENT, "web", None)
        (folder / "audio.wav").write_bytes(b"0123456789")
        runtime.logs.update(IDENT, "complete", 1)
        app = FastAPI()
        runtime.install_routes(app)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost"
        ) as client:
            path = "/api/avatar-config"
            assert (await client.get(path + "?op=calls")).status_code == 401
            headers = {"Authorization": "Bearer " + TOKEN}
            assert (await client.get(path + "?op=calls", headers=headers)).json()[
                "calls"
            ][0]["id"] == IDENT
            assert (
                await client.get(path + "?op=call&id=../x", headers=headers)
            ).status_code == 400
            ticket = (
                await client.get(
                    path,
                    params={"op": "ticket", "id": IDENT, "file": "audio"},
                    headers=headers,
                )
            ).json()
            response = await client.get(
                path,
                params={"op": "download", **ticket},
                headers={"Range": "bytes=2-5"},
            )
            assert response.status_code == 206 and response.content == b"2345"
            assert "attachment" in response.headers["content-disposition"]
            assert response.headers["cache-control"] == "private, no-store"
            ticket["file"] = "video"
            assert (
                await client.get(path, params={"op": "download", **ticket})
            ).status_code == 403

    asyncio.run(run())


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
def test_real_media_encoding_both_speakers_and_video(tmp_path):
    writer = MediaWriter(tmp_path)
    writer.begin()
    frame = rtc.VideoFrame(
        64, 48, rtc.VideoBufferType.RGB24, bytes([120, 30, 50]) * 64 * 48
    )
    chunk = (
        (np.sin(np.arange(960) * 2 * np.pi * 440 / 48000) * 3000)
        .astype(np.int16)
        .tobytes()
    )
    for _ in range(20):
        writer.video("agent", frame)
        writer.video("caller", frame)
        writer.pcm("caller", chunk)
        writer.pcm("agent", chunk)
        time.sleep(0.02)
    duration, ready = writer.finish()
    assert ready and 0.3 < duration < 2
    for name in ("caller.wav", "agent.wav", "audio.wav"):
        with wave.open(str(tmp_path / name)) as stream:
            assert stream.getnframes() > 12000
            assert any(stream.readframes(1000))
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-of",
                "json",
                str(tmp_path / "video.mp4"),
            ]
        )
    )
    assert {s["codec_type"] for s in probe["streams"]} == {"audio", "video"}
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (960, 360)


def test_transcript_archive_survives_rolling_window_and_close(tmp_path):
    async def run():
        worker = SimpleNamespace(id=IDENT, voice=None, transcripts=TranscriptLedger())
        recorder = CallRecorder(worker, CallLogs(tmp_path, TOKEN), "web")
        for index in range(205):
            worker.transcripts.user(str(index), f"Words {index}", final=True)
        worker.transcripts.close()
        assert len(recorder.rows) == 205
        assert recorder.rows["user:0"]["text"] == "Words 0"
        await recorder.close()
        saved = json.loads((tmp_path / IDENT / "transcript.json").read_text())
        assert len(saved) == 205

    asyncio.run(run())


def test_facetime_transcript_uses_sent_audio_without_browser_receipts(tmp_path):
    async def run():
        worker = SimpleNamespace(
            id=IDENT,
            voice=None,
            transcripts=TranscriptLedger(),
            buffer=SimpleNamespace(epoch=1),
            server_played_ms=lambda: 50,
            response_active=True,
            published_response_samples=2400,
            audio=SimpleNamespace(queued_duration=0),
        )
        recorder = CallRecorder(worker, CallLogs(tmp_path, TOKEN), "facetime")
        worker.transcripts.begin("reply", 1)
        worker.transcripts.cue("reply", 1000, "Heard words")
        worker.transcripts.cue("reply", 48000, "Heard words plus unplayed words")
        assert not worker.transcripts.snapshot()["utterances"][0]["text"]
        recorder.capture_transcript()
        assert recorder.rows["assistant:reply"]["text"] == "Heard words"
        worker.transcripts.close()
        assert recorder.rows["assistant:reply"]["text"] == "Heard words"
        await recorder.close()

    asyncio.run(run())
