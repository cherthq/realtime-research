import asyncio
import importlib
import time
from types import SimpleNamespace as Obj
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
from livekit import rtc
from pydantic import SecretStr

from realtime.camera import CameraInput
from realtime.config import Settings
from realtime.transcripts import TranscriptLedger
from realtime.voice_agent import VoiceAgent
from realtime.tests.test_voice_stream import Sink


def test_captions_follow_playback_cancel_and_bound_history():
    ledger = TranscriptLedger()
    ledger.user("u1")
    ledger.begin("r1", 3)
    ledger.user("u1", "What is this?", final=True)
    ledger.user("u1", "late duplicate", append=True)
    ledger.cue("r1", 4800, "A")
    ledger.cue("r1", 9600, "A tomato.")
    ledger.finish("r1", 10000)
    assert ledger.snapshot()["utterances"][1]["text"] == ""
    ledger.advance(3, 5000)
    ledger.interrupt(3, 5000)
    ledger.advance(3, 20000, complete=True)
    ledger.cue("r1", 0, "unheard future text")
    assert ledger.snapshot()["utterances"] == [
        dict(
            id="user:u1",
            speaker="user",
            text="What is this?",
            state="complete",
            order=0,
        ),
        dict(
            id="assistant:r1",
            speaker="assistant",
            text="A",
            state="interrupted",
            order=2,
        ),
    ]
    for i in range(250):
        ledger.user(f"next-{i}", "Hello", final=True)
    assert len(ledger.snapshot()["utterances"]) == 200
    ledger.close()
    assert not ledger.snapshot()["utterances"] and ledger.snapshot()["ended"]


def test_vad_fragments_merge_live_and_late_finals_keep_their_speech_order():
    ledger = TranscriptLedger()
    ledger.speech_started("first", 0)
    ledger.user("first", "Could you tell me", append=True)
    ledger.user("first")  # commit after the first live delta must not erase it
    ledger.speech_stopped("first", 896)
    ledger.begin("unheard", 1)
    ledger.speech_started("continuation", 852)  # Real provider VAD prefix overlap.
    ledger.user("continuation", "about tomatoes?", final=True)
    ledger.user("first", "Could you tell me", final=True)  # final arrives out of order
    rows = [row for row in ledger.snapshot()["utterances"] if row["speaker"] == "user"]
    assert len(rows) == 1
    assert rows[0]["text"] == "Could you tell me about tomatoes?"
    assert rows[0]["state"] == "complete"
    ledger.cue("unheard", 1, "Yes.")
    ledger.advance(1, 100)
    ledger.speech_stopped("continuation", 2000)
    ledger.speech_started("new-turn", 2300)
    ledger.user("new-turn", "Wait.", final=True)
    assert (
        len(
            [row for row in ledger.snapshot()["utterances"] if row["speaker"] == "user"]
        )
        == 2
    )


def test_long_pause_starts_a_new_bubble_and_final_duplicates_are_ignored():
    ledger = TranscriptLedger()
    ledger.speech_started("a", 0)
    ledger.user("a", "Hi.", final=True)
    ledger.speech_stopped("a", 1000)
    ledger.speech_started("b", 6000)
    ledger.user("b", "Hello.", final=True)
    ledger.user("a", "duplicate", append=True)
    assert [row["text"] for row in ledger.snapshot()["utterances"]] == ["Hi.", "Hello."]


def test_english_defaults_use_model_specific_fields_without_restoring_token_cap():
    from realtime.voice_agent import MAX_RESPONSE_TOKENS

    for model, expected in [
        ("gpt-live-transcribe", {"languages": ["en"], "delay": "low"}),
        ("gpt-4o-transcribe", {"language": "en"}),
    ]:
        agent = VoiceAgent(
            Settings(
                avatar_catalog="research",
                _env_file=None,
                input_transcription_model=model,
            ),
            {"voice": "cedar"},
            Sink(),
        )
        assert agent.input_options()["transcription"] == {"model": model, **expected}
        assert "one or two short sentences" in agent.instructions()
        assert "Speak English by default" in agent.instructions()
    assert MAX_RESPONSE_TOKENS == "inf"


def test_late_final_transcript_is_not_lost_after_audio_finishes():
    ledger = TranscriptLedger()
    ledger.begin("r", 1)
    ledger.finish("r", 48000)
    ledger.advance(1, 48000, complete=True)
    ledger.cue("r", 48000, "Completed speech.")
    assert ledger.snapshot()["utterances"][0]["text"] == "Completed speech."
    assert ledger.snapshot()["utterances"][0]["state"] == "complete"


def test_camera_age_disable_and_size():
    async def run():
        camera = CameraInput()
        camera.frame = rtc.VideoFrame(
            1280, 720, rtc.VideoBufferType.RGB24, bytes(1280 * 720 * 3)
        )
        camera.received_at = time.monotonic()
        image = await camera.snapshot()
        assert image.startswith("data:image/jpeg;base64,")
        import base64
        import io
        from PIL import Image

        decoded = Image.open(io.BytesIO(base64.b64decode(image.split(",")[1])))
        assert decoded.size == (640, 360)
        camera.received_at -= 3
        assert await camera.snapshot() is None
        camera.clear()
        assert await camera.snapshot() is None

    asyncio.run(run())


def test_camera_arrives_before_response_and_interruption_cancels_snapshot():
    async def run():
        sink = Sink()
        sink.transcripts = TranscriptLedger()
        sink.camera = Obj(
            snapshot=AsyncMock(return_value="data:image/jpeg;base64,fixture")
        )
        agent = VoiceAgent(
            Settings(avatar_catalog="research", _env_file=None),
            {"voice": "cedar"},
            sink,
        )
        events = []

        async def create_item(**kwargs):
            events.append(("image", kwargs))

        async def respond(**kwargs):
            events.append(("response", kwargs))

        agent.connection = Obj(
            conversation=Obj(
                item=Obj(create=AsyncMock(side_effect=create_item), delete=AsyncMock())
            ),
            response=Obj(create=AsyncMock(side_effect=respond), cancel=AsyncMock()),
        )
        assert agent.input_options()["turn_detection"]["create_response"] is False
        for i in range(3):
            await agent.handle(
                Obj(type="input_audio_buffer.committed", item_id=f"u{i}")
            )
            await agent.turn_task
            await agent.handle(Obj(type="response.created", response=Obj(id=f"r{i}")))
            await agent.handle(
                Obj(type="response.done", response=Obj(id=f"r{i}", status="completed"))
            )
        assert [x[0] for x in events] == ["image", "response"] * 3
        assert len(agent.image_items) == 2
        assert agent.connection.conversation.item.delete.await_count == 1
        sink.camera.snapshot.return_value = None
        await agent._respond_with_camera(agent.turn_generation)
        assert not agent.image_items
        assert "No fresh camera" in events[-1][1]["response"]["instructions"]
        await agent.handle(Obj(type="response.created", response=Obj(id="r3")))
        await agent.handle(
            Obj(type="response.done", response=Obj(id="r3", status="completed"))
        )
        waiting = asyncio.Event()

        async def delayed():
            waiting.set()
            await asyncio.sleep(10)

        sink.camera.snapshot.side_effect = delayed
        await agent.handle(Obj(type="input_audio_buffer.committed", item_id="u4"))
        await waiting.wait()
        count = len(events)
        await agent.handle(Obj(type="input_audio_buffer.speech_started"))
        await asyncio.gather(agent.turn_task, return_exceptions=True)
        assert len(events) == count
        await agent.close()

    asyncio.run(run())


def test_transcript_endpoints_require_session_key_and_never_enter_metrics(monkeypatch):
    module = importlib.import_module("realtime.app")
    app = module.create_app(Settings(avatar_catalog="research", _env_file=None))
    ledger = TranscriptLedger()
    ledger.user("u", "private fixture", final=True)
    worker = Obj(transcripts=ledger)
    session = Obj(worker=worker, ended=False)

    def own(sid, key):
        if key != "fixture-key":
            from fastapi import HTTPException

            raise HTTPException(403, "Unknown session")
        return session

    monkeypatch.setattr(app.state.manager, "get", own)
    with TestClient(app) as client:
        assert client.get("/api/sessions/s/transcripts").status_code == 403
        response = client.get(
            "/api/sessions/s/transcripts", headers={"X-Session-Key": "fixture-key"}
        )
        assert response.json()["utterances"][0]["text"] == "private fixture"
        ledger.close()
        assert (
            client.get(
                "/api/sessions/s/transcripts", headers={"X-Session-Key": "fixture-key"}
            ).json()["utterances"]
            == []
        )


def test_elevenlabs_missing_config_fails_before_session_start():
    module = importlib.import_module("realtime.app")
    config = Settings(
        avatar_catalog="research",
        _env_file=None,
        tts_provider="elevenlabs",
        openai_api_key=SecretStr("fixture"),
    )
    with TestClient(module.create_app(config)) as client:
        response = client.post(
            "/api/sessions",
            json={"avatar_id": "ref01", "model_id": "mock", "mode": "voice_diagnostic"},
        )
        assert response.status_code == 409
        assert "ELEVENLABS_VOICE_ID" in response.json()["detail"]


def test_exhausted_speech_balance_blocks_call_before_gpu_lease(monkeypatch, tmp_path):
    from realtime.elevenlabs_client import ElevenLabsSpeechError

    module = importlib.import_module("realtime.app")
    config = Settings(
        avatar_catalog="research",
        _env_file=None,
        state_dir=tmp_path,
        tts_provider="elevenlabs",
        elevenlabs_voice_id="fixtureVoice",
        elevenlabs_api_key=SecretStr("fixture"),
        openai_api_key=SecretStr("fixture"),
    )

    async def exhausted(_):
        raise ElevenLabsSpeechError("quota_exceeded")

    monkeypatch.setattr(module, "ensure_speech_credits", exhausted)
    app = module.create_app(config)
    create = AsyncMock(side_effect=AssertionError("Must not lease GPU"))
    monkeypatch.setattr(app.state.manager, "create", create)
    with TestClient(app) as client:
        response = client.post(
            "/api/sessions",
            json={"avatar_id": "ref01", "model_id": "mock", "mode": "voice_diagnostic"},
        )
    assert response.status_code == 409
    assert "no speech credits remaining" in response.json()["detail"]
    create.assert_not_awaited()


def test_no_audio_marks_agent_error_and_clears_pending_speech():
    from realtime.text_voice_agent import TextVoiceAgent
    from realtime.elevenlabs_client import ElevenLabsSpeechError

    class EmptyTTS:
        async def stream_response(self, phrases):
            async for _ in phrases:
                pass
            raise ElevenLabsSpeechError("no_audio")
            yield  # async generator shape

    async def run():
        sink = Sink()
        agent = TextVoiceAgent(
            Settings(
                avatar_catalog="research", _env_file=None, tts_provider="elevenlabs"
            ),
            {"voice": "fixtureVoice"},
            sink,
            tts=EmptyTTS(),
        )
        await agent.handle(Obj(type="response.created", response=Obj(id="r1")))
        await agent.handle(
            Obj(
                type="response.output_text.delta",
                response_id="r1",
                item_id="i1",
                delta="Hello, how are you doing?",
            )
        )
        await agent.handle(Obj(type="response.output_text.done", response_id="r1"))
        await agent.tts_task
        await asyncio.gather(*agent.tasks)
        assert sink.status == "error"
        assert "no speech audio" in sink.error
        assert not sink.buffer.active
        assert agent.diagnostics()["tts_error_code"] == "no_audio"

    asyncio.run(run())
