import asyncio
from contextlib import contextmanager
import sys
from types import SimpleNamespace as Obj
from unittest.mock import AsyncMock

import numpy as np
import pytest

from realtime.config import Settings
from realtime.qwen_voice_agent import QwenVoiceAgent, TextPhrases, speech_text
from realtime.tests.test_voice_stream import Sink


@pytest.mark.parametrize("prioritized", [False, True])
def test_tts_stream_context_covers_lazy_generation_and_cancellation(
    monkeypatch, prioritized
):
    from realtime import qwen_server

    state = {"inside": False, "closed": False}

    @contextmanager
    def context(stream):
        state["inside"] = True
        try:
            yield
        finally:
            state["inside"] = False

    def generate(**kwargs):
        assert state["inside"] is prioritized
        try:
            yield b"first PCM chunk"
            pytest.fail("Cancellation should not generate another chunk")
        finally:
            assert state["inside"] is prioritized
            state["closed"] = True

    monkeypatch.setitem(sys.modules, "torch", Obj(cuda=Obj(stream=context)))
    monkeypatch.setattr(qwen_server, "speech_stream", object() if prioritized else None)
    monkeypatch.setattr(
        qwen_server, "model", Obj(generate_custom_voice_streaming=generate)
    )
    stream = qwen_server.speech_chunks(text="Synthetic reply")
    assert next(stream) == b"first PCM chunk"
    stream.close()
    assert state == {"inside": False, "closed": True}


class TTS:
    async def stream(self, text, voice, instructions):
        yield np.ones(2400, np.int16).tobytes()


def test_phrase_chunking_preserves_all_text_with_bounded_phrase_size():
    chunker = TextPhrases()
    text = (
        "Hello there, how are you doing today? "
        + "A long reply with no punctuation " * 20
    )
    pieces = []
    for letter in text:
        pieces.extend(chunker.push(letter))
    pieces.extend(chunker.push("", final=True))
    assert "".join("".join(pieces).split()) == "".join(text.split())
    assert max(map(len, pieces)) <= 180


def test_short_first_clause_streams_and_large_deltas_remain_bounded():
    chunker = TextPhrases()
    first = "Hello, this is a short greeting, "
    assert chunker.push(first) == [first.strip()]
    rest = "and now there is enough context to speak naturally."
    assert chunker.push(rest) == [rest]
    # A genuinely short complete response still finishes promptly.
    assert TextPhrases().push("Hello!", final=True) == ["Hello!"]
    text = "An entire reply in one delta " * 30 + "."
    pieces = TextPhrases().push(text, final=True)
    assert max(map(len, pieces)) <= 180
    assert "".join("".join(pieces).split()) == "".join(text.split())


def test_queued_speech_coalesces_without_waiting_or_losing_text_at_the_bound():
    async def run():
        queue = asyncio.Queue()
        first = "Speak this available phrase now."
        queue.put_nowait(first)
        stream = speech_text(queue)
        assert await asyncio.wait_for(anext(stream), 0.1) == first
        remaining = ["a" * 70, "b" * 70, "c" * 100, "d" * 100]
        for text in remaining:
            queue.put_nowait(text)
        queue.put_nowait(None)
        chunks = [text async for text in stream]
        assert chunks == [remaining[0] + " " + remaining[1], *remaining[2:]]
        assert queue.empty()
        assert max(map(len, chunks)) <= 180

    asyncio.run(run())


def test_text_done_does_not_finish_pcm_before_local_tts_finishes():
    async def run():
        sink = Sink()
        agent = QwenVoiceAgent(
            Settings(avatar_catalog="research", _env_file=None, tts_provider="qwen"),
            {"voice": "aiden"},
            sink,
            tts=TTS(),
        )
        await agent.handle(Obj(type="response.created", response=Obj(id="r1")))
        await agent.handle(
            Obj(
                type="response.output_text.delta",
                response_id="r1",
                item_id="i1",
                delta="Hello there, how are you doing today?",
            )
        )
        await agent.handle(Obj(type="response.output_text.done", response_id="r1"))
        assert sink.buffer.receiving
        await agent.tts_task
        assert not sink.buffer.receiving and len(sink.buffer.samples) > 0
        assert agent.first_audio_at is not None
        diagnostics = agent.diagnostics()
        assert diagnostics["tts_requests"] == 1
        assert diagnostics["first_tts_characters"] == len(
            "Hello there, how are you doing today?"
        )
        assert all(value >= 0 for value in diagnostics["text_speech_timing"].values())
        await agent.handle(Obj(type="input_audio_buffer.speech_started"))
        assert agent.diagnostics()["tts_requests"] == 0
        assert all(
            value is None
            for value in agent.diagnostics()["text_speech_timing"].values()
        )

    asyncio.run(run())


def test_qwen_interruption_deletes_unheard_text_instead_of_audio_truncation():
    async def run():
        sink = Sink()
        sink.server_played_ms = lambda: 320
        sink.interrupted_played_ms = AsyncMock(return_value=200)
        agent = QwenVoiceAgent(
            Settings(avatar_catalog="research", _env_file=None, tts_provider="qwen"),
            {"voice": "aiden"},
            sink,
            tts=TTS(),
        )
        item = Obj(delete=AsyncMock(), create=AsyncMock(), truncate=AsyncMock())
        agent.connection = Obj(
            response=Obj(cancel=AsyncMock()), conversation=Obj(item=item)
        )
        agent.response_id, agent.item_id, agent.epoch, agent.generating = (
            "r1",
            "i1",
            1,
            True,
        )
        agent.spoken_spans = [(4800, "Heard phrase."), (48000, "Unheard phrase.")]
        await agent.interrupt()
        item.delete.assert_awaited_once_with(item_id="i1")
        item.truncate.assert_not_awaited()
        assert (
            item.create.await_args.kwargs["item"]["content"][0]["text"]
            == "Heard phrase."
        )
        assert not sink.buffer.active and agent.response_id is None

    asyncio.run(run())
