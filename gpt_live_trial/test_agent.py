"""Regression coverage for the continuous Live audio and paired-video boundary."""

import asyncio
import json
from unittest.mock import AsyncMock, patch


import numpy as np
from realtime.contracts import ResponseBuffer
from realtime.config import Settings
from realtime.continuous import ContinuousTimeline
from realtime.playback import frame_id
from realtime.transcripts import TranscriptLedger
from gpt_live_trial.agent import LiveAgent
from gpt_live_trial.profile import CHERT_PROFILE


class Sink:
    def __init__(self):
        self.buffer = ResponseBuffer()
        self.transcripts = TranscriptLedger()
        self.cuts = 0
        self.begins = 0

    @property
    def response_active(self):
        return self.buffer.active

    async def begin_voice_response(self, name):
        self.begins += 1
        return self.buffer.begin(name)

    async def cancel_media(self, count=True):
        self.buffer.cancel()
        self.cuts += count

    def estimated_played_ms(self):
        return 0


def block(loud, ms=100):
    return np.full(24 * ms, 1000 if loud else 0, dtype="<i2").tobytes()


def test_each_new_voice_session_starts_with_the_chert_profile():
    class Socket:
        def __init__(self):
            self.sent = []
            self.events = asyncio.Queue()

        async def send(self, message):
            event = json.loads(message)
            self.sent.append(event)
            if event["type"] == "session.start":
                await self.events.put(json.dumps({"type": "session.started", "session": event["session"]}))
            elif event["type"] == "session.close":
                await self.events.put('{"type":"session.closed"}')

        async def recv(self):
            return await self.events.get()

        def __aiter__(self):
            return self

        __anext__ = recv

        async def close(self):
            pass

    async def scenario():
        settings = Settings(_env_file=None, openai_api_key="test-only")
        for _ in range(2):
            socket = Socket()
            agent = LiveAgent(settings, Sink())
            with patch("gpt_live_trial.agent.connect", new=AsyncMock(return_value=socket)):
                await agent.start()
                try:
                    first = socket.sent[0]
                    assert first["type"] == "session.start"
                    assert first["session"]["instructions"].startswith(CHERT_PROFILE)
                    assert first["session"]["model"] == "gpt-live-1.1"
                finally:
                    await agent.close()

    asyncio.run(scenario())


def test_silence_preroll_phrase_pause_and_flush():
    async def scenario():
        sink = Sink()
        agent = LiveAgent(None, sink)
        await agent._audio(block(False, 500))
        assert sink.begins == 0
        agent._words("Hey")
        await agent._audio(block(True))
        await agent._audio(block(False, 200))
        await agent._audio(block(True))
        assert sink.begins == 1 and sink.buffer.receiving
        assert sink.buffer.samples.max() >= 990
        await agent._audio(block(False, 440))
        assert not sink.buffer.receiving
        before = len(sink.buffer.samples)
        await agent._audio(block(True))
        assert sink.begins == 1 and len(sink.buffer.samples) > before

    asyncio.run(scenario())


def test_real_interrupt_and_backchannel_have_different_effects():
    async def scenario():
        sink = Sink()
        agent = LiveAgent(None, sink)
        await agent._audio(block(True))
        for start, text in [(200, "mm"), (400, "-hm")]:
            await agent._caller(dict(start_ms=start, end_ms=start + 200, delta=text))
        assert sink.cuts == 0
        await agent._caller(dict(start_ms=2600, end_ms=2800, delta="Wait"))
        await agent._caller(dict(start_ms=2800, end_ms=3000, delta=", the red one."))
        assert sink.cuts == 1
        await agent._audio(block(True, 100))
        assert sink.begins == 1 and not sink.buffer.active
        await agent._audio(block(False, 200))
        await agent._audio(block(True))
        assert sink.begins == 2 and sink.buffer.active
        agent._finish_caller()
        texts = [
            row["text"]
            for row in sink.transcripts.rows.values()
            if row["speaker"] == "user"
        ]
        assert texts == ["mm-hm", "Wait, the red one."]

    asyncio.run(scenario())


def test_live_phrase_can_resume_while_its_final_video_frame_is_still_playing():
    class PlayingSink(Sink):
        @property
        def response_active(self):
            return self.begins > 0

    async def scenario():
        sink = PlayingSink()
        agent = LiveAgent(None, sink)
        timeline = ContinuousTimeline()
        await agent._audio(block(True, 100))
        await agent._audio(block(False, 420))
        timeline.take(sink.buffer)
        await agent._audio(block(True, 240))
        await agent._audio(block(False, 440))
        _, position, valid = timeline.take(sink.buffer)
        assert valid > 0 and sink.begins == 1
        frame_id(sink.buffer.epoch, position, True)

    asyncio.run(scenario())


def test_facetime_greeting_is_once_and_does_not_interrupt_existing_speech():
    async def scenario():
        for turns in (0, 1):
            agent = LiveAgent(None, Sink())
            agent.ws = object()
            agent.send = AsyncMock()
            agent.turns = turns
            await asyncio.gather(agent.greet(), agent.greet())
            assert agent.send.await_count == (0 if turns else 1)
            if not turns:
                args = agent.send.call_args
                assert args.args == ("session.instructions.append",)
                assert args.kwargs["delegation_id"] is None
                assert "<say>Hi, I'm Chert." in args.kwargs["content"]
            agent.closed = True
            try:
                await agent.greet()
            except RuntimeError:
                pass
            else:
                raise AssertionError("Ended sessions must reject greetings")

    asyncio.run(scenario())
