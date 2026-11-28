import asyncio
import json
from unittest.mock import AsyncMock

import numpy as np

from gpt_live_trial.agent import LiveAgent, MODEL, VOICE
from gpt_live_trial.greeting import GreetingCache, GreetingClip, greeting_text, identity
from gpt_live_trial.test_agent import Sink, block
from realtime.continuous import ContinuousTimeline
from realtime.contracts import FRAME_SAMPLES


def clip():
    # A different value per native video frame makes sample loss/reordering visible.
    pcm = np.repeat(np.arange(1, 81, dtype=np.int16), FRAME_SAMPLES)
    return GreetingClip(pcm, ((1000, "Hi, I'm Chert."),
                             (len(pcm), greeting_text(True))))


def agent(sink=None):
    voice = LiveAgent(None, sink or Sink())
    voice.ws = object()
    voice.send = AsyncMock()
    voice.recording_enabled = True
    voice.greeting_clip = clip()
    return voice


def test_cache_integrity_and_invalidation(tmp_path):
    cache = GreetingCache(tmp_path)
    descriptor = identity(MODEL, VOICE, "prompt", greeting_text(True))
    greeting = clip()
    path = cache.save(descriptor, greeting.pcm, greeting.cues)
    assert np.array_equal(cache.load(descriptor).pcm, greeting.pcm)
    assert path.stat().st_mode & 0o777 == 0o600
    for field in ("model", "voice", "instructions_sha256", "text", "rate", "version"):
        assert cache.load({**descriptor, field: "changed"}) is None
    original = json.loads(path.read_text())
    for change in ({"sha256": "bad"}, {"pcm": "!"}, {"cues": [[-1, "bad"]]},
                   {"cues": [[0, "Missing the notice"]]}, {"identity": {}}):
        path.write_text(json.dumps({**original, **change}))
        assert cache.load(descriptor) is None
    path.write_bytes(b"x" * 2_000_001)
    assert cache.load(descriptor) is None
    assert cache.load({**descriptor, "text": greeting_text(False)}) is None
    path.unlink()
    assert cache.load(descriptor) is None


def test_cached_pcm_preserves_continuous_video_history_and_exact_sample_order():
    async def run():
        voice = agent()
        timeline = ContinuousTimeline()
        timeline.take(voice.sink.buffer)  # already-rendered listening video
        previous = timeline.position
        await voice.greet()
        await voice.greet()  # bridge-ready retries cannot duplicate playback
        assert voice.sink.begins == voice.turns == voice.send.await_count == 1
        assert "Do not say or repeat" in voice.send.call_args.kwargs["content"]
        assert "This call is recorded" in voice.send.call_args.kwargs["content"]
        assert not voice.sink.buffer.receiving
        captured = []
        positions = []
        while voice.sink.buffer.active:
            audio, offset, valid = timeline.take(voice.sink.buffer)
            assert audio.sample_start >= previous  # no renderer/model rewind
            positions.append(offset)
            captured.extend(audio.pcm[:valid])
        assert positions == [0, 32 * FRAME_SAMPLES, 64 * FRAME_SAMPLES]
        assert np.array_equal(captured, voice.greeting_clip.pcm)
        voice.sink.transcripts.advance(voice.epoch, len(captured), complete=True)
        row = voice.sink.transcripts.snapshot()["utterances"][0]
        assert row["text"] == greeting_text(True) and row["state"] == "complete"
        # Completing the canned epoch does not suppress the next live response.
        voice._words("Live answer")
        await voice._audio(block(True))
        assert not voice.cached_greeting and voice.segment == "gpt-live-2"
        assert voice.words == "Live answer"
        assert voice.sink.begins == 2
    asyncio.run(run())


def test_model_output_cannot_overwrite_cached_audio_and_caller_can_interrupt():
    async def run():
        voice = agent()
        await voice.greet()
        old_epoch = voice.epoch
        before = voice.sink.buffer.samples.copy()
        await voice._audio(block(True, 200))
        voice._words("Unsolicited provider speech")
        assert np.array_equal(before, voice.sink.buffer.samples)
        assert voice.pending_words == []
        await voice._caller(dict(start_ms=100, end_ms=400, delta="Tell me about this"))
        assert not voice.cached_greeting and voice.sink.cuts == 1
        assert voice.sink.buffer.epoch != old_epoch and not voice.sink.buffer.active
        assert voice.sink.transcripts.rows["assistant:cached-greeting"]["state"] == "interrupted"
        await voice._audio(block(True))
        assert voice.sink.begins == 2 and voice.segment == "gpt-live-2"
        await voice.greet()
        assert voice.sink.begins == 2  # late bridge retry cannot replay opening
    asyncio.run(run())


def test_caller_first_cache_miss_and_failed_context_send():
    async def run():
        voice = agent()
        await voice._caller(dict(start_ms=100, end_ms=200, delta="Hello"))
        await voice.greet()
        assert voice.sink.begins == 0 and not voice.send.called
        assert voice.greeting_status == "skipped_caller_started"
        voice = agent()
        voice.greeting_clip = None
        await voice.greet()
        await voice.greet()
        assert voice.send.await_count == 1 and voice.sink.begins == 0
        assert "Read the following exactly" in voice.send.call_args.kwargs["content"]
        voice = agent()
        voice.send.side_effect = RuntimeError("Disconnected")
        try:
            await voice.greet()
        except RuntimeError:
            pass
        assert not voice.greeted and not voice.sink.buffer.active
    asyncio.run(run())
