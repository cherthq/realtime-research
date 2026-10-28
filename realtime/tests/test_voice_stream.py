import asyncio
import base64
from types import SimpleNamespace as Obj

import numpy as np
import pytest

from realtime.audio import PCMResampler
from realtime.config import Settings
from realtime.contracts import FRAME_SAMPLES, LIVE_VOICE_PENDING_SECONDS, ResponseBuffer
from realtime.voice_agent import VoiceAgent


def test_stream_waits_for_arrived_samples_and_keeps_final_partial():
    buffer = ResponseBuffer()
    epoch = buffer.begin("r1")
    first = np.arange(1000, dtype=np.int16)
    buffer.append("r1", epoch, first)
    assert buffer.next().valid_samples == 0
    assert buffer.position == 0  # no invented silence inside the response timeline
    buffer.append("r1", epoch, np.arange(1000, 2501, dtype=np.int16))
    chunk = buffer.next()
    assert np.array_equal(chunk.pcm, np.arange(FRAME_SAMPLES, dtype=np.int16))
    assert buffer.next().valid_samples == 0
    buffer.finish("r1", epoch)
    final = buffer.next()
    assert final.valid_samples == 581 and final.sample_start == FRAME_SAMPLES
    assert not buffer.active and not np.any(final.pcm[581:])
    buffer.cancel()
    assert not buffer.append("r1", epoch, first)
    buffer.begin("r2")
    assert buffer.position == 0 and len(buffer.samples) == 0


def test_stateful_resampling_preserves_pause_and_final_coverage_across_packet_sizes():
    samples = np.zeros(24_013, dtype=np.int16)
    samples[:6000] = (np.sin(np.arange(6000) * 2 * np.pi * 440 / 24_000) * 5000).astype(
        np.int16
    )
    samples[18_000:] = 1000
    whole = PCMResampler(24_000, 48_000)
    expected = np.concatenate((whole.push(samples.tobytes()), whole.finish()))
    streamed = PCMResampler(24_000, 48_000)
    parts = [
        streamed.push(samples[i : i + 479].tobytes())
        for i in range(0, len(samples), 479)
    ]
    actual = np.concatenate([*parts, streamed.finish()])
    assert len(actual) == 2 * len(samples) == len(expected)
    # Sox's independent int16 output dither can differ by two least-significant bits.
    assert np.max(np.abs(actual.astype(int) - expected.astype(int))) <= 2
    assert np.max(np.abs(actual[14000:34000])) <= 1
    with pytest.raises(ValueError):
        streamed.push(b"\x00\x00")


class Sink:
    def __init__(self):
        self.buffer = ResponseBuffer()
        self.error = None
        self.status = "listening"
        self.cancelled = 0

    async def begin_voice_response(self, response_id):
        return self.buffer.begin(response_id)

    async def cancel_media(self):
        self.cancelled += 1
        self.buffer.cancel()

    def estimated_played_ms(self):
        return 320


def test_interrupt_truncates_played_position_and_drops_late_audio():
    async def run():
        sink, sent = Sink(), []
        agent = VoiceAgent(Settings(_env_file=None), {"voice": "cedar"}, sink)

        async def cancel(**kw):
            sent.append(("cancel", kw))

        async def truncate(**kw):
            # Queue has already been cleared before network acknowledgement.
            assert not sink.buffer.active
            sent.append(("truncate", kw))

        agent.connection = Obj(
            response=Obj(cancel=cancel), conversation=Obj(item=Obj(truncate=truncate))
        )
        await agent.handle(Obj(type="response.created", response=Obj(id="r1")))
        delta = Obj(
            type="response.output_audio.delta",
            response_id="r1",
            item_id="item1",
            content_index=0,
            delta=base64.b64encode(np.ones(24_000, dtype=np.int16).tobytes()).decode(),
        )
        await agent.handle(delta)
        assert sink.buffer.active
        await agent.interrupt()
        await agent.handle(delta)
        assert len(sink.buffer.samples) == 0
        assert sent == [
            ("cancel", {"response_id": "r1"}),
            ("truncate", {"item_id": "item1", "content_index": 0, "audio_end_ms": 320}),
        ]
        assert agent.profile["voice"] == "cedar"
        await agent.handle(Obj(type="response.created", response=Obj(id="r2")))
        assert sink.buffer.response_id == "r2" and sink.buffer.position == 0

    asyncio.run(run())


def test_provider_voice_mismatch_and_uplink_overflow_are_explicit_failures():
    async def run():
        sink = Sink()
        agent = VoiceAgent(Settings(_env_file=None), {"voice": "cedar"}, sink)
        await agent.handle(
            Obj(
                type="session.updated",
                session=Obj(model="test", audio=Obj(output=Obj(voice="marin"))),
            )
        )
        assert agent.ready.is_set() and "selected voice" in sink.error
        agent = VoiceAgent(Settings(_env_file=None), {"voice": "cedar"}, Sink())
        for _ in range(51):
            agent.microphone(bytes(1920))
        assert agent.incoming.qsize() == 50 and "uplink stalled" in agent.failure

    asyncio.run(run())


def test_interrupt_waits_for_browser_receipt_after_cancelling_output():
    async def run():
        class FeedbackSink(Sink):
            async def interrupted_played_ms(self, epoch, estimate):
                assert self.cancelled == 1 and not self.buffer.active
                assert epoch == 1 and estimate == 320
                return 160

        sink, truncations = FeedbackSink(), []
        agent = VoiceAgent(Settings(_env_file=None), {"voice": "cedar"}, sink)

        async def truncate(**fields):
            truncations.append(fields)

        agent.connection = Obj(conversation=Obj(item=Obj(truncate=truncate)))
        await agent.handle(Obj(type="response.created", response=Obj(id="r1")))
        agent.item_id = "item1"
        await agent.interrupt(provider_already_cancelled=True)
        assert truncations == [
            {"item_id": "item1", "content_index": 0, "audio_end_ms": 160}
        ]

    asyncio.run(run())


def test_voice_timings_use_one_clock_and_reset_on_a_new_speech_turn(monkeypatch):
    import realtime.voice_agent as module

    now = [100.0]
    monkeypatch.setattr(module, "time", Obj(monotonic=lambda: now[0]))

    async def run():
        sink = Sink()
        agent = VoiceAgent(Settings(_env_file=None), {"voice": "cedar"}, sink)
        await agent.handle(Obj(type="input_audio_buffer.speech_stopped"))
        now[0] = 100.1
        await agent.handle(Obj(type="response.created", response=Obj(id="r1")))
        sink.request_time = 100.11
        now[0] = 100.3
        await agent.handle(
            Obj(
                type="response.output_audio.delta",
                response_id="r1",
                item_id="i1",
                content_index=0,
                delta=base64.b64encode(bytes(480)).decode(),
            )
        )
        measured = agent.timing(700)
        assert measured["speech_stop_to_response_ms"] == 100
        assert measured["response_to_first_audio_ms"] == 200
        assert measured["first_audio_to_publish_ms"] == 510
        assert measured["speech_stop_to_publish_ms"] == 810
        await agent.handle(Obj(type="input_audio_buffer.speech_started"))
        assert all(
            value is None
            for name, value in agent.timing().items()
            if name.endswith("_ms")
        )

    asyncio.run(run())


@pytest.mark.parametrize("reason", ["max_output_tokens", "content_filter", None])
@pytest.mark.parametrize("audio_done_first", [False, True])
def test_incomplete_reply_drains_final_audio_and_allows_next_turn(
    reason, audio_done_first
):
    async def run():
        sink = Sink()
        agent = VoiceAgent(Settings(_env_file=None), {"voice": "cedar"}, sink)
        await agent.handle(Obj(type="response.created", response=Obj(id="r1")))
        pcm = np.ones(24013, dtype=np.int16)
        delta = Obj(
            type="response.output_audio.delta",
            response_id="r1",
            item_id="i1",
            content_index=0,
            delta=base64.b64encode(pcm.tobytes()).decode(),
        )
        await agent.handle(delta)
        done = Obj(type="response.output_audio.done", response_id="r1")
        if audio_done_first:
            await agent.handle(done)
        await agent.handle(
            Obj(
                type="response.done",
                response=Obj(
                    id="r1",
                    status="incomplete",
                    status_details=Obj(reason=reason),
                    usage=Obj(output_tokens=240),
                ),
            )
        )
        await agent.handle(done)  # duplicate/late finalization cannot duplicate samples
        await agent.handle(delta)  # late audio cannot crash or reopen a finalized turn
        assert agent.failure is None and sink.error is None and sink.cancelled == 0
        assert not agent.generating and not sink.buffer.receiving
        assert len(sink.buffer.samples) == len(pcm) * 2
        published = []
        while sink.buffer.active:
            published.append(sink.buffer.next().valid_samples)
        assert sum(published) == len(pcm) * 2 and published[-1] < FRAME_SAMPLES
        assert agent.diagnostics()["recent_responses"] == [
            {
                "status": "incomplete",
                "reason": reason,
                "error_code": None,
                "output_tokens": 240,
            }
        ]
        await agent.handle(Obj(type="response.created", response=Obj(id="r2")))
        await agent.handle(
            Obj(type="response.done", response=Obj(id="r2", status="completed"))
        )
        assert agent.failure is None and not sink.buffer.active
        assert agent.diagnostics()["counts"] == {"incomplete": 1, "completed": 1}

    asyncio.run(run())


def test_failed_reply_remains_fatal_and_diagnostics_do_not_expose_provider_message(
    caplog,
):
    async def run():
        sink = Sink()
        agent = VoiceAgent(Settings(_env_file=None), {"voice": "cedar"}, sink)
        await agent.handle(Obj(type="response.created", response=Obj(id="r1")))
        await agent.handle(
            Obj(
                type="response.done",
                response=Obj(
                    id="r1",
                    status="failed",
                    status_details=Obj(
                        error=Obj(
                            code="insufficient_quota",
                            message="private-provider-message",
                        )
                    ),
                    output=[{"text": "private-user-content"}],
                ),
            )
        )
        await asyncio.sleep(0)
        assert agent.failure and "could not generate" in agent.failure
        assert sink.status == "error" and not sink.buffer.active
        assert (
            agent.diagnostics()["recent_responses"][0]["error_code"]
            == "insufficient_quota"
        )
        assert "private-" not in str(agent.diagnostics()) + caplog.text

    asyncio.run(run())


def test_late_terminal_event_cannot_cancel_new_response():
    async def run():
        sink = Sink()
        agent = VoiceAgent(Settings(_env_file=None), {"voice": "cedar"}, sink)
        await agent.handle(Obj(type="response.created", response=Obj(id="old")))
        await agent.interrupt(provider_already_cancelled=True)
        await agent.handle(Obj(type="response.created", response=Obj(id="new")))
        await agent.handle(
            Obj(type="response.done", response=Obj(id="old", status="failed"))
        )
        assert agent.failure is None and agent.generating
        assert sink.buffer.response_id == "new" and sink.buffer.receiving
        assert not agent.diagnostics()["recent_responses"]

    asyncio.run(run())


def test_response_diagnostics_are_bounded_and_do_not_relax_audio_queue_limit():
    async def run():
        sink = Sink()
        agent = VoiceAgent(Settings(_env_file=None), {"voice": "cedar"}, sink)
        for i in range(30):
            await agent.handle(Obj(type="response.created", response=Obj(id=str(i))))
            await agent.handle(
                Obj(type="response.done", response=Obj(id=str(i), status="completed"))
            )
        assert len(agent.diagnostics()["recent_responses"]) == 12
        assert agent.diagnostics()["counts"] == {"completed": 30}
        await agent.handle(Obj(type="response.created", response=Obj(id="too-long")))
        with pytest.raises(ValueError, match="20-second"):
            sink.buffer.append(
                agent.response_id, agent.epoch, np.zeros(48000 * 20 + 1, dtype=np.int16)
            )

    asyncio.run(run())


def test_long_voice_reply_survives_old_queue_limit_and_interrupt_discards_backlog():
    async def run():
        sink = Sink()
        sink.buffer = ResponseBuffer(max_pending_seconds=LIVE_VOICE_PENDING_SECONDS)
        agent = VoiceAgent(Settings(_env_file=None), {"voice": "cedar"}, sink)
        await agent.handle(Obj(type="response.created", response=Obj(id="long")))
        # Simulate 45 seconds arriving before the renderer can play anything.
        delta = Obj(
            type="response.output_audio.delta",
            response_id="long",
            item_id="i1",
            content_index=0,
            delta=base64.b64encode(np.ones(24000, dtype=np.int16).tobytes()).decode(),
        )
        for _ in range(45):
            await agent.handle(delta)
        await agent.handle(
            Obj(type="response.done", response=Obj(id="long", status="completed"))
        )
        assert agent.failure is None and len(sink.buffer.samples) == 48000 * 45
        assert agent.diagnostics()["max_output_tokens"] == "inf"
        await agent.interrupt(provider_already_cancelled=True)
        await agent.handle(delta)
        assert not sink.buffer.active and len(sink.buffer.samples) == 0
        await agent.handle(Obj(type="response.created", response=Obj(id="next")))
        assert sink.buffer.response_id == "next" and agent.failure is None

    asyncio.run(run())


def test_live_voice_backlog_is_bounded_and_reuses_played_capacity():
    buffer = ResponseBuffer(max_pending_seconds=LIVE_VOICE_PENDING_SECONDS)
    epoch = buffer.begin("long")
    capacity = 48000 * LIVE_VOICE_PENDING_SECONDS
    buffer.append("long", epoch, np.zeros(capacity, dtype=np.int16))
    with pytest.raises(ValueError, match="audio queue bound"):
        buffer.append("long", epoch, np.zeros(1, dtype=np.int16))
    assert buffer.next().valid_samples == FRAME_SAMPLES
    assert buffer.append("long", epoch, np.zeros(FRAME_SAMPLES, dtype=np.int16))
    buffer.cancel()
    assert not buffer.active and len(buffer.samples) == 0
