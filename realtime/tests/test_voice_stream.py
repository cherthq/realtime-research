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


@pytest.mark.parametrize("text_only", [False, True])
def test_barge_in_waits_for_provider_done_before_requesting_next_response(text_only):
    from unittest.mock import AsyncMock
    from realtime.text_voice_agent import TextVoiceAgent

    async def run():
        sink = Sink()
        cls = TextVoiceAgent if text_only else VoiceAgent
        agent = cls(
            Settings(avatar_catalog="research", _env_file=None, camera_vision=False),
            {"voice": "cedar"},
            sink,
        )
        agent.connection = Obj(
            response=Obj(create=AsyncMock(), cancel=AsyncMock()),
            conversation=Obj(item=Obj(delete=AsyncMock(), create=AsyncMock())),
        )
        assert not agent.input_options()["turn_detection"]["create_response"]
        await agent.handle(Obj(type="input_audio_buffer.committed", item_id="u1"))
        await agent.turn_task
        await agent.handle(Obj(type="response.created", response=Obj(id="r1")))
        await agent.handle(Obj(type="input_audio_buffer.speech_started"))
        assert agent.response_id is None and not sink.buffer.active
        assert not agent.provider_idle.is_set()
        await agent.handle(Obj(type="input_audio_buffer.speech_stopped"))
        await agent.handle(Obj(type="input_audio_buffer.committed", item_id="u2"))
        await asyncio.sleep(0)
        assert agent.connection.response.create.await_count == 1
        agent.connection.response.cancel.assert_awaited_with(response_id="r1")
        await agent.handle(
            Obj(type="response.done", response=Obj(id="r1", status="cancelled"))
        )
        await agent.turn_task
        assert agent.connection.response.create.await_count == 2
        await agent.handle(Obj(type="response.created", response=Obj(id="r2")))
        assert agent.response_id == "r2" and sink.buffer.receiving
        # Duplicate commitment must neither cancel this reply nor create another.
        await agent.handle(Obj(type="input_audio_buffer.committed", item_id="u2"))
        assert agent.connection.response.create.await_count == 2
        assert not agent.failure
        await agent.close()

    asyncio.run(run())


@pytest.mark.parametrize("text_only", [False, True])
def test_late_created_response_after_interrupt_never_reopens_speech(text_only):
    from unittest.mock import AsyncMock
    from realtime.text_voice_agent import TextVoiceAgent

    async def run():
        sink = Sink()
        cls = TextVoiceAgent if text_only else VoiceAgent
        agent = cls(
            Settings(avatar_catalog="research", _env_file=None),
            {"voice": "cedar"},
            sink,
        )
        agent.connection = Obj(response=Obj(create=AsyncMock(), cancel=AsyncMock()))
        await agent.handle(Obj(type="input_audio_buffer.committed", item_id="u1"))
        await agent.turn_task
        generation = agent.turn_generation
        await agent.handle(Obj(type="input_audio_buffer.speech_started"))
        await agent.handle(Obj(type="input_audio_buffer.speech_stopped"))
        await agent.handle(Obj(type="input_audio_buffer.committed", item_id="u2"))
        await asyncio.sleep(0)
        await agent.handle(
            Obj(
                type="response.created",
                response=Obj(
                    id="obsolete", metadata={"turn_generation": str(generation)}
                ),
            )
        )
        agent.connection.response.cancel.assert_awaited_once_with(
            response_id="obsolete"
        )
        assert agent.response_id is None and not sink.buffer.active
        assert agent.connection.response.create.await_count == 1
        await agent.handle(
            Obj(type="response.done", response=Obj(id="obsolete", status="cancelled"))
        )
        await agent.turn_task
        assert agent.connection.response.create.await_count == 2
        await agent.handle(Obj(type="response.created", response=Obj(id="current")))
        assert agent.response_id == "current"
        # A delayed old done must not unlock the new active provider slot.
        await agent.handle(
            Obj(type="response.done", response=Obj(id="obsolete", status="cancelled"))
        )
        assert not agent.provider_idle.is_set()
        assert not agent.failure
        await agent.close()

    asyncio.run(run())


def test_interrupt_during_create_send_keeps_provider_slot_reserved():
    from unittest.mock import AsyncMock

    async def run():
        entered = asyncio.Event()

        async def slow_send(**kwargs):
            entered.set()
            await asyncio.Event().wait()

        agent = VoiceAgent(
            Settings(avatar_catalog="research", _env_file=None),
            {"voice": "cedar"},
            Sink(),
        )
        agent.connection = Obj(
            response=Obj(create=AsyncMock(side_effect=slow_send), cancel=AsyncMock())
        )
        await agent.handle(Obj(type="input_audio_buffer.committed", item_id="u1"))
        sending = agent.turn_task
        await entered.wait()
        await agent.handle(Obj(type="input_audio_buffer.speech_started"))
        await asyncio.gather(sending, return_exceptions=True)
        assert not agent.provider_idle.is_set()
        await agent.handle(Obj(type="response.created", response=Obj(id="late")))
        assert agent.response_id is None
        agent.connection.response.cancel.assert_awaited_once_with(response_id="late")
        await agent.handle(
            Obj(type="response.done", response=Obj(id="late", status="cancelled"))
        )
        assert agent.provider_idle.is_set() and not agent.failure
        await agent.close()

    asyncio.run(run())


@pytest.mark.parametrize("text_only", [False, True])
def test_provider_event_codes_are_retained_without_message_or_request(text_only):
    from realtime.text_voice_agent import TextVoiceAgent

    async def run():
        sink = Sink()
        agent_type = TextVoiceAgent if text_only else VoiceAgent
        agent = agent_type(
            Settings(avatar_catalog="research", _env_file=None),
            {"voice": "cedar"},
            sink,
        )
        for _ in range(20):
            await agent.handle(
                Obj(
                    type="error",
                    error=Obj(
                        code="response_cancel_not_active",
                        type="invalid_request_error",
                        message="private-transcript",
                        request={"api_key": "private-key"},
                    ),
                )
            )
        assert not agent.failure
        assert len(agent.diagnostics()["provider_errors"]) == 12
        await agent.handle(
            Obj(
                type="error",
                error=Obj(
                    code="rate_limit_exceeded",
                    type="server_error",
                    message="private-key",
                ),
            )
        )
        assert agent.failure
        assert agent.diagnostics()["provider_errors"][-1] == {
            "code": "rate_limit_exceeded",
            "type": "server_error",
        }
        assert "private-" not in str(agent.diagnostics())
        await agent._shared_event(
            Obj(type="error", error=Obj(code="private data\n", type=None))
        )
        assert agent.diagnostics()["provider_errors"][-1] == {
            "code": "unknown",
            "type": "unknown",
        }
        await asyncio.gather(*agent.tasks)

    asyncio.run(run())


def test_interrupt_truncates_played_position_and_drops_late_audio():
    async def run():
        sink, sent = Sink(), []
        agent = VoiceAgent(
            Settings(avatar_catalog="research", _env_file=None),
            {"voice": "cedar"},
            sink,
        )

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


@pytest.mark.parametrize("has_item,bound", [(False, 320), (True, 0), (True, 320)])
def test_interrupt_only_waits_for_receipt_when_it_can_affect_history(has_item, bound):
    from unittest.mock import AsyncMock

    async def run():
        sink = Sink()
        sink.server_played_ms = lambda: bound
        sink.interrupted_played_ms = AsyncMock(return_value=160)
        agent = VoiceAgent(
            Settings(
                avatar_catalog="research",
            ),
            {"voice": "cedar"},
            sink,
        )
        agent.connection = Obj(
            response=Obj(cancel=AsyncMock()),
            conversation=Obj(item=Obj(truncate=AsyncMock())),
        )
        await agent.handle(Obj(type="response.created", response=Obj(id="r1")))
        agent.item_id = "i1" if has_item else None
        await agent.interrupt()
        assert not sink.buffer.active
        agent.connection.response.cancel.assert_awaited_once_with(response_id="r1")
        if has_item and bound:
            sink.interrupted_played_ms.assert_awaited_once()
        else:
            sink.interrupted_played_ms.assert_not_awaited()
        if has_item:
            agent.connection.conversation.item.truncate.assert_awaited_once_with(
                item_id="i1", content_index=0, audio_end_ms=160 if bound else 0
            )
        else:
            agent.connection.conversation.item.truncate.assert_not_awaited()

    asyncio.run(run())


def test_provider_voice_mismatch_and_uplink_overflow_are_explicit_failures():
    async def run():
        sink = Sink()
        agent = VoiceAgent(
            Settings(avatar_catalog="research", _env_file=None),
            {"voice": "cedar"},
            sink,
        )
        await agent.handle(
            Obj(
                type="session.updated",
                session=Obj(model="test", audio=Obj(output=Obj(voice="marin"))),
            )
        )
        assert agent.ready.is_set() and "selected voice" in sink.error
        agent = VoiceAgent(
            Settings(avatar_catalog="research", _env_file=None),
            {"voice": "cedar"},
            Sink(),
        )
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
        agent = VoiceAgent(
            Settings(avatar_catalog="research", _env_file=None),
            {"voice": "cedar"},
            sink,
        )

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
        agent = VoiceAgent(
            Settings(avatar_catalog="research", _env_file=None),
            {"voice": "cedar"},
            sink,
        )
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
        agent = VoiceAgent(
            Settings(avatar_catalog="research", _env_file=None),
            {"voice": "cedar"},
            sink,
        )
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
        agent = VoiceAgent(
            Settings(avatar_catalog="research", _env_file=None),
            {"voice": "cedar"},
            sink,
        )
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
        agent = VoiceAgent(
            Settings(avatar_catalog="research", _env_file=None),
            {"voice": "cedar"},
            sink,
        )
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
        agent = VoiceAgent(
            Settings(avatar_catalog="research", _env_file=None),
            {"voice": "cedar"},
            sink,
        )
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
        agent = VoiceAgent(
            Settings(avatar_catalog="research", _env_file=None),
            {"voice": "cedar"},
            sink,
        )
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


@pytest.mark.parametrize("silence_ms", [350, 500])
def test_voice_session_uses_configured_turn_wait_without_reintroducing_token_cap(
    silence_ms,
):
    from unittest.mock import AsyncMock

    async def run():
        connection = Obj(session=Obj(update=AsyncMock()))
        context = AsyncMock()
        context.__aenter__.return_value = connection
        client = Obj(realtime=Obj(connect=lambda **_: context), close=AsyncMock())
        agent = VoiceAgent(
            Settings(avatar_catalog="research", voice_silence_duration_ms=silence_ms),
            {"voice": "cedar"},
            Sink(),
            client,
        )

        async def receive():
            agent.ready.set()
            await asyncio.Event().wait()

        agent._receive = receive
        try:
            await agent.start()
            payload = connection.session.update.call_args.kwargs["session"]
            assert (
                payload["audio"]["input"]["turn_detection"]["silence_duration_ms"]
                == silence_ms
            )
            assert payload["max_output_tokens"] == "inf"
            assert agent.diagnostics()["silence_duration_ms"] == silence_ms
        finally:
            await agent.close()

    asyncio.run(run())
