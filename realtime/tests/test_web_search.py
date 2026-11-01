import asyncio
import json
from types import SimpleNamespace as Obj
from unittest.mock import AsyncMock

import pytest

from realtime.config import Settings
from realtime.text_voice_agent import TextVoiceAgent
from realtime.tests.test_qwen_voice import TTS
from realtime.tests.test_voice_stream import Sink
from realtime.transcripts import TranscriptLedger
from realtime.web_search import SearchResult, WebSearch, safe_source
from realtime.search_turn import SEARCH_ACKNOWLEDGEMENTS


SOURCE = {"url": "https://example.com/report", "title": "Official report"}


def response(*, url=SOURCE["url"], searched=True):
    return Obj(
        status="completed",
        output=[
            *([Obj(type="web_search_call", status="completed")] if searched else []),
            Obj(
                type="message",
                content=[
                    Obj(
                        type="output_text",
                        text="Verified fact.",
                        annotations=[
                            Obj(type="url_citation", url=url, title=SOURCE["title"]),
                        ],
                    )
                ],
            ),
        ],
    )


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "https://127.0.0.1/a",
        "http://localhost",
        "file:///a",
        "https://user:pass@example.com",
        "https://example.com:9000",
        "https://host.local/a",
    ],
)
def test_unsafe_source_links_are_omitted(url):
    assert safe_source(url) is None


def test_only_completed_sourced_searches_can_ground_an_answer():
    assert WebSearch.parse(response()).sources == (SOURCE,)
    assert WebSearch.parse(response(searched=False)).status == "no_verified_sources"
    assert WebSearch.parse(response(url="javascript:alert(1)")).summary == ""
    assert WebSearch.parse(Obj(status="incomplete")).status == "unavailable"


def test_search_cache_is_bounded_private_and_requests_are_not_stored():
    async def run():
        settings = Settings(avatar_catalog="research", _env_file=None)
        search = WebSearch(settings)
        client = Obj(responses=Obj(create=AsyncMock(return_value=response())))
        first = await search.search("latest public report", client=client)
        again = await search.search("  LATEST   public report ", client=client)
        assert first.status == "ok" and again.cached
        assert client.responses.create.await_count == 1
        kwargs = client.responses.create.call_args.kwargs
        assert kwargs["tools"][0]["type"] == "web_search"
        assert kwargs["tool_choice"] == "required" and kwargs["store"] is False
        assert "latest public report" not in str(search.diagnostics())
        search.cache["latest public report"] = (0, first)
        await search.search("latest public report", client=client)
        assert client.responses.create.await_count == 2
        for i in range(20):
            await search.search(f"query {i}", client=client)
        assert len(search.cache) == 16
        search.clear()
        assert not search.cache

    asyncio.run(run())


def test_errors_timeout_and_invalid_queries_fail_without_inventing_sources():
    async def run():
        search = WebSearch(Settings(avatar_catalog="research", _env_file=None))
        client = Obj(
            responses=Obj(create=AsyncMock(side_effect=RuntimeError("private error")))
        )
        for query in ["", None, "x" * 401]:
            assert (await search.search(query, client=client)).status == "invalid_query"
        assert client.responses.create.await_count == 0
        assert (await search.search("news", client=client)).status == "unavailable"
        client.responses.create.side_effect = TimeoutError()
        assert (await search.search("news", client=client)).status == "timeout"
        assert not search.cache

    asyncio.run(run())


def agent(search=None, *, enabled=True):
    sink = Sink()
    sink.transcripts = TranscriptLedger()
    settings = Settings(
        avatar_catalog="research",
        _env_file=None,
        tts_provider="qwen",
        web_search_enabled=enabled,
    )
    value = TextVoiceAgent(settings, {"voice": "aiden"}, sink, tts=TTS(), search=search)
    value.connection = Obj(
        response=Obj(create=AsyncMock(), cancel=AsyncMock()),
        conversation=Obj(item=Obj(create=AsyncMock(), delete=AsyncMock())),
    )
    return value


async def tool_turn(
    value, *, arguments='{"query":"latest public report"}', ident="tool-1"
):
    await value.handle(Obj(type="response.created", response=Obj(id=ident)))
    result = Obj(
        id=ident,
        status="completed",
        output=[
            Obj(
                type="function_call",
                name="search_web",
                call_id=f"call-{ident}",
                arguments=arguments,
            )
        ],
    )
    await value.handle(Obj(type="response.done", response=result))
    return result


def fake_search():
    return Obj(
        search=AsyncMock(return_value=SearchResult("ok", "Verified fact.", (SOURCE,))),
        clear=lambda: None,
        diagnostics=lambda: {},
    )


def test_tool_continuation_waits_for_tts_and_sources_follow_played_speech():
    async def run():
        search = fake_search()
        value = agent(search)
        result = await tool_turn(value)
        assert value.transcripts.snapshot()["lookup"]["state"] == "checking"
        await value.tts_task
        value.transcripts.advance(value.epoch, 99999, complete=True)
        await value.lookup.task
        assert value.tts_task.done() and not value.failure
        assert value.connection.response.create.await_count == 1
        request = value.connection.response.create.call_args.kwargs["response"]
        assert request["tool_choice"] == "none" and "max_output_tokens" not in request
        value.lookup.start(result)  # duplicate done must not spend twice
        assert search.search.await_count == 1
        await value.handle(
            Obj(
                type="response.created",
                response=Obj(id="spoken", metadata=request["metadata"]),
            )
        )
        assert all(
            "sources" not in row for row in value.transcripts.snapshot()["utterances"]
        )
        await value.handle(
            Obj(
                type="response.output_text.delta",
                response_id="spoken",
                item_id="text",
                delta="Verified fact.",
            )
        )
        await value.handle(
            Obj(
                type="response.done",
                response=Obj(id="spoken", status="completed", output=[]),
            )
        )
        await value.tts_task
        value.transcripts.advance(value.epoch, 99999, complete=True)
        assert value.transcripts.snapshot()["utterances"][-1]["sources"] == [SOURCE]

    asyncio.run(run())


@pytest.mark.parametrize("late_completion", [False, True])
def test_interrupt_cancels_lookup_even_if_transport_returns_a_late_result(
    late_completion,
):
    async def run():
        search = fake_search()
        entered = asyncio.Event()

        async def slow(*args, **kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if late_completion:
                    return SearchResult("ok", "Stale answer", (SOURCE,))
                raise

        search.search.side_effect = slow
        value = agent(search)
        await tool_turn(value)
        await entered.wait()
        await value.handle(Obj(type="input_audio_buffer.speech_started"))
        assert value.connection.response.create.await_count == 0
        assert value.transcripts.snapshot()["lookup"]["state"] == "idle"
        assert not value.lookup.reply_sources
        outputs = value.connection.conversation.item.create.call_args_list
        assert json.loads(outputs[0].kwargs["item"]["output"])["status"] == "cancelled"
        assert not value.failure

    asyncio.run(run())


def test_failed_lookup_continues_honestly_and_normal_chat_never_searches():
    async def run():
        search = fake_search()
        value = agent(search)
        await tool_turn(value, arguments="malformed JSON")
        await value.lookup.task
        assert search.search.await_count == 0
        item = value.connection.conversation.item.create.call_args.kwargs["item"]
        assert json.loads(item["output"])["status"] == "invalid_query"
        assert value.connection.response.create.await_count == 1 and not value.failure
        ordinary = agent(search)
        await ordinary.handle(Obj(type="response.created", response=Obj(id="hello")))
        await ordinary.handle(
            Obj(
                type="response.done",
                response=Obj(id="hello", status="completed", output=[]),
            )
        )
        await ordinary.tts_task
        assert ordinary.lookup.task is None and search.search.await_count == 0
        assert agent(enabled=False).lookup.session_options() == {}

    asyncio.run(run())


def test_invalid_tool_response_does_not_announce_a_search():
    async def run():
        class HostedTTS:
            async def stream_response(self, phrases):
                pytest.fail("Empty tool output must never call the speech provider")
                yield

        value = agent(fake_search())
        value.tts = HostedTTS()
        await tool_turn(value, arguments="malformed")
        await value.lookup.task
        assert value.tts_task.done() and not value.failure
        assert value.connection.response.create.await_count == 1

    asyncio.run(run())


def test_lookup_runs_while_acknowledgement_plays_and_never_cuts_it_off():
    async def run():
        search = fake_search()
        value = agent(search)
        result = await tool_turn(value)
        await value.tts_task
        await asyncio.sleep(0)
        assert search.search.await_count == 1  # lookup already ran; playback has not
        assert value.connection.response.create.await_count == 0
        assert value.lookup.acknowledgements == 1
        assert not value.transcripts.playback_complete(result.id)
        assert not value.transcripts.snapshot()["utterances"][0]["text"]
        value.transcripts.advance(value.epoch, 99999, complete=True)
        await value.lookup.task
        assert (
            value.transcripts.snapshot()["utterances"][0]["text"]
            == SEARCH_ACKNOWLEDGEMENTS[0]
        )
        assert value.connection.response.create.await_count == 1
        assert value.lookup.acknowledgements == 1

    asyncio.run(run())


def test_early_text_done_keeps_room_for_notice_and_existing_preamble_is_not_duplicated():
    async def run():
        for with_preamble in (False, True):
            value = agent(fake_search())
            await value.handle(Obj(type="response.created", response=Obj(id="tool")))
            if with_preamble:
                await value.handle(
                    Obj(
                        type="response.output_text.delta",
                        response_id="tool",
                        item_id="pre",
                        delta="Let me check.",
                    )
                )
            await value.handle(
                Obj(type="response.output_text.done", response_id="tool")
            )
            assert not value.text_finished
            await value.handle(
                Obj(
                    type="response.done",
                    response=Obj(
                        id="tool",
                        status="completed",
                        output=[
                            Obj(
                                type="function_call",
                                name="search_web",
                                call_id="c",
                                arguments='{"query":"news"}',
                            )
                        ],
                    ),
                )
            )
            await value.tts_task
            value.transcripts.advance(value.epoch, 99999, complete=True)
            await value.lookup.task
            rows = value.transcripts.snapshot()["utterances"]
            assert len(rows) == 1
            assert rows[0]["text"] == (
                "Let me check." if with_preamble else SEARCH_ACKNOWLEDGEMENTS[0]
            )
            assert value.lookup.acknowledgements == (0 if with_preamble else 1)

    asyncio.run(run())


def test_interrupt_during_notice_playback_discards_already_returned_search_results():
    async def run():
        value = agent(fake_search())
        await tool_turn(value)
        await value.tts_task
        await asyncio.sleep(0)
        assert value.lookup.search.search.await_count == 1
        assert value.connection.response.create.await_count == 0
        await value.handle(Obj(type="input_audio_buffer.speech_started"))
        assert value.connection.response.create.await_count == 0
        assert value.lookup.reply_sources == ()
        assert not value.sink.buffer.active
        assert value.transcripts.snapshot()["lookup"]["state"] == "idle"

    asyncio.run(run())
