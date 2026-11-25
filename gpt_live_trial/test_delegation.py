"""Check document retrieval and how evidence reaches the spoken answer."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

from realtime.camera import CameraInput
from realtime.config import Settings
from realtime.transcripts import TranscriptLedger
from realtime.web_search import SearchResult

from gpt_live_trial.delegation import LiveTools
from gpt_live_trial.knowledge import retrieve


def test_local_documents_have_grounded_answers_and_no_match():
    for query, expected, file in (
        ("waiting queue claim seconds", "30 seconds", "README.md"),
        ("gallery brick room", "Brick room", "realtime/demo-avatars/README.md"),
    ):
        result = retrieve(query)
        assert expected in result.summary
        assert any(src["url"].endswith("/" + file) for src in result.sources)
    result = retrieve("lunar zeppelin prototype")
    assert result.status == "no_matching_documents" and not result.sources


def make_tools():
    settings = Settings(
        _env_file=None, openai_api_key="test-only", web_search_enabled=True
    )
    agent = SimpleNamespace(
        settings=settings,
        closed=False,
        send=AsyncMock(),
        sink=SimpleNamespace(camera=CameraInput(), transcripts=TranscriptLedger()),
    )
    tools = LiveTools(agent)
    tools.caller("u1", 4000, 6000, "When is that game coming out?")
    tools.updated = time.monotonic() - 1
    return tools


def test_verified_lookup_returns_once_and_cites_played_answer():
    async def scenario():
        tools = make_tools()
        try:
            tools.route = AsyncMock(return_value=[("web", "release date")])
            link = {"url": "https://example.org/game", "title": "Release"}
            tools.search.search = AsyncMock(
                return_value=SearchResult("ok", "Out today.", (link,))
            )
            event = {"offset_ms": 6000, "delegation": {"id": "d1", "target": "client"}}
            tools.dispatch(event)
            tools.dispatch(event)
            await tools.task
            tools.route.assert_awaited_once()
            assert (
                tools.route.await_args.args[0]["latest_caller_request"]
                == "When is that game coming out?"
            )
            tools.search.search.assert_awaited_once()
            tools.agent.send.assert_awaited_once()
            assert "Out today." in tools.agent.send.await_args.kwargs["content"]
            ledger = tools.agent.sink.transcripts
            ledger.begin("reply", 1)
            tools.caption("reply")
            assert "sources" not in ledger.snapshot()["utterances"][-1]
            ledger.cue("reply", 1, "Out today.")
            ledger.advance(1, 2)
            assert ledger.snapshot()["utterances"][-1]["sources"] == [link]
            assert ledger.lookup_state == "idle"
        finally:
            await tools.close()

    asyncio.run(scenario())


def test_failed_search_has_no_evidence_or_source():
    async def scenario():
        tools = make_tools()
        try:
            tools.route = AsyncMock(return_value=[("web", "release date")])
            tools.search.search = AsyncMock(return_value=SearchResult("timeout"))
            await tools._run("d1", 6000)
            assert (
                '"status": "timeout"' in tools.agent.send.await_args.kwargs["content"]
            )
            assert (
                not tools.pending_sources
                and tools.agent.sink.transcripts.lookup_state == "idle"
            )
        finally:
            await tools.close()

    asyncio.run(scenario())


def test_query_superseded_during_search_does_not_speak_old_result():
    async def scenario():
        tools = make_tools()
        try:
            tools.route = AsyncMock(return_value=[("web", "release date")])

            async def search(*args, **kwargs):
                tools.caller("u2", 9000, 9400, "Actually, never mind.")
                return SearchResult("ok", "Stale answer")

            tools.search.search = search
            await tools._run("d1", 6000)
            tools.agent.send.assert_not_awaited()
            assert (
                not tools.pending_sources
                and tools.agent.sink.transcripts.lookup_state == "idle"
            )
        finally:
            await tools.close()

    asyncio.run(scenario())


def test_revoked_camera_drops_inflight_visual_evidence():
    async def scenario():
        tools = make_tools()
        try:
            camera = tools.agent.sink.camera
            tools.settings.camera_vision = True
            assert (
                await tools.read_camera("Read the card", camera.generation)
            ).status == "camera_unavailable"
            camera.track_sid, camera.frame = "shared", object()
            camera.snapshot = AsyncMock(return_value="data:image/jpeg;base64,example")

            async def answer(**kwargs):
                camera.clear()
                return SimpleNamespace(status="completed", output_text="Old card")

            tools.client.responses.create = answer
            result = await tools.read_camera("Read the card", camera.generation)
            assert result.status == "camera_unavailable" and not result.summary
        finally:
            await tools.close()

    asyncio.run(scenario())
