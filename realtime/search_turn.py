"""Cancelable Realtime tool continuation, separate from the media receive loop."""

import asyncio
from collections import deque
import contextlib
import json

from .web_search import SEARCH_INSTRUCTIONS, SEARCH_TOOL, SearchResult, WebSearch


SEARCH_ACKNOWLEDGEMENTS = (
    "Let me check that online.",
    "I'll look that up for you.",
    "One moment, let me check.",
)


def search_query(call):
    try:
        arguments = json.loads(call.arguments)
        if (
            call.name == "search_web"
            and isinstance(arguments, dict)
            and set(arguments) == {"query"}
        ):
            query = arguments["query"]
            if isinstance(query, str) and 1 <= len(query.strip()) <= 400:
                return query
    except (ValueError, TypeError):
        pass
    return None


class SearchTurn:
    def __init__(self, agent, search=None):
        self.agent = agent
        self.search = search or WebSearch(agent.settings)
        self.task = None
        self.handled = deque(maxlen=64)
        self.reply_sources = ()
        self.reply_generation = None
        self.state = "idle"
        self.acknowledgements = 0

    @property
    def enabled(self):
        return self.agent.settings.web_search_enabled

    def session_options(self):
        return {"tools": [SEARCH_TOOL], "tool_choice": "auto"} if self.enabled else {}

    def instructions(self):
        return SEARCH_INSTRUCTIONS if self.enabled else ""

    def start(self, response):
        calls = [
            item
            for item in getattr(response, "output", [])
            if getattr(item, "type", None) == "function_call"
        ]
        if (
            not calls
            or not self.enabled
            or response.status != "completed"
            or response.id in self.handled
        ):
            return
        self.handled.append(response.id)
        agent = self.agent
        if agent.closed or agent.user_speaking:
            return
        self.cancel()
        if (
            search_query(calls[0]) is not None
            and agent.first_text_at is None
            and not agent.text_finished
        ):
            # Reuse the current voice, avatar epoch and played-caption path.
            # The caller starts this before closing the response's text queue.
            agent.text_queue.put_nowait(
                SEARCH_ACKNOWLEDGEMENTS[
                    self.acknowledgements % len(SEARCH_ACKNOWLEDGEMENTS)
                ]
            )
            self.acknowledgements += 1
        self.state = "checking"
        if agent.transcripts:
            agent.transcripts.begin_lookup(response.id)
        self.task = asyncio.create_task(
            self._run(calls, response.id, agent.turn_generation, agent.tts_task)
        )

    def current(self, generation):
        return (
            not self.agent.closed
            and not self.agent.user_speaking
            and generation == self.agent.turn_generation
        )

    async def _run(self, calls, token, generation, speech_task):
        agent, sent = self.agent, set()

        async def deliver(call, result):
            if call.call_id in sent:
                return
            # A cancellation after a websocket write must not duplicate its output.
            sent.add(call.call_id)
            await agent.connection.conversation.item.create(
                item={
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result.payload()),
                }
            )

        try:
            results = []
            for index, call in enumerate(calls):
                result = SearchResult("lookup_budget_exhausted")
                if index == 0:
                    query = search_query(call)
                    result = (
                        await self.search.search(query, client=agent.client)
                        if query is not None
                        else SearchResult("invalid_query")
                    )
                results.append(result)
            if not self.current(generation):
                raise asyncio.CancelledError()
            # Retrieval overlaps acknowledgement synthesis and avatar playback.
            # Finishing TTS alone does not mean the caller has heard the audio.
            if speech_task:
                await asyncio.wait_for(asyncio.shield(speech_task), 15)
            if not self.current(generation) or agent.failure:
                raise asyncio.CancelledError()
            if agent.synthesized_samples and agent.transcripts:
                async with asyncio.timeout(30):
                    while not agent.transcripts.playback_complete(token):
                        if not self.current(generation) or agent.failure:
                            raise asyncio.CancelledError()
                        await asyncio.sleep(0.04)
            for call, result in zip(calls, results):
                await deliver(call, result)
                if not self.current(generation):
                    raise asyncio.CancelledError()
            self.reply_sources = tuple(
                source for result in results for source in result.sources
            )[:5]
            self.reply_generation = generation
            self.state = (
                "idle"
                if any(result.status == "ok" for result in results)
                else "unavailable"
            )
            if agent.transcripts:
                agent.transcripts.end_lookup(token, self.state)
            agent.pending_generation = generation
            agent.provider_idle.clear()
            await agent.connection.response.create(
                response={
                    "tool_choice": "none",  # One lookup per user turn; never loop on a failure.
                    "instructions": agent.instructions()
                    + " The caller already heard a short lookup acknowledgement. Now answer directly using the tool result, without another acknowledgement. If it failed, say you could not verify current information. Do not describe an old camera image as a current view.",
                    "metadata": {
                        "turn_generation": str(generation),
                        "web_search_followup": "1",
                    },
                }
            )
            agent.turn_counts["search_followups"] += 1
        except asyncio.CancelledError:
            # Close outstanding calls in conversation history, but never request
            # a spoken continuation from an interrupted/stale turn.
            with contextlib.suppress(Exception):
                async with asyncio.timeout(0.75):
                    for call in calls:
                        await deliver(call, SearchResult("cancelled"))
            raise
        except Exception:
            agent._fail(
                "Could not resume the conversation after lookup; reconnect the call."
            )
        finally:
            if agent.transcripts:
                agent.transcripts.end_lookup(token, "idle")
            if self.task is asyncio.current_task():
                self.state = "idle"

    def attach_sources(self, response_id):
        if (
            self.reply_generation == self.agent.turn_generation
            and self.agent.transcripts
        ):
            self.agent.transcripts.sources(response_id, self.reply_sources)
        self.reply_sources = ()
        self.reply_generation = None

    def cancel(self):
        task, self.task = self.task, None
        if task and not task.done():
            task.cancel()
        self.reply_sources = ()
        self.reply_generation = None
        self.state = "idle"
        if self.agent.transcripts:
            self.agent.transcripts.clear_lookup()
        return task

    async def close(self):
        task = self.cancel()
        if task:
            await asyncio.gather(task, return_exceptions=True)
        self.search.clear()

    def diagnostics(self):
        return {
            "enabled": self.enabled,
            "state": self.state,
            "acknowledgements": self.acknowledgements,
            **self.search.diagnostics(),
        }
