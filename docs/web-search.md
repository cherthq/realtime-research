# On-demand web search

The text conversation agent can look up current public information before speaking.
This works with `TTS_PROVIDER=elevenlabs` or `qwen`; the native OpenAI audio-output
path does not implement this tool. Avatar generation and voice selection are unchanged.
This is web grounding, not an uploaded-document knowledge base or persistent memory.

In the private `realtime/.env`, set:

```dotenv
WEB_SEARCH_ENABLED=true
WEB_SEARCH_MODEL=gpt-4.1-mini
WEB_SEARCH_TIMEOUT_SECONDS=8
WEB_SEARCH_CACHE_SECONDS=60
```

Restart the demo after changing these settings. It uses the existing server-side
`OPENAI_API_KEY` with access to the Responses API and web search. Lookup requests
add API usage and latency. Set `WEB_SEARCH_ENABLED=false` to disable the tool.

Try “Search the web for the latest release of Python” or ask for a recent fact.
The conversation model decides when to call the tool; ordinary chat and camera
descriptions should bypass it. The character says a short line such as “Let me
check that online” using its selected voice and synchronized avatar animation.
Retrieval runs concurrently with that acknowledgement. The result waits for the
acknowledgement's playback confirmation before starting, so fast searches cannot
cut it off. One notice is spoken per lookup; an existing model-written preamble
is not duplicated. “Checking…” appears during lookup. The short spoken
answer has clickable sources in its captions and conversation history. Citations
support the retrieved evidence; the spoken synthesis can still make mistakes.

Only one lookup is allowed in each tool continuation, with a bounded timeout and
no automatic retry. Failure returns an explicit unavailable result and instructions
to acknowledge uncertainty. The next spoken turn can search again. Interrupting
cancels the acknowledgement and request, closes its tool call, and prevents stale
spoken continuations. An already completed search is also discarded if the caller
interrupts while its acknowledgement is still playing. Playback confirmation is
bounded: a stalled stream cannot indefinitely retain a pending result.

Successful identical queries are cached for up to 60 seconds in that call only,
with at most 16 entries. The cache and transcript ledger are cleared at teardown.
No application logs contain search queries or retrieved content. Requests use
`store=false`; this is not a claim about provider-side abuse-monitoring retention.
The voice agent is instructed to send minimal public queries, never private camera
content. Retrieved pages are untrusted evidence, not instructions. Results must
contain a completed search and valid HTTP(S) citations; the server never downloads
their URLs. Source links open separately, without access to the demo window.

Implementation: `realtime/web_search.py` handles retrieval and caching;
`realtime/search_turn.py` owns cancellation and Realtime tool continuation;
`realtime/transcripts.py` publishes lookup state and sources with played speech.
Worker voice diagnostics expose lookup counts and duration, not query text.

## Validation

Verified on 2026-11-01 with 78 backend tests, six browser checks and a production
frontend build. A live synthetic-microphone call exercised Realtime, hosted search,
ElevenLabs, AvatarForever and LiveKit together: ordinary conversation made no
lookup; a searched reply displayed two clickable sources; a second lookup was
cancelled by speech and followed by a new spoken reply. No provider errors occurred.
The deployed test retained 512×384 generation, the 24 FPS publication cap and the
`onset_boost` listening profile. This was a short integration test, not a long-call
visual-quality or factual-accuracy benchmark. Test audio, screenshots and reports
are ignored under `.cache/web-search-20261101/`.

The spoken-acknowledgement update passed 81 backend tests and a live call on the
same date. The first acknowledgement completed before the searched answer; a
second was interrupted during playback and its pending answer was discarded.
The call then answered the new request without provider errors. Synthetic-input
recordings and the report are ignored under `.cache/search-ack-20261101/`.

References: [Realtime function calling](https://developers.openai.com/api/docs/guides/realtime-conversations)
and [Responses web search](https://developers.openai.com/api/docs/guides/tools-web-search).
