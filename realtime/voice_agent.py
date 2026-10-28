"""OpenAI voice connection with avatar-path audio and explicit history truncation.

This module never publishes audio directly. The worker owns paired A/V playout.
Playback uses frame-associated browser feedback when available; speaker latency
is not measured by these acknowledgements.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
from collections import Counter, deque
import logging
import re
import time

from openai import AsyncOpenAI

from .audio import PCMResampler
from .voices import VOICES

logger = logging.getLogger(__name__)

# Use the provider's maximum available output; no application token ceiling.
MAX_RESPONSE_TOKENS = "inf"


class VoiceAgent:
    def __init__(self, settings, profile: dict, sink, client=None):
        if profile["voice"] not in VOICES:
            raise ValueError("Unsupported voice")
        self.settings, self.profile, self.sink = settings, dict(profile), sink
        self.client = client
        self.connection = None
        self.context = None
        self.tasks = []
        self.incoming = asyncio.Queue(
            maxsize=50
        )  # at most one second of 20ms mic frames
        self.input_resampler = PCMResampler(48_000, 24_000)
        self.output_resampler = None
        self.response_id = None
        self.item_id = None
        self.content_index = 0
        self.epoch = None
        self.generating = False
        self.closed = False
        self.ready = asyncio.Event()
        self.failure = None
        self.resolved_model = None
        self.effective_voice = None
        self.control_lock = asyncio.Lock()
        self.response_outcomes = deque(maxlen=12)
        self.response_counts = Counter()
        self.speech_stopped_at = None
        self.response_created_at = None
        self.first_audio_at = None

    async def start(self):
        key = self.settings.openai_api_key.get_secret_value()
        if self.client is None:
            if not key:
                raise ValueError("OPENAI_API_KEY is not configured")
            self.client = AsyncOpenAI(api_key=key, timeout=20, max_retries=0)
        self.context = self.client.realtime.connect(
            model=self.settings.openai_realtime_model, max_retries=0
        )
        self.connection = await self.context.__aenter__()
        await self.connection.session.update(
            session={
                "type": "realtime",
                "output_modalities": ["audio"],
                "max_output_tokens": MAX_RESPONSE_TOKENS,
                "instructions": (
                    "You are a friendly AI assistant in a video-call demo. Respond naturally, with detail "
                    "appropriate to the question. Do not claim to see the caller or to be human. "
                    + self.profile.get("instructions", "")
                ),
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24_000},
                        "turn_detection": {
                            "type": "server_vad",
                            "create_response": True,
                            "interrupt_response": True,
                            "silence_duration_ms": 500,
                        },
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": 24_000},
                        "voice": self.profile["voice"],
                    },
                },
            }
        )
        self.tasks = [
            asyncio.create_task(self._receive()),
            asyncio.create_task(self._send_microphone()),
        ]
        await asyncio.wait_for(self.ready.wait(), 15)
        if self.failure:
            raise RuntimeError(self.failure)

    def microphone(self, pcm: bytes):
        if self.closed:
            return
        try:
            self.incoming.put_nowait(pcm)
        except asyncio.QueueFull:
            self._fail("Voice uplink stalled; end and reconnect the call.")

    def _fail(self, message: str):
        if self.failure:
            return
        self.failure = message
        self.ready.set()
        self.sink.error = message
        self.sink.buffer.cancel()
        self.sink.status = "error"

        async def silence():
            with contextlib.suppress(Exception):
                await self.sink.cancel_media()
            self.sink.status = "error"

        self.tasks.append(asyncio.create_task(silence()))

    async def _send_microphone(self):
        try:
            while not self.closed:
                data = self.input_resampler.push(await self.incoming.get())
                if len(data):
                    await asyncio.wait_for(
                        self.connection.input_audio_buffer.append(
                            audio=base64.b64encode(data.tobytes()).decode()
                        ),
                        2,
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            self._fail("Voice uplink failed; end and reconnect the call.")

    async def _receive(self):
        try:
            async for event in self.connection:
                await self.handle(event)
            if not self.closed:
                self._fail("Voice connection ended; start a new call.")
        except asyncio.CancelledError:
            raise
        except Exception:
            self._fail("Voice response failed; end and reconnect the call.")

    async def handle(self, event):
        if event.type == "session.updated":
            self.effective_voice = event.session.audio.output.voice
            self.resolved_model = event.session.model
            if self.effective_voice != self.profile["voice"]:
                self._fail("The provider did not apply the selected voice.")
            self.ready.set()
        elif event.type == "input_audio_buffer.speech_started":
            self.speech_stopped_at = None
            self.response_created_at = None
            self.first_audio_at = None
            await self.interrupt(provider_already_cancelled=True)
        elif event.type == "input_audio_buffer.speech_stopped":
            self.speech_stopped_at = time.monotonic()
        elif event.type == "response.created":
            async with self.control_lock:
                self.response_created_at = time.monotonic()
                self.first_audio_at = None
                if self.response_id and self.sink.buffer.active:
                    raise ValueError("Unexpected overlapping voice response")
                self.response_id = event.response.id
                self.item_id = None
                self.generating = True
                self.output_resampler = PCMResampler(24_000, 48_000)
                self.epoch = await self.sink.begin_voice_response(self.response_id)
        elif (
            event.type == "response.output_audio.delta"
            and event.response_id == self.response_id
        ):
            if self.output_resampler.closed:
                return  # A late delta must not reopen an already finalized response.
            if self.first_audio_at is None:
                self.first_audio_at = time.monotonic()
            self.item_id, self.content_index = event.item_id, event.content_index
            samples = self.output_resampler.push(
                base64.b64decode(event.delta, validate=True)
            )
            self.sink.buffer.append(self.response_id, self.epoch, samples)
        elif (
            event.type == "response.output_audio.done"
            and event.response_id == self.response_id
        ):
            self._finish_output()
        elif event.type == "response.done" and event.response.id == self.response_id:
            self.generating = False
            self._record_response_outcome(event.response)
            if event.response.status in {"completed", "cancelled", "incomplete"}:
                # Incomplete ends this reply, not the Realtime session. Play the
                # received audio, including its final partial chunk, then listen.
                # Never retry automatically (including content-filter outcomes).
                self._finish_output()
            else:
                self._fail(
                    "Voice provider could not generate a reply; start a new call."
                )
        elif event.type == "error":
            if getattr(event.error, "code", None) != "response_cancel_not_active":
                self._fail(
                    "Voice provider returned an error; check API access and restart the call."
                )

    def _finish_output(self):
        if self.output_resampler is not None:
            self.sink.buffer.append(
                self.response_id, self.epoch, self.output_resampler.finish()
            )
        self.sink.buffer.finish(self.response_id, self.epoch)

    def _record_response_outcome(self, response):
        """Keep bounded operational codes; never retain provider messages/content."""
        status = (
            response.status
            if response.status in {"completed", "cancelled", "incomplete", "failed"}
            else "unknown"
        )
        details = getattr(response, "status_details", None)
        reason = getattr(details, "reason", None)
        if reason is not None and reason not in {
            "max_output_tokens",
            "content_filter",
            "turn_detected",
            "client_cancelled",
        }:
            reason = "unknown"
        code = getattr(getattr(details, "error", None), "code", None)
        if code is not None and (
            not isinstance(code, str)
            or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", code)
        ):
            code = "unknown"
        tokens = getattr(getattr(response, "usage", None), "output_tokens", None)
        outcome = {
            "status": status,
            "reason": reason,
            "error_code": code,
            "output_tokens": tokens
            if isinstance(tokens, int) and tokens >= 0
            else None,
        }
        self.response_counts[status] += 1
        self.response_outcomes.append(outcome)
        if status in {"incomplete", "failed", "unknown"}:
            logger.warning(
                "Voice response ended: status=%s reason=%s error_code=%s output_tokens=%s",
                status,
                reason,
                code,
                outcome["output_tokens"],
            )

    def diagnostics(self):
        return {
            "max_output_tokens": MAX_RESPONSE_TOKENS,
            "counts": dict(self.response_counts),
            "recent_responses": list(self.response_outcomes),
        }

    async def interrupt(self, *, provider_already_cancelled=False):
        async with self.control_lock:
            # Snapshot playback BEFORE clearing the buffer. Invalidate output before
            # awaiting provider I/O so late deltas cannot repopulate the old queue.
            played_ms = self.sink.estimated_played_ms()
            publication_bound = (
                self.sink.server_played_ms()
                if hasattr(self.sink, "server_played_ms")
                else played_ms
            )
            epoch = self.epoch
            response_id, item_id = self.response_id, self.item_id
            generating = self.generating
            self.response_id = None
            self.generating = False
            await self.sink.cancel_media()
            if epoch is not None and hasattr(self.sink, "interrupted_played_ms"):
                played_ms = await self.sink.interrupted_played_ms(
                    epoch, publication_bound
                )
            if not self.connection:
                return
            if generating and response_id and not provider_already_cancelled:
                await self.connection.response.cancel(response_id=response_id)
            if item_id:
                await self.connection.conversation.item.truncate(
                    item_id=item_id,
                    content_index=self.content_index,
                    audio_end_ms=played_ms,
                )
            self.item_id = None

    def timing(self, first_media_ms=None):
        """Server receipt intervals only; no transcripts or cross-clock subtraction."""

        def elapsed(start, end):
            return (
                round((end - start) * 1000, 2)
                if start is not None and end is not None and end >= start
                else None
            )

        first_published = (
            self.sink.request_time + first_media_ms / 1000
            if first_media_ms is not None
            and getattr(self.sink, "request_time", None) is not None
            else None
        )
        return {
            "scope": "server_monotonic_event_receipt_to_first_published_frame",
            "speech_stop_to_response_ms": elapsed(
                self.speech_stopped_at, self.response_created_at
            ),
            "response_to_first_audio_ms": elapsed(
                self.response_created_at, self.first_audio_at
            ),
            "first_audio_to_publish_ms": elapsed(self.first_audio_at, first_published),
            "speech_stop_to_publish_ms": elapsed(
                self.speech_stopped_at, first_published
            ),
            "note": "Excludes user-to-provider speech-end/VAD delay and downstream browser playback.",
        }

    async def close(self):
        self.closed = True
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        if self.context:
            with contextlib.suppress(Exception):
                await self.context.__aexit__(None, None, None)
        if self.client:
            await self.client.close()
        self.connection = None
