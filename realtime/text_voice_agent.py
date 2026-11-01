"""Provider-neutral text conversation with streamed Qwen or ElevenLabs speech."""

import asyncio
import contextlib
import re
import time

from openai import AsyncOpenAI

from .audio import PCMResampler
from .qwen_client import QwenTTSClient
from .elevenlabs_client import ElevenLabsTTSClient, ElevenLabsSpeechError
from .voice_agent import MAX_RESPONSE_TOKENS, VoiceAgent
from .search_turn import SearchTurn


class TextPhrases:
    """Keep sentence/clause context while allowing speech before text completes."""

    def __init__(self):
        self.pending = ""

    def push(self, text, *, final=False):
        self.pending += text
        result = []
        while self.pending:
            end = next(
                (
                    m.end()
                    for m in re.finditer(r"[.!?;\n](?:\s|$)|[,：:](?=\s)", self.pending)
                    if m.end() >= 24
                ),
                None,
            )
            if len(self.pending) > 180 and (end is None or end > 180):
                end = self.pending.rfind(" ", 30, 180)
                if end < 0:
                    end = 180
            if end is None:
                if not final:
                    break
                end = len(self.pending)
            phrase, self.pending = self.pending[:end].strip(), self.pending[end:]
            if phrase:
                result.append(phrase)
        return result


async def speech_text(phrases):
    """Coalesce text already available; never wait to fill a synthesis request.

    Text generation often finishes while the previous phrase is being spoken.
    Sending each queued clause separately then repeats prefill and starves the
    avatar of audio. Preserve order, the finish sentinel and the 180-char bound.
    """
    carry = ""
    while True:
        text = carry or await phrases.get()
        carry = ""
        if text is None:
            return
        finished = False
        while len(text) < 180:
            try:
                extra = phrases.get_nowait()
            except asyncio.QueueEmpty:
                break
            if extra is None:
                finished = True
                break
            if len(text) + 1 + len(extra) > 180:
                carry = extra
                break
            text += " " + extra
        yield text
        if finished:
            return


class TextVoiceAgent(VoiceAgent):
    def __init__(self, settings, profile, sink, client=None, tts=None, search=None):
        super().__init__(settings, profile, sink, client)
        self.tts = tts or (
            ElevenLabsTTSClient(settings, voice_id=profile["voice"])
            if settings.tts_provider == "elevenlabs"
            else QwenTTSClient(settings)
        )
        self.tts_task = None
        self.phrases = TextPhrases()
        self.text_queue = None
        self.spoken_spans = []
        self.synthesized_samples = 0
        self.text_finished = False
        self.first_text_at = None
        self.first_tts_started_at = None
        self.tts_requests = 0
        self.first_tts_characters = 0
        self.lookup = SearchTurn(self, search)

    def instructions(self):
        return super().instructions() + self.lookup.instructions()

    def _begin_captions(self):
        super()._begin_captions()
        self.lookup.attach_sources(self.response_id)

    async def start(self):
        await self.tts.warm()
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
                "output_modalities": ["text"],
                "max_output_tokens": MAX_RESPONSE_TOKENS,
                "instructions": self.instructions(),
                "audio": {"input": self.input_options()},
                **self.lookup.session_options(),
            }
        )
        self.tasks = [
            asyncio.create_task(self._receive()),
            asyncio.create_task(self._send_microphone()),
        ]
        await asyncio.wait_for(self.ready.wait(), 15)
        if self.failure:
            raise RuntimeError(self.failure)

    async def handle(self, event):
        if await self._shared_event(event):
            return
        if event.type == "session.updated":
            if list(event.session.output_modalities) != ["text"]:
                self._fail("Conversation service did not enable text-only replies.")
            self.effective_voice = self.profile["voice"]
            self.resolved_model = (
                self.settings.elevenlabs_model
                if self.settings.tts_provider == "elevenlabs"
                else "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
            )
            self.ready.set()
        elif event.type == "input_audio_buffer.speech_started":
            self.speech_stopped_at = self.response_created_at = self.first_audio_at = (
                None
            )
            self.first_text_at = self.first_tts_started_at = None
            self.tts_requests = self.first_tts_characters = 0
            await self.interrupt(provider_already_cancelled=True)
        elif event.type == "input_audio_buffer.speech_stopped":
            self.speech_stopped_at = time.monotonic()
        elif event.type == "response.created":
            async with self.control_lock:
                self.response_created_at, self.first_audio_at = time.monotonic(), None
                if self.tts_task and not self.tts_task.done():
                    raise ValueError("Unexpected overlapping TTS response")
                self.response_id, self.item_id = event.response.id, None
                self.generating, self.text_finished = True, False
                self.phrases = TextPhrases()
                self.text_queue = asyncio.Queue()
                self.spoken_spans, self.synthesized_samples = [], 0
                self.first_text_at = self.first_tts_started_at = None
                self.tts_requests = self.first_tts_characters = 0
                self.output_resampler = PCMResampler(24000, 48000)
                self.epoch = await self.sink.begin_voice_response(self.response_id)
                self._begin_captions()
                self.tts_task = asyncio.create_task(
                    self._speak(self.response_id, self.epoch, self.text_queue)
                )
        elif (
            event.type == "response.output_text.delta"
            and event.response_id == self.response_id
        ):
            if not self.text_finished:
                if self.first_text_at is None:
                    self.first_text_at = time.monotonic()
                self.item_id = event.item_id
                for phrase in self.phrases.push(event.delta):
                    self.text_queue.put_nowait(phrase)
        elif (
            event.type == "response.output_text.done"
            and event.response_id == self.response_id
        ):
            # With tools enabled, response.done can still include a function call.
            # Keep its queue open until then so a tool-only reply can speak its
            # acknowledgement. Already emitted phrases continue streaming.
            if not self.lookup.enabled:
                self._finish_text()
        elif event.type == "response.done" and event.response.id == self.response_id:
            self.generating = False
            self._record_response_outcome(event.response)
            if event.response.status in {"completed", "cancelled", "incomplete"}:
                self.lookup.start(event.response)
                self._finish_text()
            else:
                self._fail("Conversation provider could not generate a reply.")
        elif event.type == "error":
            if getattr(event.error, "code", None) != "response_cancel_not_active":
                self._fail(
                    "Conversation provider returned an error; reconnect the call."
                )

    def _finish_text(self):
        if not self.text_finished:
            self.text_finished = True
            for phrase in self.phrases.push("", final=True):
                self.text_queue.put_nowait(phrase)
            self.text_queue.put_nowait(None)

    async def _speak(self, response_id, epoch, phrases):
        try:
            texts = speech_text(phrases)
            first = await anext(texts, None)
            if first is None:
                # Tool-only replies have no speech. Do not open an empty TTS
                # context or wait for its provider to acknowledge a close.
                if response_id == self.response_id and epoch == self.epoch:
                    self._finish_output()
                return

            async def available_text():
                yield first
                async for text in texts:
                    yield text

            if hasattr(self.tts, "stream_response"):

                async def inputs():
                    async for text in available_text():
                        self._tts_started(text)
                        yield text

                async for chunk in self.tts.stream_response(inputs()):
                    if response_id != self.response_id or epoch != self.epoch:
                        return
                    self._append_audio(response_id, epoch, chunk.pcm)
                    for sample, text in chunk.cues:
                        self._caption(sample * 2, text)
                        self.spoken_spans.append((sample * 2, text))
            else:
                completed_text = []
                async for text in available_text():
                    self._tts_started(text)
                    async for pcm in self.tts.stream(
                        text,
                        self.profile["voice"],
                        self.profile.get("instructions", ""),
                    ):
                        if response_id != self.response_id or epoch != self.epoch:
                            return
                        self._append_audio(response_id, epoch, pcm)
                    completed_text.append(text)
                    self.spoken_spans.append((self.synthesized_samples, text))
                    self._caption(self.synthesized_samples, " ".join(completed_text))
            if response_id == self.response_id and epoch == self.epoch:
                self._finish_output()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.tts_error_type = type(exc).__name__
            self.tts_error_code = (
                exc.code if isinstance(exc, ElevenLabsSpeechError) else None
            )
            self.tts_http_status = getattr(
                getattr(exc, "response", None), "status_code", None
            )
            self._fail(
                str(exc)
                if isinstance(exc, ElevenLabsSpeechError)
                else f"{self.settings.tts_provider.title()} speech failed; check provider configuration and reconnect."
            )

    def _tts_started(self, text):
        self.tts_requests += 1
        if self.first_tts_started_at is None:
            self.first_tts_started_at = time.monotonic()
            self.first_tts_characters = len(text)

    def _append_audio(self, response_id, epoch, pcm):
        if not pcm:
            return
        if self.first_audio_at is None:
            self.first_audio_at = time.monotonic()
        samples = self.output_resampler.push(pcm)
        self.synthesized_samples += len(samples)
        self.sink.buffer.append(response_id, epoch, samples)

    async def interrupt(self, *, provider_already_cancelled=False):
        self.turn_generation += 1
        search_task = self.lookup.cancel()
        if self.turn_task:
            self.turn_task.cancel()
        async with self.control_lock:
            epoch, item_id, response_id = self.epoch, self.item_id, self.response_id
            generating = self.generating
            bound = (
                self.sink.server_played_ms()
                if hasattr(self.sink, "server_played_ms")
                else self.sink.estimated_played_ms()
            )
            self.response_id, self.item_id, self.generating = None, None, False
            task, self.tts_task = self.tts_task, None
            if task:
                task.cancel()
            await self.sink.cancel_media()
            if task:
                await asyncio.gather(task, return_exceptions=True)
            if search_task:
                await asyncio.gather(search_task, return_exceptions=True)
            played_ms = (
                await self.sink.interrupted_played_ms(epoch, bound)
                if epoch is not None
                and bound > 0
                and hasattr(self.sink, "interrupted_played_ms")
                else min(bound, self.sink.estimated_played_ms())
            )
            self._interrupt_captions(epoch, played_ms)
            if not self.connection:
                return
            if generating and response_id and not provider_already_cancelled:
                await self.connection.response.cancel(response_id=response_id)
            if item_id:
                # Text items cannot be audio-truncated. Retain only fully heard
                # phrases, using confirmed browser playback, not generated text.
                heard = " ".join(
                    text for end, text in self.spoken_spans if end <= played_ms * 48
                )
                if hasattr(self.tts, "stream_response"):
                    heard = next(
                        (
                            text
                            for end, text in reversed(self.spoken_spans)
                            if end <= played_ms * 48
                        ),
                        "",
                    )
                await self.connection.conversation.item.delete(item_id=item_id)
                if heard:
                    await self.connection.conversation.item.create(
                        item={
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": heard}],
                        }
                    )

    def diagnostics(self):
        def elapsed(start, end):
            return (
                round((end - start) * 1000, 2)
                if start is not None and end is not None and end >= start
                else None
            )

        return {
            **super().diagnostics(),
            "tts_provider": self.settings.tts_provider,
            "web_search": self.lookup.diagnostics(),
            "tts_error_type": getattr(self, "tts_error_type", None),
            "tts_error_code": getattr(self, "tts_error_code", None),
            "tts_http_status": getattr(self, "tts_http_status", None),
            "tts_busy_retries": getattr(self.tts, "busy_retries", 0),
            "conversation_model": self.settings.openai_realtime_model,
            "history_policy": "retain_fully_heard_phrases_on_interruption",
            "tts_requests": self.tts_requests,
            "first_tts_characters": self.first_tts_characters,
            "text_speech_timing": {
                "response_to_first_text_ms": elapsed(
                    self.response_created_at, self.first_text_at
                ),
                "first_text_to_tts_request_ms": elapsed(
                    self.first_text_at, self.first_tts_started_at
                ),
                "tts_request_to_first_audio_ms": elapsed(
                    self.first_tts_started_at, self.first_audio_at
                ),
            },
        }

    async def close(self):
        self.closed = True
        await self.lookup.close()
        if self.tts_task:
            self.tts_task.cancel()
            await asyncio.gather(self.tts_task, return_exceptions=True)
        with contextlib.suppress(Exception):
            await self.tts.close()
        await super().close()
