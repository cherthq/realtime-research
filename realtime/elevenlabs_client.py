"""One private ElevenLabs socket per call, one cancellable context per reply.

Flash uses context-relative alignment; v3/v4 dialogue uses chunk-relative
alignment. Both become 24kHz sample positions on the reply's playback clock.
Delivery to this client is not evidence that the user heard the audio.
"""

import asyncio
import base64
import contextlib
import json
import re
from dataclasses import dataclass, field
from urllib.parse import urlencode
from uuid import uuid4

import httpx
from websockets.asyncio.client import connect


@dataclass
class SpeechChunk:
    pcm: bytes = b""
    cues: list[tuple[int, str]] = field(default_factory=list)


class ElevenLabsSpeechError(RuntimeError):
    """Safe public messages; never forward provider text or request content."""

    def __init__(self, code):
        messages = {
            "quota_exceeded": "ElevenLabs has no speech credits remaining. Top up the account, then start a new call. Test audio still works.",
            "no_audio": "ElevenLabs returned no speech audio. Check the account's credits and voice access, then start a new call.",
            "provider_error": "ElevenLabs rejected the speech request. Check the provider account and reconnect.",
        }
        self.code = code if code in messages else "provider_error"
        super().__init__(messages[self.code])


async def ensure_speech_credits(settings, *, transport=None):
    """Best-effort read before acquiring a GPU lease, never a paid generation.

    Subscription visibility can be restricted independently of TTS. Unknown
    status and enabled overage do not block an otherwise valid speech request.
    No negative cache: a top-up can be used on the very next call.
    """
    try:
        async with httpx.AsyncClient(timeout=3, transport=transport) as client:
            response = await client.get(
                "https://api.elevenlabs.io/v1/user/subscription",
                headers={"xi-api-key": settings.elevenlabs_api_key.get_secret_value()},
            )
            if response.status_code != 200:
                return
            data = response.json()
    except (httpx.HTTPError, ValueError):
        return
    if not isinstance(data, dict):
        return
    count, limit = data.get("character_count"), data.get("character_limit")
    if (
        type(count) is int
        and type(limit) is int
        and 0 <= limit <= count
        and data.get("can_extend_character_limit") is False
    ):
        raise ElevenLabsSpeechError("quota_exceeded")


class ElevenLabsTTSClient:
    KEEPALIVE_SECONDS = 10

    def __init__(self, settings, *, connector=connect, voice_id=None):
        self.settings, self.connector = settings, connector
        self.voice_id = (
            voice_id if voice_id is not None else settings.elevenlabs_voice_id
        )
        self.socket = self.reader = None
        self.contexts = {}
        self.lock = asyncio.Lock()
        self.closed = False
        self.dialogue = settings.elevenlabs_model.startswith(("eleven_v3", "eleven_v4"))
        self.heartbeat = None

    async def warm(self):
        async with self.lock:
            if self.closed:
                raise RuntimeError("Speech client is closed")
            if self.reader and not self.reader.done():
                return
            if self.heartbeat:
                self.heartbeat.cancel()
                await asyncio.gather(self.heartbeat, return_exceptions=True)
                self.heartbeat = None
            key = self.settings.elevenlabs_api_key.get_secret_value()
            voice = self.voice_id
            if not key or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", voice):
                raise ValueError("Configure ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID")
            params = dict(
                model_id=self.settings.elevenlabs_model,
                output_format="pcm_24000",
                sync_alignment="true",
            )
            if self.dialogue:
                endpoint = "text-to-dialogue/multi-stream-input"
            else:
                endpoint = f"text-to-speech/{voice}/multi-stream-input"
                params.update(inactivity_timeout=180, auto_mode="false")
            self.socket = await self.connector(
                f"wss://api.elevenlabs.io/v1/{endpoint}?{urlencode(params)}",
                additional_headers={"xi-api-key": key},
                open_timeout=15,
                max_size=16 * 1024 * 1024,
                ping_interval=20,
            )
            if self.dialogue:
                # TTD keepalives must name a registered context, including while
                # the user is listening. This context never receives speech text.
                idle_context = uuid4().hex
                try:
                    await self._send({"context_id": idle_context, "voices": [voice]})
                except BaseException:
                    await self.socket.close()
                    raise
                self.heartbeat = asyncio.create_task(
                    self._keep_connection_alive(self.socket, idle_context)
                )
            self.reader = asyncio.create_task(self._receive(self.socket))

    async def _keep_connection_alive(self, socket, ident):
        try:
            while True:
                await asyncio.sleep(self.KEEPALIVE_SECONDS)
                async with asyncio.timeout(10):
                    await socket.send(
                        json.dumps({"context_id": ident, "keep_alive": True})
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            # Wake any active response through the reader's disconnect handling.
            await socket.close()

    async def _send(self, body):
        # Keep send cancellation in this task. wait_for can swallow cancellation
        # when a fast send completes in the same event-loop tick on Python 3.11.
        async with asyncio.timeout(10):
            await self.socket.send(json.dumps(body))

    async def _receive(self, socket):
        failure = RuntimeError("ElevenLabs speech connection ended")
        try:
            async for raw in socket:
                data = json.loads(raw)
                ident = data.get("context_id", data.get("contextId"))
                if (
                    data.get("error")
                    or data.get("type") == "error"
                    or data.get("error_code")
                ):
                    # Provider messages may contain input text. Never log them.
                    details = data.get("error")
                    details = details if isinstance(details, dict) else {}
                    code = (
                        details.get("code")
                        or details.get("status")
                        or data.get("error_code")
                        or data.get("code")
                    )
                    raise ElevenLabsSpeechError(
                        "quota_exceeded"
                        if code == "quota_exceeded"
                        else "provider_error"
                    )
                queue = self.contexts.get(ident)
                if queue is not None:
                    queue.put_nowait(data)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if isinstance(exc, ElevenLabsSpeechError):
                failure = exc
        finally:
            for queue in list(self.contexts.values()):
                if queue.full():
                    queue.get_nowait()
                queue.put_nowait(failure)
            await socket.close()

    async def stream_response(self, phrases):
        await self.warm()
        ident = uuid4().hex
        queue = self.contexts[ident] = asyncio.Queue(maxsize=128)
        finished_input = False
        sent_text = []

        async def send():
            nonlocal finished_input
            voice = self.voice_id
            opening = {"voices": [voice]} if self.dialogue else {"text": " "}
            if self.settings.elevenlabs_stability is not None:
                opening["voice_settings"] = {
                    "stability": self.settings.elevenlabs_stability
                }
            await self._send({"context_id": ident, **opening})
            async for text in phrases:
                if not text.strip():
                    continue
                sent_text.append(text)
                payload = (
                    {"inputs": [{"text": text + " ", "voice_id": voice}]}
                    if self.dialogue
                    else {"text": text + " "}
                )
                # Keep one prosody turn across phrases; do not set new_turn for
                # each sentence. Flush short replies instead of waiting for size.
                await self._send({"context_id": ident, **payload, "flush": True})
            # Mark closing before awaiting the send so a heartbeat or cancellation
            # cannot send another message to a context the server is closing.
            finished_input = True
            await self._send({"context_id": ident, "close_context": True})

        async def keepalive():
            while True:
                await asyncio.sleep(self.KEEPALIVE_SECONDS)
                if not finished_input:
                    payload = {"keep_alive": True} if self.dialogue else {"text": ""}
                    await self._send({"context_id": ident, **payload})

        sender, heartbeat = (
            asyncio.create_task(send()),
            asyncio.create_task(keepalive()),
        )

        def failed(task):
            if not task.cancelled() and task.exception() is not None:
                if queue.full():
                    queue.get_nowait()
                queue.put_nowait(RuntimeError("ElevenLabs speech request failed"))

        sender.add_done_callback(failed)
        heartbeat.add_done_callback(failed)
        aligned_text, samples, last_cue = "", 0, 0
        pending_cues = []
        try:
            while True:
                if sender.done():
                    sender.result()
                data = await asyncio.wait_for(queue.get(), 30)
                if isinstance(data, Exception):
                    raise data
                pcm = base64.b64decode(data.get("audio") or "", validate=True)
                if len(pcm) % 2:
                    raise ValueError("Invalid PCM from speech provider")
                offset = samples if self.dialogue else 0
                samples += len(pcm) // 2
                alignment = (
                    data.get("normalized_alignment")
                    or data.get("normalizedAlignment")
                    or data.get("alignment")
                    or {}
                )
                chars = alignment.get("chars", [])
                starts = alignment.get(
                    "charStartTimesMs",
                    alignment.get(
                        "char_start_times_ms", alignment.get("charsStartTimesMs", [])
                    ),
                )
                durations = alignment.get(
                    "charDurationsMs",
                    alignment.get(
                        "char_durations_ms", alignment.get("charsDurationsMs", [])
                    ),
                )
                cues = []
                if len(chars) == len(starts) == len(durations):
                    for char, start, duration in zip(chars, starts, durations):
                        aligned_text += char
                        if char.isspace() or char in ".!?":
                            # Count ALL preceding PCM, including audio-only frames.
                            # TTD timing restarts at zero in each aligned chunk.
                            last_cue = max(
                                last_cue, offset + round((start + duration) * 24)
                            )
                            pending_cues.append((last_cue, aligned_text.strip()))
                final = data.get("isFinal", data.get("is_final", False))
                while pending_cues and pending_cues[0][0] <= samples:
                    cues.append(pending_cues.pop(0))
                if final:
                    if sent_text and samples == 0:
                        raise ElevenLabsSpeechError("no_audio")
                    # With no word alignment, expose the completed utterance only
                    # after its final audio has played, never guess word timings.
                    cues.append((samples, aligned_text.strip() or " ".join(sent_text)))
                yield SpeechChunk(pcm, cues)
                if final:
                    await sender
                    return
        finally:
            # Unregister BEFORE closing: the service may flush old audio on close.
            self.contexts.pop(ident, None)
            sender.cancel()
            heartbeat.cancel()
            await asyncio.gather(sender, heartbeat, return_exceptions=True)
            if not finished_input and self.socket:
                with contextlib.suppress(Exception):
                    await self._send({"context_id": ident, "close_context": True})

    async def close(self):
        self.closed = True
        if self.heartbeat:
            self.heartbeat.cancel()
            await asyncio.gather(self.heartbeat, return_exceptions=True)
        if self.reader:
            self.reader.cancel()
            await asyncio.gather(self.reader, return_exceptions=True)
        if self.socket:
            await self.socket.close()
