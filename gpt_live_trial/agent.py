"""Bridge the continuous GPT-Live WebSocket to Chert's paired A/V publisher."""

import asyncio
import base64
from collections import deque
import contextlib
import json
import re
import time

import numpy as np
from websockets.asyncio.client import connect

from realtime.audio import PCMResampler

MODEL = "gpt-live-1.1"
VOICE = "marin"
RATE = 24000
FRAME = RATE // 50
INSTRUCTIONS = (
    "You are Chert, a conversational AI on a casual English voice call. Let the caller speak first. "
    "Listen while speaking. A brief mm-hm usually means keep going. Yield when corrected. "
    "Respond directly in a calm conversational voice. Give enough detail for the question. "
    "For simple questions one sentence often works. Use earlier details from this call. "
    "Camera, web search and external tools are unavailable in this voice trial."
)
STOP = re.compile(r"^\s*(?:hey[, ]+)?(?:wait\b|stop\b|hold on\b|hang on\b)", re.I)


class LiveAgent:
    resolved_model = MODEL
    effective_voice = VOICE

    def __init__(self, settings, sink):
        self.settings, self.sink = settings, sink
        self.in_resampler = PCMResampler(48000, RATE)
        self.out_resampler = PCMResampler(RATE, 48000)
        self.input = asyncio.Queue(maxsize=75)
        self.tasks = []
        self.ws = None
        self.closed = False
        self.failure = None
        self.finalized = asyncio.Event()
        self.lock = asyncio.Lock()
        self.preroll = deque(maxlen=3)
        self.remainder = b""
        self.segment = None
        self.epoch = None
        self.quiet = 0
        self.muting = False
        self.synthesized_samples = 0
        self.active = False
        self.sent_samples = 0
        self.received_samples = 0
        self.turns = 0
        self.cuts = 0
        self.first_audio_at = None
        self.last_caller_end = None
        self.caller_id = None
        self.caller_text = ""
        self.caller_stopped = False
        self.caller_updated = 0.0
        self.pending_words = []
        self.words = ""
        self.peak_buffer_samples = 0

    async def send(self, kind, **fields):
        await asyncio.wait_for(self.ws.send(json.dumps({"type": kind, **fields})), 3)

    async def start(self):
        try:
            self.ws = await connect(
                "wss://api.openai.com/v1/live/sessions",
                additional_headers={
                    "Authorization": "Bearer "
                    + self.settings.openai_api_key.get_secret_value()
                },
                open_timeout=10,
                close_timeout=1,
                max_size=2**20,
                max_queue=32,
            )
            await self.send(
                "session.start",
                session=dict(
                    model=MODEL,
                    instructions=INSTRUCTIONS,
                    audio=dict(
                        format={"type": "audio/pcm", "rate": RATE},
                        output={"voice": VOICE},
                    ),
                    delegation={"type": "client"},
                ),
            )
            ready = json.loads(await asyncio.wait_for(self.ws.recv(), 12))
            if (
                ready.get("type") != "session.started"
                or ready["session"].get("model") != MODEL
            ):
                raise RuntimeError("GPT-Live session was not accepted")
            self.tasks = [
                asyncio.create_task(self._uplink()),
                asyncio.create_task(self._downlink()),
            ]
        except BaseException:
            if self.ws:
                await self.ws.close()
            raise

    def microphone(self, pcm):
        if self.closed or self.failure:
            return
        try:
            self.input.put_nowait(pcm)
        except asyncio.QueueFull:
            self._fail("Microphone uplink fell behind")

    async def _uplink(self):
        try:
            while not self.closed:
                frame = self.in_resampler.push(await self.input.get())
                if len(frame):
                    await self.send(
                        "session.input_audio.append",
                        audio=base64.b64encode(frame.tobytes()).decode(),
                    )
                    self.sent_samples += len(frame)
        except Exception:
            self._fail("GPT-Live microphone connection failed")

    def _fail(self, message):
        if self.closed or self.failure:
            return
        self.failure = self.sink.error = message
        self.sink.status = "error"
        self.tasks.append(asyncio.create_task(self._stop_failed()))

    async def _stop_failed(self):
        try:
            await self.sink.cancel_media(count=False)
        finally:
            self.sink.status = "error"
            if self.ws:
                await self.ws.close()

    async def _downlink(self):
        try:
            async for raw in self.ws:
                event = json.loads(raw)
                kind = event["type"]
                if kind == "session.closed":
                    self.finalized.set()
                    break
                if self.closed:
                    continue
                if kind == "error":
                    raise RuntimeError("Live session error")
                async with self.lock:
                    if kind == "session.output_audio.delta":
                        await self._audio(
                            base64.b64decode(event["delta"], validate=True)
                        )
                    elif kind == "session.input_transcript.delta":
                        await self._caller(event)
                    elif kind == "session.output_transcript.delta":
                        if not self.muting:
                            self._words(event["delta"])
                    elif kind == "session.delegation.created":
                        await self.send(
                            "session.commentary.append",
                            delegation_id=event["delegation"]["id"],
                            content="The private voice trial has no external tools available.",
                        )
            if not self.closed:
                self._fail("GPT-Live session ended")
        except Exception:
            self._fail("GPT-Live voice stream failed")

    async def _caller(self, event):
        start, end = event["start_ms"], event["end_ms"]
        if (
            self.caller_id is None
            or self.last_caller_end is None
            or start - self.last_caller_end > 800
        ):
            self._finish_caller()
            self.caller_id = f"live-user-{start}"
            self.caller_text = ""
            self.caller_stopped = False
            self.sink.transcripts.speech_started(self.caller_id, start)
        self.last_caller_end = end
        self.caller_updated = time.monotonic()
        self.caller_text += event["delta"]
        self.sink.transcripts.user(self.caller_id, self.caller_text.strip())
        if not self.caller_stopped and STOP.match(self.caller_text):
            self.caller_stopped = True
            if self.sink.response_active:
                await self._cut()

    def _finish_caller(self):
        if self.caller_id is not None:
            self.sink.transcripts.user(
                self.caller_id, self.caller_text.strip(), final=True
            )
            self.sink.transcripts.speech_stopped(self.caller_id, self.last_caller_end)
            self.caller_id = None

    def _words(self, piece):
        if self.active:
            self.words += piece
            self.sink.transcripts.cue(
                self.segment, self.synthesized_samples, self.words.strip()
            )
        else:
            self.pending_words.append(piece)

    async def _audio(self, data):
        if self.caller_id and time.monotonic() - self.caller_updated > 1.6:
            self._finish_caller()
        if self.segment and not self.active and not self.sink.response_active:
            self.sink.transcripts.finish(self.segment, self.synthesized_samples)
            self.segment = self.epoch = None
        if len(data) % 2:
            raise ValueError("Partial PCM sample")
        data = self.remainder + data
        frame_bytes = FRAME * 2
        end = len(data) // frame_bytes * frame_bytes
        self.remainder = data[end:]
        for offset in range(0, end, frame_bytes):
            block = data[offset : offset + frame_bytes]
            raw = np.frombuffer(block, "<i2").astype(np.float32)
            loud = np.mean(raw * raw) > 48**2
            samples = self.out_resampler.push(block)
            self.received_samples += FRAME
            if self.muting:
                self.quiet = 0 if loud else self.quiet + 1
                if self.quiet >= 8:
                    self.muting = False
                    self.quiet = 0
                continue
            if not self.active:
                self.preroll.append(samples)
                if not loud:
                    continue
                await self._begin()
                for frame in self.preroll:
                    self._append(frame)
                self.preroll.clear()
            else:
                self._append(samples)
            self.quiet = 0 if loud else self.quiet + 1
            if self.quiet >= 20:
                self.active = False
                self.sink.buffer.finish(self.segment, self.epoch)
                self.quiet = 0

    async def _begin(self):
        if (
            self.segment is not None
            and self.sink.response_active
            and self.sink.buffer.epoch == self.epoch
        ):
            # The previous words still occupy the A/V queue. Extend the same media epoch.
            self.sink.buffer.receiving = True
        else:
            self.turns += 1
            self.segment = f"gpt-live-{self.turns}"
            self.epoch = await self.sink.begin_voice_response(self.segment)
            self.sink.transcripts.begin(self.segment, self.epoch)
            self.first_audio_at = time.monotonic()
            self.synthesized_samples = 0
            self.words = ""
        self.active = True
        if self.pending_words:
            self._words("".join(self.pending_words))
            self.pending_words.clear()

    def _append(self, samples):
        if not self.sink.buffer.append(self.segment, self.epoch, samples):
            raise RuntimeError("Audio epoch changed")
        self.synthesized_samples += len(samples)
        self.peak_buffer_samples = max(
            self.peak_buffer_samples, len(self.sink.buffer.samples)
        )

    async def _cut(self):
        if self.epoch is not None:
            self.sink.transcripts.interrupt(
                self.epoch, int(self.sink.estimated_played_ms() * 48)
            )
        # Flush obsolete paired A/V. The upstream microphone never stops for a cut.
        self.muting = self.active
        self.active = False
        self.segment = self.epoch = None
        self.quiet = self.synthesized_samples = 0
        self.pending_words.clear()
        self.preroll.clear()
        self.cuts += 1
        await self.sink.cancel_media()

    async def interrupt(self):
        if self.closed or self.failure:
            return
        async with self.lock:
            await self._cut()
            await self.send(
                "session.instructions.append",
                delegation_id=None,
                content="Stop the current reply now and listen for the caller's next words.",
            )

    def diagnostics(self):
        return dict(
            model=MODEL,
            voice=VOICE,
            caller_samples_24khz=self.sent_samples,
            output_samples_24khz=self.received_samples,
            segments=self.turns,
            interrupted=self.cuts,
            peak_queued_audio_ms=round(self.peak_buffer_samples / 48),
            failed=bool(self.failure),
        )

    def timing(self, first_media_ms=None):
        return dict(first_media_ms=first_media_ms, source="GPT-Live continuous PCM")

    async def close(self):
        if self.closed:
            return
        self.closed = True
        self._finish_caller()
        try:
            if self.ws:
                with contextlib.suppress(Exception):
                    await self.send("session.close")
                    await asyncio.wait_for(self.finalized.wait(), 1)
        finally:
            for task in self.tasks:
                task.cancel()
            await asyncio.gather(*self.tasks, return_exceptions=True)
            if self.ws:
                await self.ws.close()
