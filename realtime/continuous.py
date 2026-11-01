"""One native video timeline across listening, speaking and interruption.

Response positions remain separate from model positions. Silence fills listening
intervals but is never counted as speech playback. No loop or dissolve is used.
"""

from dataclasses import dataclass
from collections import Counter
import asyncio
import contextlib
import time

import numpy as np

from .contracts import AudioChunk, FRAME_SAMPLES, HEIGHT, SAMPLE_RATE, WIDTH
from .playback import tag
from .streaming import GeneratedBatch, chunk_frames, LOOKAHEAD_SAMPLES


@dataclass(frozen=True)
class ContinuousBatch:
    batch: GeneratedBatch
    response_start: int
    speech_samples: int

    def speech_at(self, frame: int, epoch: int) -> bool:
        return (
            self.batch.audio.epoch == epoch
            and frame * FRAME_SAMPLES < self.speech_samples
        )


class ContinuousTimeline:
    def __init__(self):
        self.position = 0

    def take(self, buffer, *, minimum_speech_samples=0):
        count = chunk_frames(self.position) * FRAME_SAMPLES
        available = min(count, len(buffer.samples))
        if buffer.receiving and len(buffer.samples) < max(
            count + LOOKAHEAD_SAMPLES, minimum_speech_samples
        ):
            available = 0  # Never splice invented silence into unfinished speech.
        # Keep response frame positions aligned while more PCM is arriving.
        # Only an utterance's final partial frame may contain trailing silence.
        if buffer.receiving:
            available -= available % FRAME_SAMPLES
        start = buffer.position
        pcm = np.zeros(count, dtype=np.int16)
        pcm[:available] = buffer.samples[:available]
        lookahead = np.zeros(LOOKAHEAD_SAMPLES, dtype=np.int16)
        if available:
            future = buffer.samples[count : count + LOOKAHEAD_SAMPLES]
            lookahead[: len(future)] = future
        buffer.samples = buffer.samples[available:]
        buffer.position += available
        audio = AudioChunk(
            buffer.response_id if available else "idle",
            buffer.epoch,
            self.position,
            pcm,
            count,
            lookahead,
        )
        self.position += count
        return audio, start, available


class ContinuousRunner:
    def __init__(self, worker):
        self.worker = worker
        self.timeline = ContinuousTimeline()
        self.published = 0
        self.compute_ema = self._compute_seed()
        self.startup_buffer_seconds = (
            getattr(getattr(worker, "settings", None), "avatar_startup_buffer_ms", 480)
            / 1000
        )
        self.startup_buffer_ms = 0.0
        self.transitions = 0
        self.previous_speaking = False
        self.speech_underflows = 0
        self.pending_speech = Counter()
        self.roundtrip_seconds = 0.0
        self.speech_preroll_frames = 0
        self.hold_reasons = Counter()
        self.consecutive_holds = 0
        self.max_consecutive_holds = 0
        self.waiting_for_audio = False

    def _compute_seed(self):
        gpu = getattr(self.worker, "gpu", None)
        info = getattr(gpu, "info", None) or {}
        memory = getattr(gpu, "last_memory", None) or info.get("memory", {})
        stages = memory.get("last_stage_ms", {})
        duration = sum(stages.values()) / 1000 + 0.04 if stages else 1.28
        # Compilation can be included in the startup CUDA-event measurement.
        # Do not turn that outlier into several seconds of queued idle video.
        return max(0.3, duration) if duration <= 2.56 else 1.28

    async def generate(self):
        w = self.worker
        # The resident GPU lease is acquired AFTER the runner is constructed.
        # Read its warmed timing now, when it is available, rather than letting
        # the unusually cheap first history-free chunk seed steady-state timing.
        self.compute_ema = self._compute_seed()
        hash_task = None
        try:
            while not w.closed and not w.error:
                w.generation_ready.clear()
                # Generate the next chunk just before it is needed, rather than
                # accumulating seconds of idle video ahead of a new spoken reply.
                lead = (self.timeline.position - self.published) / SAMPLE_RATE
                lookahead = min(2.56, max(0.3, self.compute_ema + 0.48))
                if w.buffer.active:
                    lookahead = max(lookahead, min(2.56, 2 * self.compute_ema + 0.16))
                if w.batches.full() or lead > lookahead:
                    try:
                        await asyncio.wait_for(w.generation_ready.wait(), 0.04)
                    except asyncio.TimeoutError:
                        pass
                    continue
                count = chunk_frames(self.timeline.position) * FRAME_SAMPLES
                needed = count + LOOKAHEAD_SAMPLES
                if (
                    w.buffer.position == 0
                    and getattr(w, "voice", None) is not None
                    and getattr(getattr(w, "settings", None), "tts_provider", None)
                    == "qwen"
                ):
                    # Qwen delivers codec batches and can pause between phrases.
                    # Enter speech with two native chunks plus lookahead, rather
                    # than exhausting the first chunk while the next is missing.
                    # Continue real listening motion while this fills. A known
                    # completed short reply bypasses the reserve in take().
                    needed += (
                        chunk_frames(self.timeline.position + count) * FRAME_SAMPLES
                    )
                if w.buffer.receiving and len(w.buffer.samples) < needed:
                    # Keep generating listening motion until the first speech
                    # chunk is ready. Once speech starts, never insert idle PCM
                    # inside it. If enough listening video is queued, give TTS
                    # the GPU until that reserve approaches one render time.
                    self.waiting_for_audio = True
                    if w.buffer.position == 0 and lead <= self.compute_ema + 0.32:
                        pass  # take() preserves partial speech and renders silence
                    else:
                        w.generation_ready.clear()
                        try:
                            await asyncio.wait_for(w.generation_ready.wait(), 0.04)
                        except asyncio.TimeoutError:
                            pass
                        continue
                else:
                    self.waiting_for_audio = False
                audio, start, speech = self.timeline.take(
                    w.buffer, minimum_speech_samples=needed
                )
                w.generating_epoch = audio.epoch if speech else None
                if speech and start == 0:
                    w.first_chunk_started_at = time.monotonic()
                requested = time.monotonic()
                batch = await w.gpu.generate(audio, continuous=True)
                roundtrip = time.monotonic() - requested
                self.roundtrip_seconds += roundtrip
                w.generating_epoch = None
                if batch is None:
                    if w.closed:
                        return
                    raise RuntimeError("Continuous generation unexpectedly cancelled")
                if speech and start == 0 and audio.epoch == w.buffer.epoch:
                    w.first_chunk_decoded_at = time.monotonic()
                # Keep model history continuous. Publication discards cancelled
                # speaking frames instead of showing silent lip movements.
                if speech:
                    self.pending_speech[audio.epoch] += 1
                await w.batches.put(ContinuousBatch(batch, start, speech))
                # The first two chunks have less history and are unusually
                # cheap. Their timing cannot predict the steady-state deadline.
                if audio.sample_start >= 57 * FRAME_SAMPLES:
                    self.compute_ema = 0.7 * self.compute_ema + 0.3 * roundtrip
                w.generated_frames += len(batch.frames)
                w.generation_seconds += batch.compute_seconds
                # One bounded diagnostic task overlaps the NEXT GPU chunk.
                # Awaiting a full RGB scan here serially cost ~35 ms per chunk
                # on the demo host, consuming its small realtime margin.
                if hash_task is not None:
                    await hash_task
                hash_task = asyncio.create_task(self._record_hashes(batch.frames))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            w.error = f"Continuous avatar generation failed ({type(exc).__name__}). Check the GPU log."
            w.status = "error"
        finally:
            w.generating_epoch = None
            if hash_task is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    await hash_task

    async def _record_hashes(self, frames):
        hashes = await asyncio.to_thread(self.worker._hash_frames, frames)
        self.worker.frame_hashes.update(hashes)

    async def publish(self):
        from PIL import Image, ImageOps
        from livekit import rtc

        w = self.worker
        with Image.open(w.catalog.image(w.avatar_id)) as image:
            last_frame = np.asarray(ImageOps.fit(image.convert("RGB"), (WIDTH, HEIGHT)))
        silent = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        started = False
        try:
            while not w.closed and not w.error:
                if not started:
                    # Build a small reserve ONCE, before publishing any media.
                    # The first native batch lasts 1s; later batches often need
                    # more than 1s to render. Starting immediately drains the
                    # queue at that first boundary and carries the deficit into
                    # speech. No per-reply freeze, retiming or history reset.
                    try:
                        item = await asyncio.wait_for(w.batches.get(), 0.1)
                    except asyncio.TimeoutError:
                        continue
                    if w.batches.empty() and self.startup_buffer_seconds:
                        waiting = time.monotonic()
                        await asyncio.sleep(self.startup_buffer_seconds)
                        self.startup_buffer_ms = (time.monotonic() - waiting) * 1000
                    started = True
                else:
                    try:
                        item = w.batches.get_nowait()
                    except asyncio.QueueEmpty:
                        item = None
                if item is None:
                    w.generation_ready.set()
                frames = item.batch.frames if item else [last_frame]
                for index, rgb in enumerate(frames):
                    if w.closed:
                        return
                    epoch = w.buffer.epoch
                    if item and item.speech_samples and item.batch.audio.epoch != epoch:
                        # Skip the unplayed tail, including its padded silence.
                        # Preserve the model's timeline, but never relabel a
                        # cancelled talking pose as current listening motion.
                        self.published += (len(frames) - index) * FRAME_SAMPLES
                        w.generation_ready.set()
                        break
                    speaking = bool(item and item.speech_at(index, epoch))
                    w.publishing_epoch = epoch if speaking else None
                    offset = index * FRAME_SAMPLES
                    position = item.response_start + offset if speaking else 0
                    pcm = (
                        item.batch.audio.pcm[offset : offset + FRAME_SAMPLES]
                        if speaking
                        else silent
                    )
                    frame = rtc.VideoFrame(
                        WIDTH, HEIGHT, rtc.VideoBufferType.RGB24, rgb.tobytes()
                    )
                    tag(frame, epoch, position, speaking)
                    await asyncio.wait_for(w.sync.push_pair(frame, pcm), 2)
                    last_frame = rgb
                    w.frames += 1
                    if item:
                        self.consecutive_holds = 0
                    else:
                        self.consecutive_holds += 1
                        self.max_consecutive_holds = max(
                            self.max_consecutive_holds, self.consecutive_holds
                        )
                        self.hold_reasons[
                            "initial_render"
                            if not self.published
                            else "waiting_for_audio"
                            if self.waiting_for_audio
                            else "render_deadline"
                        ] += 1
                    if speaking and epoch == w.buffer.epoch:
                        w.speech_frames += 1
                        w.published_response_samples += min(
                            FRAME_SAMPLES, item.speech_samples - offset
                        )
                        w.playing_epoch = epoch
                        w.last_response_publish = time.monotonic()
                        w.response_audio_until = (
                            w.last_response_publish + w.audio.queued_duration
                        )
                        w.status = "responding"
                        if w.first_chunk_published_at is None:
                            w.first_chunk_published_at = time.monotonic()
                        if w.first_media_ms is None and w.request_time is not None:
                            w.first_media_ms = (
                                time.monotonic() - w.request_time
                            ) * 1000
                    else:
                        if item:
                            w.idle_motion_frames += 1
                        else:
                            w.held_frames += 1
                            if w.published_response_samples and (
                                w.buffer.active or w.generating_epoch == epoch
                            ):
                                self.speech_underflows += 1
                        w.status = "buffering" if w.buffer.active else "listening"
                    if speaking != self.previous_speaking:
                        self.transitions += 1
                        self.previous_speaking = speaking
                    if item:
                        self.published += FRAME_SAMPLES
                    w.generation_ready.set()
                w.publishing_epoch = None
                if item and item.speech_samples:
                    epoch = item.batch.audio.epoch
                    self.pending_speech[epoch] -= 1
                    if not self.pending_speech[epoch]:
                        del self.pending_speech[epoch]
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            w.error = f"Continuous avatar publication failed ({type(exc).__name__})."
            w.status = "error"
        finally:
            w.publishing_epoch = None
