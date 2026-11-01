from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import time
from datetime import timedelta

import numpy as np
from livekit import api, rtc

from .catalog import Catalog
from .config import ROOT, Settings
from .contracts import (
    FPS,
    FRAME_SAMPLES,
    HEIGHT,
    LIVE_VOICE_PENDING_SECONDS,
    PUBLISH_FPS,
    SAMPLE_RATE,
    WIDTH,
    ResponseBuffer,
)
from .gpu_client import GPUProcess
from .idle import IdlePlayer
from .mock import MockRenderer
from .playback import PlaybackLedger, TaggedVideoSource, tag
from .streaming import take_batch
from .voice_agent import VoiceAgent
from .text_voice_agent import TextVoiceAgent
from .camera import CameraInput
from .transcripts import TranscriptLedger
from .continuous import ContinuousRunner
from .media_clock import AudioClock


def room_token(
    settings: Settings, room: str, identity: str, *, worker: bool = False
) -> str:
    return (
        api.AccessToken(
            settings.livekit_api_key, settings.livekit_api_secret.get_secret_value()
        )
        .with_identity(identity)
        .with_name("Avatar worker" if worker else "You")
        .with_ttl(timedelta(seconds=settings.session_limit_seconds + 120))
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=worker,
                can_publish_sources=["camera", "microphone"]
                if worker or settings.camera_vision
                else ["microphone"],
            )
        )
        .to_jwt()
    )


class DemoWorker:
    """Resident room publisher. The mock renderer is replaceable without changing transport."""

    def __init__(
        self,
        session_id: str,
        avatar_id: str,
        settings: Settings,
        catalog: Catalog,
        voice_selection: dict | None = None,
        model_id: str = "mock",
        gpu_pool=None,
    ):
        self.id = session_id
        self.room_name = f"chert-{session_id}"
        self.settings = settings
        self.catalog = catalog
        self.model_id = model_id
        self.continuous_motion = (
            settings.avatar_continuous_motion and model_id == "avatarforever"
        )
        self.continuous_runner = (
            ContinuousRunner(self) if self.continuous_motion else None
        )
        self.gpu_pool = gpu_pool
        self.renderer = (
            MockRenderer(catalog.image(avatar_id)) if model_id == "mock" else None
        )
        self.gpu = (
            GPUProcess(
                [
                    str(settings.avatarforever_python),
                    "-m",
                    "realtime.gpu.server",
                    "--weights",
                    str(settings.avatarforever_weights),
                    "--image",
                    str(catalog.image(avatar_id)),
                    *(
                        ["--compile-blocks"]
                        if settings.avatarforever_compile_blocks
                        else []
                    ),
                    *(
                        ["--cache-prompt"]
                        if settings.avatarforever_cache_prompt
                        else []
                    ),
                ],
                ROOT / "runs/realtime/sessions" / session_id / "gpu.log",
                load_timeout=settings.gpu_load_timeout_seconds,
                chunk_timeout=settings.gpu_chunk_timeout_seconds,
            )
            if model_id == "avatarforever" and gpu_pool is None
            else None
        )
        self.avatar_id = avatar_id
        self.batches = asyncio.Queue(maxsize=2)
        self.generation_ready = asyncio.Event()
        self.generate_task = None
        self.generated_frames = 0
        self.frame_hashes: set[bytes] = set()
        self.generation_seconds = 0.0
        self.control_task = None
        self.caption_task = None
        self.held_frames = 0
        self.idle_motion_frames = 0
        self.first_chunk_started_at = None
        self.first_chunk_decoded_at = None
        self.first_chunk_published_at = None
        self.generating_epoch = None
        self.publishing_epoch = None
        self.playing_epoch = None
        self.last_response_publish = None
        self.response_audio_until = 0.0
        self.stalls = 0
        self.buffer = ResponseBuffer(
            max_pending_seconds=LIVE_VOICE_PENDING_SECONDS if voice_selection else 20,
            on_change=self.generation_ready.set,
        )
        self.playback = PlaybackLedger()
        self.transcripts = TranscriptLedger()
        self.camera = CameraInput()
        self.state_changed = asyncio.Event()
        self.room = rtc.Room()
        self.audio = rtc.AudioSource(SAMPLE_RATE, 1, queue_size_ms=100)
        self.video = TaggedVideoSource(
            WIDTH,
            HEIGHT,
            self.playback,
            lambda: self.buffer.epoch,
            preserve_stale=self.continuous_motion,
        )
        self.sync = (
            AudioClock(self.audio, self.video, lambda: self.buffer.epoch)
            if self.continuous_motion or self.model_id == "mock"
            else rtc.AVSynchronizer(
                audio_source=self.audio,
                video_source=self.video,
                video_fps=FPS,
                video_queue_size_ms=80,
            )
        )
        self.media_task: asyncio.Task | None = None
        self.mic_tasks: set[asyncio.Task] = set()
        self.closed = False
        self._closing: asyncio.Task | None = None
        self.connected_user = False
        self.last_presence = time.monotonic()
        self.started = time.monotonic()
        self.frames = 0
        self.speech_frames = 0
        self.mic_rms = 0.0
        self.mic_packets = 0
        self.interruptions = 0
        self.last_cancel_ms: float | None = None
        self.first_media_ms: float | None = None
        self.request_time: float | None = None
        self.error: str | None = None
        self.status = "preparing"
        self.last_scenario: str | None = None
        self.voice = (
            (
                TextVoiceAgent
                if settings.tts_provider in {"qwen", "elevenlabs"}
                else VoiceAgent
            )(settings, voice_selection["profile"], self)
            if voice_selection
            else None
        )
        self.published_response_samples = 0

    async def start(self):
        if self.gpu_pool:
            self.gpu = await self.gpu_pool.acquire(
                self.id, self.catalog.image(self.avatar_id)
            )
        elif self.gpu:
            await self.gpu.start()
            await self.gpu.open_session(self.id, self.catalog.image(self.avatar_id))
            if self.settings.avatar_idle_motion and not self.continuous_motion:
                await self.gpu.prepare_idle(self.catalog.image(self.avatar_id))
        self.last_presence = time.monotonic()

        @self.room.on("participant_connected")
        def joined(participant):
            if participant.identity == "browser":
                self.connected_user = True
                self.last_presence = time.monotonic()

        @self.room.on("participant_disconnected")
        def left(participant):
            if participant.identity == "browser":
                self.camera.clear()
                self.connected_user = False
                self.last_presence = time.monotonic()

        @self.room.on("track_subscribed")
        def subscribed(track, publication, participant):
            if (
                participant.identity == "browser"
                and track.kind == rtc.TrackKind.KIND_AUDIO
            ):
                task = asyncio.create_task(self._microphone(track))
                self.mic_tasks.add(task)
                task.add_done_callback(self._mic_done)
            elif (
                self.voice
                and self.settings.camera_vision
                and participant.identity == "browser"
                and publication.source == rtc.TrackSource.SOURCE_CAMERA
                and track.kind == rtc.TrackKind.KIND_VIDEO
            ):
                self.camera.attach(track, publication.sid)

        @self.room.on("track_unsubscribed")
        def unsubscribed(track, publication, participant):
            if (
                participant.identity == "browser"
                and publication.sid == self.camera.track_sid
            ):
                self.camera.clear()

        @self.room.on("track_muted")
        def muted(publication, participant):
            if (
                participant.identity == "browser"
                and publication.sid == self.camera.track_sid
            ):
                self.camera.clear()

        @self.room.on("track_unmuted")
        def unmuted(publication, participant):
            if (
                self.voice
                and self.settings.camera_vision
                and participant.identity == "browser"
                and publication.source == rtc.TrackSource.SOURCE_CAMERA
                and publication.track
            ):
                self.camera.attach(publication.track, publication.sid)

        await self.room.connect(
            self.settings.livekit_url,
            room_token(self.settings, self.room_name, "avatar", worker=True),
            options=rtc.RoomOptions(connect_timeout=10, single_peer_connection=False),
        )
        self.connected_user = "browser" in self.room.remote_participants
        video = rtc.LocalVideoTrack.create_video_track("avatar-video", self.video)
        audio = rtc.LocalAudioTrack.create_audio_track("avatar-audio", self.audio)
        await self.room.local_participant.publish_track(
            video,
            rtc.TrackPublishOptions(
                source=rtc.TrackSource.SOURCE_CAMERA,
                stream="avatar",
                simulcast=False,
                frame_metadata_features=[rtc.FrameMetadataFeature.FMF_FRAME_ID],
                video_encoding=rtc.VideoEncoding(
                    max_framerate=PUBLISH_FPS, max_bitrate=1_600_000
                ),
            ),
        )
        await self.room.local_participant.publish_track(
            audio,
            rtc.TrackPublishOptions(
                source=rtc.TrackSource.SOURCE_MICROPHONE, stream="avatar"
            ),
        )
        self.status = "listening"
        self.media_task = asyncio.create_task(
            self.continuous_runner.publish()
            if self.continuous_runner
            else self._publish_gpu()
            if self.gpu
            else self._publish(),
            name=f"media-{self.id}",
        )
        if self.gpu:
            self.generate_task = asyncio.create_task(
                self.continuous_runner.generate()
                if self.continuous_runner
                else self._generate_gpu(),
                name=f"generate-{self.id}",
            )
        if self.voice:
            await self.voice.start()
        self.caption_task = asyncio.create_task(self._publish_captions())

    async def _publish_captions(self):
        # Room data avoids the Windows HTTP bridge's buffered polling route.
        # HTTP snapshots remain the authoritative full-history recovery path.
        revision = -1
        while not self.closed:
            self.transcripts.changed.clear()
            snapshot = self.transcripts.snapshot()
            if snapshot["revision"] != revision:
                packet = {**snapshot, "utterances": snapshot["utterances"][-4:]}
                encoded = json.dumps(packet).encode()
                if len(encoded) <= 14000:
                    try:
                        await asyncio.wait_for(
                            self.room.local_participant.publish_data(
                                encoded,
                                reliable=True,
                                destination_identities=["browser"],
                                topic="chert-captions",
                            ),
                            1,
                        )
                        revision = snapshot["revision"]
                    except Exception:
                        pass  # A caption outage must not interrupt paired media.
            try:
                await asyncio.wait_for(self.transcripts.changed.wait(), 0.5)
            except asyncio.TimeoutError:
                pass
            await asyncio.sleep(
                0.04
            )  # Coalesce token bursts; no per-word task backlog.

    def _mic_done(self, task):
        self.mic_tasks.discard(task)
        if not task.cancelled() and task.exception():
            self.error = "Microphone stream ended unexpectedly. Reconnect to retry."

    async def _microphone(self, track):
        # This energy gate is diagnostic only; the live voice provider owns real VAD.
        stream = rtc.AudioStream(
            track,
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            capacity=10,
            frame_size_ms=20,
        )
        voiced = 0
        try:
            async for event in stream:
                samples = (
                    np.frombuffer(event.frame.data, dtype=np.int16).astype(np.float32)
                    / 32768
                )
                self.mic_rms = (
                    float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0
                )
                self.mic_packets += 1
                if self.voice:
                    self.voice.microphone(bytes(event.frame.data))
                    continue
                voiced = voiced + len(samples) if self.mic_rms > 0.04 else 0
                if voiced >= SAMPLE_RATE * 0.12 and self.response_active:
                    await self.interrupt()
                    voiced = 0
        finally:
            await stream.aclose()

    async def play(self, scenario_id: str):
        if self.voice:
            raise ValueError("Scripted playback is disabled during a live voice call.")
        samples = await asyncio.to_thread(self.catalog.audio, scenario_id)
        if self.closed:
            raise RuntimeError("Session has ended")
        await self.interrupt(count=False)
        self.buffer.replace(scenario_id, samples)
        self.request_time = time.monotonic()
        self.first_media_ms = None
        self.last_scenario = scenario_id
        self.status = "responding"

    async def interrupt(self, *, count: bool = True):
        if self.voice:
            await self.voice.interrupt()
        else:
            await self.cancel_media(count=count)

    async def begin_voice_response(self, response_id: str):
        await self.cancel_media(count=False)
        epoch = self.buffer.begin(response_id)
        self.request_time = time.monotonic()
        self.first_media_ms = None
        self.published_response_samples = 0
        self.status = "buffering"
        return epoch

    def server_played_ms(self):
        seconds = (
            self.published_response_samples / SAMPLE_RATE
            - self.audio.queued_duration
            - 0.2
        )
        return max(0, int(seconds * 1000))

    def estimated_played_ms(self):
        return self.playback.played_ms(self.buffer.epoch, self.server_played_ms())

    def advance_captions(self, frame_id, audio_playing):
        epoch = self.buffer.epoch
        sent = self.playback.sent.get(frame_id)
        # A subsequent idle frame confirms the end of the response, including
        # its final partial audio frame. It cannot complete another epoch.
        complete = bool(
            sent
            and sent[0] == epoch
            and not sent[2]
            and not self.response_active
            and audio_playing
            and self.voice
            and self.voice.synthesized_samples > 0
            and self.playback.positions.get(epoch, -2 * FRAME_SAMPLES)
            >= self.voice.synthesized_samples - 2 * FRAME_SAMPLES
        )
        self.transcripts.advance(
            epoch, int(self.estimated_played_ms() * 48), complete=complete
        )

    async def interrupted_played_ms(self, epoch: int, estimate: int):
        if self.connected_user:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.playback.cancel_ack.wait(), 0.2)
        return self.playback.played_ms(epoch, estimate)

    @property
    def response_active(self):
        epoch = self.buffer.epoch
        return (
            self.buffer.active
            or self.generating_epoch == epoch
            or self.publishing_epoch == epoch
            or bool(
                self.continuous_runner and self.continuous_runner.pending_speech[epoch]
            )
            or (not self.continuous_motion and not self.batches.empty())
            or (
                self.playing_epoch == epoch
                and time.monotonic() < self.response_audio_until
            )
        )

    async def cancel_media(self, *, count: bool = True):
        mark = time.monotonic()
        self.buffer.cancel()
        self.playback.cancelled(self.buffer.epoch)
        self.state_changed.set()
        if self.control_task:
            self.control_task.cancel()
        self.control_task = asyncio.create_task(self._notify_cancel(self.buffer.epoch))
        self.playing_epoch = None
        self.last_response_publish = None
        self.response_audio_until = 0.0
        if self.continuous_motion:
            await self.sync.mute_audio()
        else:
            await self.sync.clear_queue()
        if not self.continuous_motion:
            while not self.batches.empty():
                self.batches.get_nowait()
        self.generation_ready.set()
        if self.gpu and not self.continuous_motion:
            await self.gpu.cancel(self.buffer.epoch)
        if self.renderer:
            self.renderer.reset()
        self.status = "listening"
        self.request_time = None
        self.first_chunk_started_at = None
        self.first_chunk_decoded_at = None
        self.first_chunk_published_at = None
        self.published_response_samples = 0
        if count:
            self.interruptions += 1
        self.last_cancel_ms = (time.monotonic() - mark) * 1000

    async def _notify_cancel(self, epoch):
        # Small room-scoped notice bypasses status-poll latency. Status remains
        # authoritative and recovers a notice lost during a disconnect.
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                self.room.local_participant.publish_data(
                    json.dumps({"type": "cancel", "epoch": epoch}),
                    reliable=True,
                    destination_identities=["browser"],
                    topic="chert-control",
                ),
                1,
            )

    async def _generate_gpu(self):
        try:
            while not self.closed:
                # Clear before the synchronous readiness check, so append/finish,
                # cancellation and queue consumption cannot lose a wake-up.
                self.generation_ready.clear()
                if self.batches.full():
                    await self.generation_ready.wait()
                    continue
                chunk = take_batch(self.buffer)
                if chunk is None:
                    await self.generation_ready.wait()
                    continue
                self.generating_epoch = chunk.epoch
                if chunk.sample_start == 0:
                    self.first_chunk_started_at = time.monotonic()
                try:
                    batch = await self.gpu.generate(chunk)
                finally:
                    self.generating_epoch = None
                if batch is None or chunk.epoch != self.buffer.epoch:
                    continue
                if chunk.sample_start == 0:
                    self.first_chunk_decoded_at = time.monotonic()
                await asyncio.wait_for(self.batches.put(batch), 5)
                # Publish first. Full RGB hashing is diagnostic work, and scanning
                # 23-29 MB here must not block voice events or first-frame playback.
                hashes = await asyncio.to_thread(self._hash_frames, batch.frames)
                self.frame_hashes.update(hashes)
                self.generated_frames += len(batch.frames)
                self.generation_seconds += batch.compute_seconds
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = f"Avatar generation failed ({type(exc).__name__}). Check the GPU session log."
            self.status = "error"
            self.buffer.cancel()

    @staticmethod
    def _hash_frames(frames):
        return {hashlib.sha256(frame).digest() for frame in frames}

    async def _publish_gpu(self):
        from PIL import Image, ImageOps

        with Image.open(self.catalog.image(self.avatar_id)) as image:
            last_frame = np.asarray(ImageOps.fit(image.convert("RGB"), (WIDTH, HEIGHT)))
        idle = IdlePlayer(
            self.gpu.idle_clip if self.settings.avatar_idle_motion else None
        )
        stream_sample = 0
        silence = bytes(960 * 2)
        try:
            while not self.closed and not self.error:
                try:
                    batch = self.batches.get_nowait()
                    self.generation_ready.set()
                except asyncio.QueueEmpty:
                    idle_started = time.monotonic()
                    waiting = self.response_active
                    if (
                        waiting
                        and self.playing_epoch == self.buffer.epoch
                        and self.last_response_publish is not None
                        and time.monotonic() - self.last_response_publish > 2
                    ):
                        self.stalls += 1
                        await self.interrupt()
                        self.error = "Avatar rendering stalled for over two seconds. End and restart the call."
                        self.status = "error"
                        return
                    # Keep both RTP tracks continuous during listening/buffering.
                    # Sending occasional video without audio between replies can
                    # destabilize receiver playout after an idle gap. These are
                    # cached idle motion or held images, never response coverage.
                    # Hold a real mid-reply gap rather than switching to listening
                    # animation while speech generation is falling behind.
                    animate = idle.clip is not None and not (
                        waiting and self.playing_epoch == self.buffer.epoch
                    )
                    if animate:
                        last_frame = idle.next(last_frame)
                    else:
                        idle.stop()
                    frame = rtc.VideoFrame(
                        WIDTH, HEIGHT, rtc.VideoBufferType.RGB24, last_frame.tobytes()
                    )
                    tag(frame, self.buffer.epoch, 0, False)
                    await asyncio.wait_for(
                        self.sync.push(frame, stream_sample / SAMPLE_RATE), 2
                    )
                    for offset in (0, 960):
                        await asyncio.wait_for(
                            self.sync.push(
                                rtc.AudioFrame(silence, SAMPLE_RATE, 1, 960),
                                (stream_sample + offset + 960) / SAMPLE_RATE,
                            ),
                            2,
                        )
                    if animate:
                        self.idle_motion_frames += 1
                    else:
                        self.held_frames += 1
                    stream_sample += FRAME_SAMPLES
                    self.status = "buffering" if waiting else "listening"
                    await asyncio.sleep(
                        max(0, idle_started + 1 / FPS - time.monotonic())
                    )
                    continue
                self.publishing_epoch = batch.audio.epoch
                idle.stop()
                start = time.monotonic()
                for index, rgb in enumerate(batch.frames):
                    if batch.audio.epoch != self.buffer.epoch or self.closed:
                        break
                    offset = index * FRAME_SAMPLES
                    pcm = batch.audio.pcm[offset : offset + FRAME_SAMPLES]
                    frame = rtc.VideoFrame(
                        WIDTH, HEIGHT, rtc.VideoBufferType.RGB24, rgb.tobytes()
                    )
                    tag(
                        frame,
                        batch.audio.epoch,
                        batch.audio.sample_start + offset,
                        True,
                    )
                    await asyncio.wait_for(
                        self.sync.push(frame, stream_sample / SAMPLE_RATE), 2
                    )
                    for part_start in (0, 960):
                        if batch.audio.epoch != self.buffer.epoch:
                            break
                        part = pcm[part_start : part_start + 960]
                        await asyncio.wait_for(
                            self.sync.push(
                                rtc.AudioFrame(
                                    part.tobytes(), SAMPLE_RATE, 1, len(part)
                                ),
                                (stream_sample + part_start + len(part)) / SAMPLE_RATE,
                            ),
                            2,
                        )
                    if batch.audio.epoch != self.buffer.epoch:
                        break
                    last_frame = rgb
                    self.frames += 1
                    self.speech_frames += 1
                    self.published_response_samples += min(
                        FRAME_SAMPLES, max(0, batch.audio.valid_samples - offset)
                    )
                    self.playing_epoch = batch.audio.epoch
                    self.last_response_publish = time.monotonic()
                    # Idle silence must not keep a completed response active.
                    self.response_audio_until = (
                        self.last_response_publish + self.audio.queued_duration
                    )
                    self.status = "responding"
                    if self.first_chunk_published_at is None:
                        self.first_chunk_published_at = time.monotonic()
                    if self.first_media_ms is None and self.request_time is not None:
                        self.first_media_ms = (
                            time.monotonic() - self.request_time
                        ) * 1000
                    stream_sample += FRAME_SAMPLES
                    await asyncio.sleep(
                        max(0, start + (index + 1) / FPS - time.monotonic())
                    )
                self.publishing_epoch = None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = f"Avatar publication failed ({type(exc).__name__}). End and restart the call."
            self.status = "error"
        finally:
            self.publishing_epoch = None

    async def _publish(self):
        try:
            while not self.closed:
                audio = self.buffer.next()
                packet = await asyncio.to_thread(self.renderer.render, audio)
                if audio.epoch != self.buffer.epoch:
                    continue
                frame = rtc.VideoFrame(
                    WIDTH, HEIGHT, rtc.VideoBufferType.RGB24, packet.rgb.tobytes()
                )
                tag(
                    frame,
                    audio.epoch,
                    audio.sample_start if audio.valid_samples else 0,
                    bool(audio.valid_samples),
                )
                await asyncio.wait_for(self.sync.push_pair(frame, audio.pcm), timeout=2)
                if audio.epoch != self.buffer.epoch:
                    continue
                self.frames += 1
                if audio.valid_samples:
                    self.speech_frames += 1
                    self.published_response_samples += audio.valid_samples
                    self.playing_epoch = audio.epoch
                    self.response_audio_until = (
                        time.monotonic() + self.audio.queued_duration
                    )
                    self.status = "responding"
                    if self.first_media_ms is None and self.request_time is not None:
                        self.first_media_ms = (
                            time.monotonic() - self.request_time
                        ) * 1000
                elif not self.buffer.active:
                    self.status = "listening"
                elif self.buffer.receiving:
                    self.status = "buffering"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = (
                f"Media worker failed ({type(exc).__name__}); end and restart the demo."
            )
            self.status = "error"

    def snapshot(self):
        elapsed = max(0.001, time.monotonic() - self.started)
        fps = self.sync.actual_fps
        return {
            "state": self.status,
            "elapsed_seconds": round(elapsed, 2),
            "renderer": "avatarforever_bounded_v1"
            if self.model_id == "avatarforever"
            else "diagnostic_mock",
            "qualification": "streaming_experimental"
            if self.model_id == "avatarforever"
            else "diagnostic_only",
            "model_generation_fps": round(
                self.generated_frames / self.generation_seconds, 2
            )
            if self.generation_seconds
            else None,
            "generated_frames": self.generated_frames,
            "held_frames": self.held_frames,
            "video_delayed": bool(
                self.continuous_runner and self.continuous_runner.consecutive_holds >= 6
            ),
            "camera_shared": self.camera.frame is not None,
            "hold_reasons": dict(self.continuous_runner.hold_reasons)
            if self.continuous_runner
            else {},
            "max_consecutive_hold_ms": self.continuous_runner.max_consecutive_holds
            * 1000
            / FPS
            if self.continuous_runner
            else None,
            "speech_underflow_frames": self.continuous_runner.speech_underflows
            if self.continuous_runner
            else None,
            "speech_preroll_frames": self.continuous_runner.speech_preroll_frames
            if self.continuous_runner
            else None,
            "startup_buffer_ms": round(self.continuous_runner.startup_buffer_ms, 2)
            if self.continuous_runner
            else None,
            "gpu_roundtrip_fps": round(
                self.generated_frames / self.continuous_runner.roundtrip_seconds, 2
            )
            if self.continuous_runner and self.continuous_runner.roundtrip_seconds
            else None,
            "max_video_capture_lateness_ms": round(self.sync.max_lateness_ms, 2)
            if self.continuous_motion
            else None,
            "idle_motion_frames": self.idle_motion_frames,
            "idle_motion_ready": bool(
                self.continuous_motion
                or (
                    self.settings.avatar_idle_motion
                    and self.gpu
                    and self.gpu.idle_clip is not None
                )
            ),
            "avatar_timing": self.avatar_timing(),
            "distinct_generated_frames": len(self.frame_hashes),
            "exact_duplicate_frames": self.generated_frames - len(self.frame_hashes),
            "distinct_generation_fps": round(
                len(self.frame_hashes) / self.generation_seconds, 2
            )
            if self.generation_seconds
            else None,
            "render_stalls": self.stalls,
            "response_active": self.response_active,
            "decoded_batch_queue": self.batches.qsize(),
            "pending_audio_seconds": round(len(self.buffer.samples) / SAMPLE_RATE, 3),
            "gpu_info": self.gpu.info if self.gpu else None,
            "gpu_memory": self.gpu.last_memory if self.gpu else None,
            "cold_load_seconds": self.gpu.load_seconds if self.gpu else None,
            "transport_fps": round(fps, 1) if math.isfinite(fps) else 0,
            "video_max_fps": PUBLISH_FPS,
            "geometry": {"width": WIDTH, "height": HEIGHT, "fps": FPS},
            "native_model_fps": FPS,
            "video_rate_limited_frames": self.video.rate_limited_frames,
            "frames_published": self.frames,
            "audio_queue_ms": round(self.audio.queued_duration * 1000, 1),
            "first_media_ms": round(self.first_media_ms, 1)
            if self.first_media_ms is not None
            else None,
            "server_cancel_ms": round(self.last_cancel_ms, 2)
            if self.last_cancel_ms is not None
            else None,
            "interruptions": self.interruptions,
            "mic_rms": round(self.mic_rms, 4),
            "mic_packets": self.mic_packets,
            "response_id": self.buffer.response_id,
            "epoch": self.buffer.epoch,
            "connected_user": self.connected_user,
            "error": self.error,
            "last_scenario": self.last_scenario,
            "voice_mode": (
                f"openai_text_{self.settings.tts_provider}_tts"
                if self.settings.tts_provider in {"qwen", "elevenlabs"}
                else "openai_realtime"
            )
            if self.voice
            else "fixed_replay",
            "continuous_motion": self.continuous_motion,
            "continuous_transitions": self.continuous_runner.transitions
            if self.continuous_runner
            else 0,
            "effective_voice": self.voice.effective_voice if self.voice else None,
            "voice_model": self.voice.resolved_model if self.voice else None,
            "voice_diagnostics": self.voice.diagnostics() if self.voice else None,
            "voice_timing": self.voice.timing(self.first_media_ms)
            if self.voice
            else None,
            "playback_offset_source": "browser_video_with_audio_playing"
            if self.playback.browser_seen
            else "unconfirmed_zero",
            "browser_playback": {
                "reports": self.playback.reports,
                "last_frame_id": self.playback.last_frame_id,
                "played_offset_ms": self.estimated_played_ms(),
                "cancel_epoch": self.playback.cancel_epoch,
                "cancel_ack_roundtrip_ms": self.playback.cancel_roundtrip_ms,
                "mute_operation_ms": self.playback.browser_mute_ms,
                "offset_source": "browser_video_with_audio_playing"
                if self.playback.browser_seen
                else "unconfirmed_zero",
                "physical_audio_stop_ms": None,
            },
            "timing_note": "Generation FPS excludes cold load and transport. Browser receipts confirm displayed frames and mute operations, not physical speaker timing.",
        }

    def avatar_timing(self):
        def interval(start, end):
            if start is None or end is None or end < start:
                return None
            return round((end - start) * 1000, 2)

        return {
            "first_audio_to_gpu_request_ms": interval(
                self.voice.first_audio_at if self.voice else self.request_time,
                self.first_chunk_started_at,
            ),
            "first_chunk_roundtrip_ms": interval(
                self.first_chunk_started_at, self.first_chunk_decoded_at
            ),
            "decoded_to_publish_ms": interval(
                self.first_chunk_decoded_at, self.first_chunk_published_at
            ),
            "scope": "server monotonic; excludes VAD and browser playback",
        }

    async def close(self):
        if self._closing is None:
            self._closing = asyncio.create_task(
                self._close(), name=f"close-worker-{self.id}"
            )
        # Request cancellation must not abandon room/GPU cleanup. Concurrent
        # callers wait for the same teardown rather than mistaking closed=True
        # (which stops production) for completed resource release.
        await asyncio.shield(self._closing)

    async def _close(self):
        self.closed = True
        self.transcripts.close()
        self.status = "closing"
        self.buffer.cancel()
        try:
            with contextlib.suppress(Exception):
                await self.camera.close()
            with contextlib.suppress(Exception):
                await self.sync.clear_queue()
            if self.gpu and not self.gpu.closed:
                with contextlib.suppress(Exception):
                    await self.gpu.cancel(self.buffer.epoch)
            if self.voice:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(self.voice.close(), 2)
            # Give cooperative cancellation a short window to preserve resident
            # weights. A stuck kernel is retired by the process boundary instead.
            if self.gpu_pool and self.generate_task:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(asyncio.shield(self.generate_task), 1)
        finally:
            tasks = (
                list(self.mic_tasks)
                + ([self.media_task] if self.media_task else [])
                + ([self.generate_task] if self.generate_task else [])
                + ([self.control_task] if self.control_task else [])
                + ([self.caption_task] if self.caption_task else [])
            )
            for task in tasks:
                task.cancel()
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                try:
                    if self.gpu_pool:
                        await self.gpu_pool.release(self.id)
                    elif self.gpu:
                        await self.gpu.close()
                finally:
                    # Attempt each release even when another media resource fails.
                    for release in (
                        self.sync.clear_queue,
                        self.sync.aclose,
                        self.room.disconnect,
                        self.audio.aclose,
                        self.video.aclose,
                    ):
                        with contextlib.suppress(Exception):
                            await asyncio.wait_for(release(), 5)
                    self.renderer = None
                    self.status = "ended"


async def delete_room(settings: Settings, room_name: str):
    url = settings.livekit_url.replace("wss://", "https://").replace("ws://", "http://")
    async with api.LiveKitAPI(
        url=url,
        api_key=settings.livekit_api_key,
        api_secret=settings.livekit_api_secret.get_secret_value(),
    ) as client:
        with contextlib.suppress(api.TwirpError):
            await client.room.delete_room(api.DeleteRoomRequest(room=room_name))
