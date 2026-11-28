"""Room observer: CPU-only recording of both call participants, never GPU work."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from datetime import timedelta
import os
import queue
import shutil
import subprocess
import threading
import time

from PIL import Image, ImageDraw, ImageOps
from livekit import api, rtc

from .call_logs import RESERVE

RATE = 48000
FPS = 15


class MediaWriter:
    """One bounded queue and one encoding thread; media callbacks never do disk IO."""

    def __init__(self, folder, clock=time.monotonic):
        self.folder, self.clock = folder, clock
        self.started = None
        self.ended = None
        self.audio = queue.Queue(maxsize=1500)
        self.frames = {}
        self.stop = threading.Event()
        self.error = None
        self.process = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def begin(self):
        if self.started is None:
            self.started = self.clock()
        self.ended = None

    def pcm(self, speaker, raw):
        if self.started is None or self.stop.is_set():
            return
        try:
            self.audio.put_nowait(
                (
                    speaker,
                    max(0, self.clock() - self.started - len(raw) / (RATE * 2)),
                    raw,
                )
            )
        except queue.Full:
            self.error = "Recording could not keep up with incoming audio"
            self.stop.set()

    def video(self, speaker, frame):
        self.frames[speaker] = (frame, self.clock())

    def run(self):
        handles = {}
        try:
            if shutil.disk_usage(self.folder).free < RESERVE:
                raise RuntimeError("Recording storage has less than 5 GB free")
            handles = {
                role: open(self.folder / (role + ".pcm"), "wb")
                for role in ("caller", "agent")
            }
            for handle in handles.values():
                os.chmod(handle.name, 0o600)
            command = [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "960x360",
                "-r",
                str(FPS),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "25",
                "-threads",
                "2",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+frag_keyframe+empty_moov",
                str(self.folder / "silent.mp4"),
            ]
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            positions = dict(caller=0, agent=0)
            frame_index = 0
            disk_check = self.clock()
            while not self.stop.is_set() or not self.audio.empty():
                for _ in range(100):
                    try:
                        role, seconds, raw = self.audio.get_nowait()
                    except queue.Empty:
                        break
                    target = round(seconds * RATE) * 2
                    # Ignore small network jitter, but preserve actual track gaps/mutes.
                    position = positions[role]
                    if target > position + RATE // 5:
                        position = target
                    handles[role].seek(position)
                    handles[role].write(raw)
                    positions[role] = position + len(raw)
                now = self.clock()
                if now - disk_check >= 5:
                    disk_check = now
                    if shutil.disk_usage(self.folder).free < RESERVE:
                        raise RuntimeError(
                            "Recording stopped: less than 5 GB storage remains"
                        )
                if self.started is not None:
                    target_frames = int(((self.ended or now) - self.started) * FPS)
                    if frame_index <= target_frames:
                        canvas = Image.new("RGB", (960, 360), "#171923")
                        draw = ImageDraw.Draw(canvas)
                        for index, role in enumerate(("caller", "agent")):
                            value = self.frames.get(role)
                            if value and now - value[1] < 2:
                                frame = value[0]
                                rgb = (
                                    frame
                                    if frame.type == rtc.VideoBufferType.RGB24
                                    else frame.convert(rtc.VideoBufferType.RGB24)
                                )
                                picture = Image.frombytes(
                                    "RGB", (rgb.width, rgb.height), bytes(rgb.data)
                                )
                                picture = ImageOps.contain(picture, (480, 360))
                                canvas.paste(
                                    picture,
                                    (
                                        index * 480 + (480 - picture.width) // 2,
                                        (360 - picture.height) // 2,
                                    ),
                                )
                            draw.rectangle(
                                (index * 480, 0, index * 480 + 55, 18), fill="#171923"
                            )
                            draw.text((index * 480 + 6, 3), role.title(), fill="white")
                        # Preserve elapsed time even if encoder scheduling briefly lags.
                        raw = canvas.tobytes()
                        while frame_index <= target_frames:
                            self.process.stdin.write(raw)
                            frame_index += 1
                self.stop.wait(1 / FPS / 2) if not self.stop.is_set() else None
            self.process.stdin.close()
            if self.process.wait(timeout=20) and self.started is not None:
                raise RuntimeError("Video encoder failed")
        except Exception as exc:
            self.failure = exc
            self.error = (
                str(exc)
                if isinstance(exc, RuntimeError)
                else "Recording media writer failed"
            )
            self.stop.set()
        finally:
            for handle in handles.values():
                handle.close()
            if self.process and self.process.poll() is None:
                self.process.kill()
                self.process.wait()

    def finish(self):
        self.ended = self.ended or self.clock()
        self.stop.set()
        self.thread.join(timeout=25)
        if self.thread.is_alive():
            if self.process:
                self.process.kill()
            self.thread.join(timeout=5)
            raise RuntimeError("Recording encoder did not finish")
        if self.error:
            raise RuntimeError(self.error)
        duration = max(0, self.ended - (self.started or self.ended))
        if duration < 0.05:
            return duration, False

        def ffmpeg(args):
            with (self.folder / "encoder.log").open("ab") as log:
                os.chmod(log.name, 0o600)
                subprocess.run(
                    ["ffmpeg", "-nostdin", "-v", "error", "-y", *args],
                    check=True,
                    timeout=max(60, duration),
                    stdout=subprocess.DEVNULL,
                    stderr=log,
                )

        for role in ("caller", "agent"):
            path = self.folder / (role + ".pcm")
            with path.open("ab") as out:
                out.truncate(max(path.stat().st_size, round(duration * RATE) * 2))
            ffmpeg(
                [
                    "-f",
                    "s16le",
                    "-ar",
                    str(RATE),
                    "-ac",
                    "1",
                    "-i",
                    str(path),
                    "-t",
                    str(duration),
                    str(self.folder / (role + ".wav")),
                ]
            )
        ffmpeg(
            [
                "-i",
                str(self.folder / "caller.wav"),
                "-i",
                str(self.folder / "agent.wav"),
                "-filter_complex",
                "[0:a][1:a]amix=inputs=2:normalize=0,alimiter=limit=0.95:attack=1[a]",
                "-map",
                "[a]",
                str(self.folder / "audio.wav"),
            ]
        )
        ffmpeg(
            [
                "-i",
                str(self.folder / "silent.mp4"),
                "-i",
                str(self.folder / "audio.wav"),
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-t",
                str(duration),
                "-movflags",
                "+faststart",
                str(self.folder / "video.mp4"),
            ]
        )
        for name in ("caller.pcm", "agent.pcm", "silent.mp4"):
            (self.folder / name).unlink()
        for path in self.folder.iterdir():
            path.chmod(0o600)
        return duration, True


class CallRecorder:
    def __init__(self, worker, logs, source):
        self.worker, self.logs, self.source = worker, logs, source
        self.id = worker.id
        self.folder = logs.begin(
            self.id, source, getattr(worker.voice, "configuration_version", None)
        )
        self.writer = None
        self.room = rtc.Room()
        self.tasks = set()
        self.track_tasks = {}
        self.rows = OrderedDict()
        self.closed = False
        self.error = None
        self.checkpoint = None
        self.start_time = time.monotonic()
        original = worker.transcripts._touch

        def changed():
            original()
            for row in worker.transcripts.snapshot()["utterances"]:
                prior = self.rows.get(row["id"])
                self.rows[row["id"]] = {
                    **row,
                    "seconds": prior["seconds"]
                    if prior
                    else round(
                        max(
                            0,
                            time.monotonic()
                            - (
                                self.writer.started
                                if self.writer and self.writer.started
                                else self.start_time
                            ),
                        ),
                        3,
                    ),
                }

        worker.transcripts._touch = changed

    async def start(self):
        self.writer = MediaWriter(self.folder)

        @self.room.on("participant_connected")
        def joined(participant):
            if participant.identity == "browser":
                self.writer.begin()

        @self.room.on("participant_disconnected")
        def left(participant):
            if participant.identity == "browser":
                self.writer.ended = time.monotonic()

        @self.room.on("track_subscribed")
        def subscribed(track, publication, participant):
            if (
                participant.identity not in ("browser", "avatar")
                or publication.sid in self.track_tasks
            ):
                return
            role = "caller" if participant.identity == "browser" else "agent"
            if role == "caller" and self.writer.started is None:
                self.writer.begin()
            task = asyncio.create_task(self.read_track(track, role))
            self.track_tasks[publication.sid] = task
            self.tasks.add(task)
            task.add_done_callback(self.done)

        @self.room.on("track_muted")
        def muted(publication, participant):
            if publication.kind == rtc.TrackKind.KIND_VIDEO:
                self.writer.frames.pop(
                    "caller" if participant.identity == "browser" else "agent", None
                )

        @self.room.on("track_unsubscribed")
        def unsubscribed(track, publication, participant):
            task = self.track_tasks.pop(publication.sid, None)
            if task:
                task.cancel()
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                self.writer.frames.pop(
                    "caller" if participant.identity == "browser" else "agent", None
                )

        @self.room.on("disconnected")
        def disconnected(*_):
            if not self.closed:
                self.error = "Recording connection interrupted"

        settings = self.worker.settings
        token = (
            api.AccessToken(
                settings.livekit_api_key, settings.livekit_api_secret.get_secret_value()
            )
            .with_identity("call-recorder")
            .with_ttl(timedelta(seconds=settings.session_limit_seconds + 120))
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=self.worker.room_name,
                    can_publish=False,
                    can_subscribe=True,
                    can_publish_data=False,
                )
            )
            .to_jwt()
        )
        await self.room.connect(
            settings.livekit_url, token, options=rtc.RoomOptions(connect_timeout=10)
        )
        if "browser" in self.room.remote_participants:
            self.writer.begin()
        self.checkpoint = asyncio.create_task(self.persist())

    def done(self, task):
        self.tasks.discard(task)
        if not task.cancelled() and task.exception():
            self.error = "A recording media stream failed"

    async def read_track(self, track, role):
        audio = track.kind == rtc.TrackKind.KIND_AUDIO
        stream = (
            rtc.AudioStream(
                track, capacity=250, sample_rate=RATE, num_channels=1, frame_size_ms=20
            )
            if audio
            else rtc.VideoStream(track, capacity=1)
        )
        try:
            async for event in stream:
                if audio:
                    self.writer.pcm(role, bytes(event.frame.data))
                else:
                    self.writer.video(role, event.frame)
        finally:
            await stream.aclose()

    async def persist(self):
        while True:
            await asyncio.to_thread(
                self.logs.transcript, self.id, list(self.rows.values())
            )
            duration = (
                max(0, (self.writer.ended or time.monotonic()) - self.writer.started)
                if self.writer.started
                else 0
            )
            await asyncio.to_thread(
                self.logs.update,
                self.id,
                "recording",
                duration,
                self.error or self.writer.error,
                self.source,
            )
            await asyncio.sleep(5)

    async def close(self):
        if self.closed:
            return
        self.closed = True
        if self.checkpoint:
            self.checkpoint.cancel()
            await asyncio.gather(self.checkpoint, return_exceptions=True)
        for task in list(self.tasks):
            task.cancel()
        await asyncio.gather(*list(self.tasks), return_exceptions=True)
        try:
            await self.room.disconnect()
        except Exception:
            self.error = self.error or "Recording connection cleanup failed"
        duration = (
            max(0, (self.writer.ended or time.monotonic()) - self.writer.started)
            if self.writer and self.writer.started
            else 0
        )
        await asyncio.to_thread(self.logs.transcript, self.id, list(self.rows.values()))
        await asyncio.to_thread(
            self.logs.update, self.id, "processing", duration, self.error, self.source
        )
        try:
            if not self.writer:
                raise RuntimeError("Recorder did not start")
            duration, media = await asyncio.to_thread(self.writer.finish)
            status = "partial" if self.error else "complete" if media else "no_media"
        except Exception as exc:
            status = "failed"
            self.error = (
                str(exc)
                if isinstance(exc, RuntimeError)
                else "Could not finalize recording"
            )
        await asyncio.to_thread(
            self.logs.update, self.id, status, duration, self.error, self.source
        )
