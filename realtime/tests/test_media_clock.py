import asyncio
import time
from types import SimpleNamespace

import numpy as np

from realtime.media_clock import AudioClock


class QueuedAudio:
    sample_rate = 48000

    def __init__(self):
        self.end = 0
        self.packets = []

    @property
    def queued_duration(self):
        return max(0, self.end - time.monotonic())

    async def capture_frame(self, frame):
        self.end = max(time.monotonic(), self.end) + frame.samples_per_channel / 48000
        self.packets.append(bytes(frame.data))
        await asyncio.sleep(max(0, self.queued_duration - 0.08))

    def clear_queue(self):
        self.end = time.monotonic()
        self.packets.clear()


def test_pairs_follow_audio_across_chunk_boundaries_and_producer_jitter():
    async def run():
        audio, captured = QueuedAudio(), []
        video = SimpleNamespace(
            capture_frame=lambda frame: captured.append((frame, time.monotonic()))
        )
        clock = AudioClock(audio, video)
        expected = []
        try:
            for index in range(30):
                if index % 8 == 0:
                    await asyncio.sleep(0.025)
                await clock.push_pair(index, np.zeros(1920, np.int16))
                expected.append(audio.end - 0.04)
            await asyncio.wait_for(clock.queue.join(), 2)
            assert [frame for frame, _ in captured] == list(range(30))
            assert (
                max(abs(actual - due) for (_, actual), due in zip(captured, expected))
                < 0.06
            )
            assert clock.queue.qsize() == 0
        finally:
            await clock.aclose()

    asyncio.run(run())


def test_interrupt_silences_queued_and_late_audio_without_dropping_video():
    async def run():
        audio, captured, epoch = QueuedAudio(), [], [1]
        clock = AudioClock(
            audio, SimpleNamespace(capture_frame=captured.append), lambda: epoch[0]
        )
        try:
            first = SimpleNamespace(playback_position=(1, 0, True))
            late = SimpleNamespace(playback_position=(1, 1920, True))
            await clock.push_pair(first, np.ones(1920, np.int16))
            epoch[0] = 2
            await clock.mute_audio()
            await clock.push_pair(late, np.ones(1920, np.int16))
            await asyncio.wait_for(clock.queue.join(), 1)
            assert captured == [first, late]
            assert all(not any(packet) for packet in audio.packets)
        finally:
            await clock.aclose()

    asyncio.run(run())
