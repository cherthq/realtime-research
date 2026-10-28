"""Stateful mono PCM resampling shared by live voice and frozen replay assets."""

import numpy as np
from livekit import rtc


class PCMResampler:
    def __init__(self, input_rate: int, output_rate: int):
        self.resampler = rtc.AudioResampler(
            input_rate,
            output_rate,
            num_channels=1,
            quality=rtc.AudioResamplerQuality.HIGH,
        )
        self.closed = False

    @staticmethod
    def _samples(frames):
        parts = [np.frombuffer(frame.data, dtype=np.int16).copy() for frame in frames]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int16)

    def push(self, pcm: bytes) -> np.ndarray:
        if self.closed or len(pcm) % 2:
            raise ValueError("Resampler is finished or PCM16 contains a partial sample")
        return self._samples(self.resampler.push(bytearray(pcm)))

    def finish(self) -> np.ndarray:
        if self.closed:
            return np.zeros(0, dtype=np.int16)
        self.closed = True
        return self._samples(self.resampler.flush())
