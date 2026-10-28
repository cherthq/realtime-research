"""Test-only subprocess exercising the real IPC server without CUDA or weights."""

import math
import sys
import time

import numpy as np

from realtime.contracts import FRAME_SAMPLES, HEIGHT, WIDTH
from realtime.gpu.server import serve
from realtime.streaming import GeneratedBatch


class Backend:
    def open_session(self, image):
        self.reset()

    def close_session(self):
        self.reset()

    def reset(self):
        self.position = 0

    def info(self):
        return {"qualification": "test_only"}

    def metrics(self):
        return {"host_peak_rss_bytes": 123456, "scope": "synthetic test fixture"}

    def generate(self, audio, cancelled):
        if audio.response_id == "crash":
            raise RuntimeError("simulated GPU failure")
        if audio.response_id in {"slow", "stuck"}:
            for _ in range(200):
                time.sleep(0.01)
                if cancelled.is_set() and audio.response_id == "slow":
                    raise RuntimeError("cancelled")
        if self.position != audio.sample_start:
            raise ValueError("History was lost")
        self.position += audio.valid_samples
        count = math.ceil(audio.valid_samples / FRAME_SAMPLES)
        frames = np.full((count, HEIGHT, WIDTH, 3), audio.epoch % 256, dtype=np.uint8)
        return GeneratedBatch(audio, frames, 0.01)

    def close(self):
        pass


if __name__ == "__main__":
    serve(Backend(), sys.stdin.buffer, sys.stdout.buffer)
