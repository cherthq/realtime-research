"""Visible diagnostic renderer, deliberately not presented as avatar inference."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .contracts import HEIGHT, WIDTH, AudioChunk, MediaChunk


class MockRenderer:
    def __init__(self, image: Path | None):
        if image and image.is_file():
            with Image.open(image) as source:
                # Resizing is only for the diagnostic canvas, never real model output.
                self.background = ImageOps.fit(source.convert("RGB"), (WIDTH, HEIGHT))
        else:
            self.background = Image.new("RGB", (WIDTH, HEIGHT), "#1e293b")
        self.counter = 0
        self.font = ImageFont.load_default(size=12)

    def reset(self):
        self.counter = 0

    def render(self, chunk: AudioChunk) -> MediaChunk:
        self.counter += 1
        frame = self.background.copy()
        draw = ImageDraw.Draw(frame)
        draw.rounded_rectangle((12, 12, 118, 38), radius=5, fill="#26352d")
        draw.text((22, 19), "Test preview", font=self.font, fill="#f2f4ec")
        rms = float(np.sqrt(np.mean((chunk.pcm.astype(np.float32) / 32768) ** 2)))
        # Small flash marker retains the diagnostic audio/visual timing fixture.
        draw.ellipse((605, 19, 617, 31), fill="#b2d985" if rms > 0.02 else "#465550")
        return MediaChunk(chunk, np.asarray(frame, dtype=np.uint8))
