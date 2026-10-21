"""Synthetic local media only. Skips when FFmpeg/FFprobe are unavailable."""

from array import array
from pathlib import Path
import shutil
import tempfile
import unittest
import wave

from data_fetching.core import Config
from data_fetching.media import frame_png, probe, render_clip, run, validate_clip


AVAILABLE = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


@unittest.skipUnless(AVAILABLE, "FFmpeg and FFprobe required")
class MediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.source = cls.root / "offset.mkv"
        # Video begins at 2s, tone at 2.5s. Independently applying STARTPTS
        # would incorrectly erase this intentional 0.5s audio delay.
        run(["ffmpeg", "-v", "error", "-nostdin", "-y",
             "-f", "lavfi", "-i", "testsrc2=size=1024x768:rate=25:duration=6",
             "-f", "lavfi", "-i", "sine=frequency=800:sample_rate=48000:duration=5.5",
             "-filter_complex", "[0:v]setpts=PTS+2/TB[v];[1:a]asetpts=PTS+2.5/TB[a]",
             "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "ultrafast",
             "-c:a", "pcm_s16le", "-fps_mode", "passthrough", "-copyts", cls.source])
        cls.info = probe(cls.source)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_probe_and_frame_with_nonzero_origin(self):
        self.assertAlmostEqual(self.info["video_start"], 2, places=2)
        self.assertAlmostEqual(self.info["audio_start"], 2.5, places=2)
        self.assertAlmostEqual(self.info["duration"], 6, places=2)
        self.assertTrue(frame_png(self.source, 1, self.info).startswith(b"\x89PNG"))

    def test_render_preserves_delay(self):
        result = render_clip(self.source, self.root / "delay", 0, 5, [128, 96, 768, 576], Config(), self.info)
        output = validate_clip(result["video"], result["audio"], Config(), 5)
        self.assertEqual((output["width"], output["height"]), (768, 576))
        with wave.open(str(result["audio"]), "rb") as handle:
            samples = array("h", handle.readframes(handle.getnframes()))
        early = max(abs(x) for x in samples[:int(0.35 * 48000)])
        tone = max(abs(x) for x in samples[int(0.7 * 48000):int(0.9 * 48000)])
        self.assertLess(early, 20, "Audio must not be shifted to remove the source delay")
        self.assertGreater(tone, 1000)

    def test_trim_into_running_audio(self):
        result = render_clip(self.source, self.root / "later", 0.75, 5.75,
                             [0, 0, 1024, 768], Config(), self.info)
        with wave.open(str(result["audio"]), "rb") as handle:
            samples = array("h", handle.readframes(4800))
        self.assertGreater(max(abs(x) for x in samples), 1000)
        self.assertTrue(result["reference_image"].is_file())

    def test_missing_audio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing_audio"):
            render_clip(self.source, self.root / "silent", 0, 5,
                        [0, 0, 768, 576], Config(), {**self.info, "audio_index": None})

    def test_render_cannot_stretch_upscale_or_escape(self):
        for crop in ([0, 0, 640, 480], [0, 0, 800, 576], [900, 0, 768, 576], [1, 0, 768, 576]):
            with self.subTest(crop=crop), self.assertRaisesRegex(ValueError, "Crop must"):
                render_clip(self.source, self.root / "invalid", 0, 5, crop, Config(), self.info)


if __name__ == "__main__":
    unittest.main()
