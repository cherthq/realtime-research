"""Synthetic local media only. Skips when FFmpeg/FFprobe are unavailable."""

from array import array
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
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

    def test_input_seek_matches_legacy_trim_with_nonzero_origin(self):
        self.assert_seek_matches_reference(self.source, self.info, "seek-match")

    def test_input_seek_matches_legacy_trim_with_vfr(self):
        video = self.root / "vfr.mkv"
        run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-copyts", "-i", self.source,
             "-vf", "setpts='PTS+if(gte(N,75),0.08/TB,0)'", "-fps_mode", "vfr",
             "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "copy", video])
        self.assert_seek_matches_reference(video, probe(video), "vfr-seek-match")

    def test_late_seek_keeps_aac_sample_phase_in_coarse_container(self):
        video = self.root / "aac-clock.mkv"
        run(["ffmpeg", "-v", "error", "-nostdin", "-y",
             "-f", "lavfi", "-i", "testsrc2=size=1024x768:rate=25:duration=12",
             "-f", "lavfi", "-i", "sine=frequency=811:sample_rate=44100:duration=12",
             "-c:v", "libx264", "-preset", "ultrafast", "-g", "25", "-c:a", "aac", video])
        self.assert_seek_matches_reference(video, probe(video), "aac-seek-match", 8, 11)

    def assert_seek_matches_reference(self, source, info, name, start=0.75, end=5.75):
        def legacy(command, **kwargs):
            command = list(command)
            if "-copyts" in command:
                for flag in ("-seek_timestamp", "-ss"):
                    index = command.index(flag)
                    del command[index:index + 2]
                command.remove("-noaccurate_seek")
                # Restore the old single-input graph, not merely a zero seek.
                first = command.index("-i")
                del command[first + 2:first + 4]
                at = command.index("-filter_complex") + 1
                command[at] = command[at].replace("[1:", "[0:")
            return run(command, **kwargs)
        with patch("data_fetching.media.run", wraps=run) as calls:
            fast = render_clip(source, self.root / name / "fast", start, end,
                               [0, 0, 1024, 768], Config(), info)
        command = next(c.args[0] for c in calls.call_args_list if "-copyts" in c.args[0])
        self.assertLess(command.index("-ss"), command.index("-i"))
        self.assertEqual(float(command[command.index("-ss") + 1]),
                         max(info["video_start"], info["video_start"] + start - 2))
        self.assertIn("-noaccurate_seek", command)
        with patch("data_fetching.media.run", side_effect=legacy):
            reference = render_clip(source, self.root / name / "reference", start, end,
                                    [0, 0, 1024, 768], Config(), info)
        frame_hashes = lambda path: run(["ffmpeg", "-v", "error", "-i", path,
                                         "-map", "0:v:0", "-f", "framemd5", "-"])
        self.assertEqual(frame_hashes(fast["video"]), frame_hashes(reference["video"]))
        self.assertEqual(fast["audio"].read_bytes(), reference["audio"].read_bytes())

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
