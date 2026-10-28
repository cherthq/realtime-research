"""CPU-only tests. No downloads, real TTS synthesis or avatar inference."""

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.prepare_zero_shot_assets import (
    REPO,
    assemble,
    asset_path,
    check_audio,
    load_json,
    samples_for,
    to_pcm,
    write_json,
)


class AssetPreparationTests(unittest.TestCase):
    def setUp(self):
        self.suite = load_json(REPO / "configs/zero_shot_suite.json")
        self.settings = self.suite["audio_preparation"]

    def fake_tts(self, text, speed):
        self.assertTrue(text.strip())
        self.assertIn(speed, (1.0, 1.25))
        return np.full(2400, 0.2, dtype=np.float32), 24000

    def test_all_scenarios_have_exact_zero_pause_samples(self):
        for scenario in self.suite["scenarios"]:
            with self.subTest(scenario=scenario["id"]):
                pcm, timeline = assemble(scenario, self.settings, self.fake_tts)
                check_audio(pcm, timeline, scenario, self.settings)
                self.assertEqual(pcm.dtype, np.dtype("<i2"))
                self.assertEqual(timeline[0]["end_sample"], 12000)
                self.assertEqual(len(pcm) - timeline[-1]["start_sample"], 12000)
                for part in timeline:
                    if part.get("role") == "scripted_pause":
                        length = part["end_sample"] - part["start_sample"]
                        self.assertIn(length, (72000, 96000))
                        self.assertFalse(
                            pcm[part["start_sample"] : part["end_sample"]].any()
                        )

    def test_fast_scenario_passes_speed_to_tts(self):
        calls = []

        def tts(text, speed):
            calls.append(speed)
            return self.fake_tts(text, speed)

        assemble(self.suite["scenarios"][3], self.settings, tts)
        self.assertEqual(calls, [1.25])

    def test_corrupt_pause_is_rejected(self):
        scenario = self.suite["scenarios"][4]
        pcm, timeline = assemble(scenario, self.settings, self.fake_tts)
        pause = next(part for part in timeline if part.get("role") == "scripted_pause")
        pcm[pause["start_sample"] + 10] = 1
        with self.assertRaisesRegex(ValueError, "not exactly zero"):
            check_audio(pcm, timeline, scenario, self.settings)

    def test_mismatched_speech_and_truncation_are_rejected(self):
        scenario = self.suite["scenarios"][0]
        pcm, timeline = assemble(scenario, self.settings, self.fake_tts)
        changed = copy.deepcopy(timeline)
        changed[1]["text"] = "Wrong reply."
        with self.assertRaises(ValueError):
            check_audio(pcm, changed, scenario, self.settings)
        with self.assertRaises(ValueError):
            check_audio(pcm[:-1], timeline, scenario, self.settings)

    def test_invalid_pcm_and_silence_are_rejected(self):
        for audio in ([], [float("nan")], [float("inf")], [0.0], [2.0], [[0.1, 0.2]]):
            with self.subTest(audio=audio), self.assertRaises(ValueError):
                to_pcm(audio, 24000, 48000, 0.9)
        for seconds in (0, -1, float("inf"), 0.000001):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                samples_for(seconds, 48000)

    def test_path_escape_and_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(asset_path(root, "audio/one.wav"), root / "audio/one.wav")
            with self.assertRaises(ValueError):
                asset_path(root, "../outside.wav")
            with self.assertRaises(ValueError):
                asset_path(root, root / "absolute.wav")
            target = root / "frozen.json"
            write_json(target, {"frozen": True})
            with self.assertRaises(FileExistsError):
                write_json(target, {"frozen": False})
            self.assertEqual(load_json(target), {"frozen": True})


if __name__ == "__main__":
    unittest.main()
