"""CPU-only validation tests, independent of model packages and cloud access."""

import json
from pathlib import Path
import tempfile
import unittest
import wave

from scripts.run_flashtalk_baseline import digest, input_path, load_input


class BaselineInputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "image.png").write_bytes(b"synthetic-file-hash-fixture")
        with wave.open(str(self.root / "audio.wav"), "wb") as audio:
            audio.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
            audio.writeframes(b"\x01\x00" * 48000)
        self.row = {
            "id": "ref01__s01",
            "reference_image": "image.png",
            "audio": "audio.wav",
            "reference_image_sha256": digest(self.root / "image.png"),
            "audio_sha256": digest(self.root / "audio.wav"),
        }
        self.freeze([self.row])

    def freeze(self, rows):
        (self.root / "inputs.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows)
        )
        (self.root / "assets.json").write_text(
            json.dumps({"inputs_sha256": digest(self.root / "inputs.jsonl")})
        )

    def test_frozen_input_and_duration(self):
        row, duration = load_input(self.root, "ref01__s01")
        self.assertEqual(row, self.row)
        self.assertEqual(duration, 1.0)

    def test_changed_media_or_manifest_are_rejected(self):
        (self.root / "image.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Changed input"):
            load_input(self.root, "ref01__s01")
        (self.root / "inputs.jsonl").write_text("{}")
        with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
            load_input(self.root, "ref01__s01")

    def test_missing_or_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            load_input(self.root, "missing")
        self.freeze([self.row, self.row])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            load_input(self.root, "ref01__s01")

    def test_paths_cannot_escape_package(self):
        with self.assertRaises(ValueError):
            input_path(self.root, "../outside.wav")
        with self.assertRaises(ValueError):
            input_path(self.root, str(self.root / "audio.wav"))


if __name__ == "__main__":
    unittest.main()
