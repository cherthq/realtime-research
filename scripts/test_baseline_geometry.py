"""Geometry tests without importing any GPU model dependencies."""

from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
import unittest

from scripts.baseline_geometry import (
    configure_flashtalk,
    fixed_liveavatar_size,
    load_geometry,
    validate_geometry,
)
from scripts.run_native_baseline import build_command, REPO
from scripts.run_4x3_baselines import check_smoke
from scripts.run_flashtalk_baseline import digest


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.profile = load_geometry()

    def test_profile_is_landscape_4x3_and_patch_aligned(self):
        self.assertEqual((self.profile["width"], self.profile["height"]), (640, 480))
        self.assertFalse(self.profile["post_generation_resize"])

    def test_portrait_or_stretched_or_rotated_output_is_rejected(self):
        validate_geometry(
            {"width": 640, "height": 480, "sample_aspect_ratio": "1:1"}, self.profile
        )
        for stream in (
            {"width": 416, "height": 720},
            {"width": 640, "height": 480, "sample_aspect_ratio": "3:4"},
            {"width": 640, "height": 480, "side_data_list": [{"rotation": 90}]},
        ):
            with self.assertRaises(ValueError):
                validate_geometry(stream, self.profile)

    def test_flashtalk_updates_actual_preparation_global(self):
        module = SimpleNamespace(
            infer_params={"width": 416, "height": 720, "sample_steps": 4},
            target_size=(720, 416),
        )
        original = module.infer_params
        configure_flashtalk(module, self.profile)
        self.assertIs(module.infer_params, original)
        self.assertEqual(module.target_size, (480, 640))
        self.assertEqual(module.infer_params["sample_steps"], 4)

    def test_liveavatar_bypasses_automatic_64_pixel_rounding(self):
        resolver = fixed_liveavatar_size(self.profile)
        self.assertEqual(
            resolver(None, None, 640 * 480, "portrait.png", None), (480, 640)
        )
        with self.assertRaises(ValueError):
            resolver(None, None, 704 * 384, "portrait.png", None)

    def test_both_cli_models_request_exact_generation_size(self):
        for model in ("liveavatar", "liveact"):
            command = build_command(
                model,
                REPO / "env/python",
                REPO / "vendor" / model,
                REPO / "checkpoints",
                REPO,
                {
                    "reference_image": "image.png",
                    "audio": "audio.wav",
                    "prompt": "hello",
                },
                REPO / "runs/test",
                0,
                10,
            )
            self.assertEqual(command[command.index("--size") + 1], "640*480")
            if model == "liveavatar":
                self.assertIn(
                    str(REPO / "scripts/liveavatar_geometry_entry.py"), command
                )

    def test_smoke_gate_requires_matching_shape_and_unchanged_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary)
            video = case / "res_video.mp4"
            video.write_bytes(b"unit-test fixture; not generated media")
            record = {
                "status": "passed_smoke_output_checks",
                "geometry_profile": self.profile,
                "output_sha256": digest(video),
                "output_probe": {
                    "streams": [{"codec_type": "video", "width": 640, "height": 480}]
                },
            }
            metadata = case / "metadata.json"
            metadata.write_text(json.dumps(record))
            check_smoke(case, "flashtalk", self.profile)
            record["output_probe"]["streams"][0]["width"] = 416
            metadata.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "Output geometry"):
                check_smoke(case, "flashtalk", self.profile)
            record["output_probe"]["streams"][0]["width"] = 640
            metadata.write_text(json.dumps(record))
            video.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "changed"):
                check_smoke(case, "flashtalk", self.profile)

    def test_recovered_native_smoke_is_bound_to_failed_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary)
            original = case / "result.json"
            original.write_text('{"status":"failed"}')
            before = original.read_bytes()
            video = case / "video.mp4"
            video.write_bytes(b"unit-test fixture; not generated media")
            recovery = {
                "status": "recovered_functional_output_checks",
                "original_result_sha256": digest(original),
                "geometry_profile": self.profile,
                "output_video": video.name,
                "output_sha256": digest(video),
                "output_probe": {
                    "streams": [{"codec_type": "video", "width": 640, "height": 480}]
                },
            }
            (case / "recovery.json").write_text(json.dumps(recovery))
            check_smoke(case, "liveavatar", self.profile)
            self.assertEqual(original.read_bytes(), before)
            original.write_text('{"status":"failed","changed":true}')
            with self.assertRaisesRegex(ValueError, "bound"):
                check_smoke(case, "liveavatar", self.profile)


if __name__ == "__main__":
    unittest.main()
