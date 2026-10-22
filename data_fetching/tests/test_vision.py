"""Adapter tests without model downloads; optional real scene backend checks."""

import importlib.util
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from data_fetching.core import Config
from data_fetching.media import probe, run
from data_fetching.vision import Detector, sample_clip, scene_intervals
from data_fetching.tests.test_core import good_sample


class AdapterTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("numpy") and importlib.util.find_spec("cv2"),
        "NumPy/OpenCV required",
    )
    def test_yunet_coordinates_map_back_from_analysis_resolution(self):
        import numpy as np

        detector = Detector.__new__(Detector)
        detector.mp = SimpleNamespace(
            Image=Mock(), ImageFormat=SimpleNamespace(SRGB="rgb")
        )
        detector.yunet = Mock()
        detector.yunet.detect.return_value = (
            None,
            np.array([[10, 20, 30, 40] + [10, 20] * 5 + [0.9]], dtype=np.float32),
        )
        detector.pose = Mock()
        detector.pose.detect.return_value = SimpleNamespace(pose_landmarks=[])
        result = detector.detect(np.zeros((1080, 1920, 3), dtype=np.uint8))
        detector.yunet.setInputSize.assert_called_once_with((960, 540))
        self.assertEqual(result["faces"][0]["box"], [20, 40, 80, 120])
        self.assertEqual(result["faces"][0]["keypoints"][0], [20, 40])

    def test_landmark_conversion(self):
        detector = Detector.__new__(Detector)
        detector.mp = SimpleNamespace(
            Image=Mock(), ImageFormat=SimpleNamespace(SRGB="rgb")
        )
        face = SimpleNamespace(
            bounding_box=SimpleNamespace(
                origin_x=100, origin_y=50, width=200, height=250
            ),
            categories=[SimpleNamespace(score=0.8)],
            keypoints=[SimpleNamespace(x=0.25, y=0.25)],
        )
        points = [SimpleNamespace(x=0.5, y=0.5, visibility=0.9, presence=0.8)] * 33
        detector.face = Mock()
        detector.face.detect.return_value = SimpleNamespace(detections=[face])
        detector.pose = Mock()
        detector.pose.detect.return_value = SimpleNamespace(pose_landmarks=[points])
        result = detector.detect(SimpleNamespace(shape=(720, 1280, 3)))
        self.assertEqual(result["faces"][0]["box"], [100, 50, 300, 300])
        self.assertEqual(result["faces"][0]["keypoints"], [[320, 180]])
        self.assertEqual(result["poses"][0][11]["x"], 640)
        self.assertEqual(result["poses"][0][11]["y"], 360)
        detector.close()
        detector.face.close.assert_called_once()
        detector.pose.close.assert_called_once()


OPTIONAL = (
    importlib.util.find_spec("scenedetect") is not None
    and importlib.util.find_spec("cv2") is not None
    and bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
)


@unittest.skipUnless(OPTIONAL, "Optional scene/OpenCV dependencies required")
class SceneTests(unittest.TestCase):
    def test_local_pipeline_with_real_media_and_mocked_neural_outputs(self):
        from data_fetching.core import Source, write_jsonl
        from data_fetching.pipeline import (
            build_export,
            candidate_records,
            fetch_one,
            process_sources,
            review_template,
        )

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            video = directory / "consented_synthetic_fixture.mp4"
            root = directory / "dataset"
            run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-nostdin",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=green:size=1024x768:rate=25:duration=6",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=800:sample_rate=48000:duration=6",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-c:a",
                    "aac",
                    video,
                ]
            )
            source = Source(id="synthetic", path=str(video))
            self.assertEqual(fetch_one(root, source), "copied")
            fake = Mock()
            fake.detect.side_effect = lambda rgb: {
                key: value for key, value in good_sample().items() if key != "time"
            }
            with (
                patch(
                    "data_fetching.vision.model_paths",
                    return_value=({}, {"face": "test", "pose": "test"}),
                ),
                patch("data_fetching.vision.Detector", return_value=fake),
            ):
                report = process_sources(root, [source], Config())
                self.assertEqual(report[0]["state"], "candidate", report)
                record = candidate_records(root)[0]
                self.assertEqual(len(record["artifacts"]), 7)
                self.assertEqual(len(review_template(root)), 1)
                reviews = directory / "reviews.jsonl"
                write_jsonl(
                    reviews,
                    [
                        {
                            "id": record["id"],
                            "status": "approved",
                            "source_approved": True,
                            "license": "synthetic test fixture",
                            "license_evidence": "generated in test",
                            "split": "test",
                            "speaker_id": "synthetic",
                            "conversation_id": "fixture",
                        }
                    ],
                )
                self.assertEqual(len(build_export(root, reviews)), 1)
                again = process_sources(root, [source], Config())
                self.assertIs(again[0]["cached"], True)
            self.assertEqual(fake.detect.call_count, 12)

    def test_actual_scene_cuts_and_previews(self):
        from data_fetching.vision import save_previews

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            video = directory / "cuts.mp4"
            run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-nostdin",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=red:size=1024x768:rate=25:duration=3",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=blue:size=1024x768:rate=25:duration=3",
                    "-filter_complex",
                    "[0:v][1:v]concat=n=2:v=1:a=0[v]",
                    "-map",
                    "[v]",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    video,
                ]
            )
            scenes = scene_intervals(video, 0, 6, Config())
            self.assertEqual(len(scenes), 2)
            self.assertAlmostEqual(scenes[0][1], 3, places=1)
            self.assertAlmostEqual(scenes[1][0], 3, places=1)
            info = probe(video)
            detector = Mock()
            detector.detect.return_value = {"faces": [], "poses": []}
            samples = sample_clip(video, 0, 3, info, Config(), detector)
            self.assertEqual(len(samples), 6)
            self.assertLess(samples[-1]["time"], 3)
            self.assertEqual(detector.detect.call_count, 6)
            previews = save_previews(
                video, directory, samples, [128, 96, 768, 576], info
            )
            self.assertEqual(len(previews), 3)
            self.assertTrue(all(path.stat().st_size > 1000 for path in previews))

    def test_single_scene_and_bounded_scan(self):
        with tempfile.TemporaryDirectory() as temp:
            video = Path(temp) / "plain.mp4"
            run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-nostdin",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=red:size=1024x768:rate=25:duration=6",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    video,
                ]
            )
            scenes = scene_intervals(video, 1, 5, Config())
            self.assertEqual(len(scenes), 1)
            self.assertAlmostEqual(scenes[0][0], 1, places=1)
            self.assertAlmostEqual(scenes[0][1], 5, places=1)


if __name__ == "__main__":
    unittest.main()
