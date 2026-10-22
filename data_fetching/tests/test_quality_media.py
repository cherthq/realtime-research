"""Real FFmpeg/PyAV tests; learned geometry is mocked, never claimed validated."""

import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

from data_fetching.core import Source, write_jsonl
from data_fetching.pipeline import (
    build_export,
    candidate_records,
    fetch_one,
    review_template,
    summary,
)
from data_fetching.quality_audio import waveform_check
from data_fetching.quality_config import QualityConfig
from data_fetching.quality_export import package_hub
from data_fetching.quality_pipeline import process_quality
from data_fetching.quality_scan import frames, read_frames, scan_source, clean_windows
from data_fetching.tests.test_core import good_sample

AVAILABLE = (
    all(importlib.util.find_spec(name) for name in ("av", "cv2", "scenedetect"))
    and shutil.which("ffmpeg")
    and shutil.which("ffprobe")
)


@unittest.skipUnless(AVAILABLE, "PyAV/OpenCV/PySceneDetect and FFmpeg required")
class QualityMediaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.video = self.directory / "fixture.mp4"
        from data_fetching.media import run

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
                "sine=frequency=500:sample_rate=48000:duration=6",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-c:a",
                "aac",
                self.video,
            ]
        )

    @staticmethod
    def detector():
        detector = Mock()
        detector.detect.side_effect = lambda *args: {
            k: v for k, v in good_sample().items() if k != "time"
        }
        return detector

    def test_pts_decoder_and_all_frame_scan(self):
        from data_fetching.media import probe

        decoded = list(frames(self.video, 1, 5))
        self.assertEqual(len(decoded), 100)
        self.assertAlmostEqual(decoded[0][1]["time"], 1)
        self.assertAlmostEqual(decoded[-1][1]["end"], 5)
        info = probe(self.video)
        scan = scan_source(
            self.video, self.directory, 1, 5, info, QualityConfig(), self.detector()
        )
        rows = list(read_frames(self.directory / "frames.jsonl.gz"))
        self.assertEqual(scan["frame_count"], 100)
        self.assertEqual(len(rows), 100)
        self.assertEqual(scan["coverage_method"], "all_decoded_frames_actual_pts")

    def test_hard_cut_is_never_crossed(self):
        from data_fetching.media import probe, run

        video = self.directory / "cut.mp4"
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
                video,
            ]
        )
        config = QualityConfig(min_seconds=1)
        scan = scan_source(
            video, self.directory, 0, 6, probe(video), config, self.detector()
        )
        self.assertTrue(any(abs(c["time"] - 3) < 0.1 for c in scan["cuts"]))
        scan.update(width=1024, height=768)
        for rows, crop, parent in clean_windows(
            self.directory / "frames.jsonl.gz", scan, config
        ):
            self.assertFalse(rows[0]["time"] < 3 < rows[-1]["end"])

    def test_pipeline_pending_then_bound_review_export_and_hub_package(self):
        root = self.directory / "dataset"
        source = Source(id="fixture", path=str(self.video))
        fetch_one(root, source)
        with (
            patch(
                "data_fetching.vision.model_paths",
                return_value=({}, {"face": "test", "pose": "test"}),
            ),
            patch(
                "data_fetching.vision.Detector",
                side_effect=lambda *a, **k: self.detector(),
            ),
        ):
            result = process_quality(root, [source], QualityConfig(), {})
            self.assertFalse(any(row["state"] == "failed" for row in result), result)
            again = process_quality(root, [source], QualityConfig(), {})
            self.assertTrue(again[0]["cached"])
        candidate = candidate_records(root)[0]
        from data_fetching.quality_gallery import build_gallery

        gallery = build_gallery(root, root / "results" / "fixture")
        self.assertTrue(gallery.exists())
        self.assertIn("unverified", gallery.read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            build_gallery(root, root / "results" / "fixture")
        with self.assertRaises(ValueError):
            build_gallery(root, self.directory / "outside-dataset")
        self.assertEqual(candidate["schema_version"], 2)
        self.assertEqual(candidate["qc_disposition"], "needs_review")
        self.assertEqual(candidate["checks"]["speech"]["status"], "unverified")
        self.assertEqual(candidate["checks"]["framing"]["output_frame_count"], 150)
        reviews = review_template(root)
        review = reviews[0]
        review.update(
            status="approved",
            source_approved=True,
            license="synthetic fixture",
            license_evidence="generated in test",
            split="test",
            speaker_id="synthetic",
            conversation_id="fixture",
            reviewer="test-runner",
            reviewed_at="2026-10-22T00:00:00Z",
            include_short=True,
            redistribution_approved=True,
        )
        for decision in review["check_decisions"].values():
            decision.update(
                status="approved",
                reason="synthetic fixture contract test, not learned-model validation",
                evidence=["generated test case"],
            )
        path = self.directory / "reviews.jsonl"
        write_jsonl(path, reviews)
        exported = build_export(root, path)
        self.assertEqual(len(exported), 1)
        self.assertAlmostEqual(summary(root)["unique_candidate_hours"], 6 / 3600)
        package = self.directory / "hub-package"
        self.assertEqual(package_hub(root, path, package), 1)
        metadata = __import__("json").loads(
            (package / "test" / "metadata.jsonl").read_text()
        )
        self.assertTrue((package / "test" / metadata["file_name"]).exists())
        self.assertTrue((package / "COMPLETE.json").exists())
        if importlib.util.find_spec("datasets"):
            import os
            from datasets import Video, load_dataset

            with patch.dict(os.environ, {"HF_DATASETS_OFFLINE": "1"}):
                dataset = load_dataset(
                    "videofolder",
                    data_dir=str(package),
                    cache_dir=str(self.directory / "hf-cache"),
                )
            self.assertEqual(set(dataset), {"test"})
            self.assertEqual(len(dataset["test"]), 1)
            self.assertEqual(
                dataset["test"].cast_column("video", Video(decode=False))[0][
                    "interval_id"
                ],
                candidate["interval_id"],
            )
        from data_fetching.quality_backends import invoke

        with (
            patch(
                "data_fetching.vision.model_paths",
                return_value=({}, {"face": "test", "pose": "test"}),
            ),
            patch(
                "data_fetching.quality_pipeline.rendered_tracks",
                side_effect=AssertionError("render should be cached"),
            ),
            patch(
                "data_fetching.quality_pipeline.invoke",
                side_effect=lambda name, registry, job, directory: invoke(
                    name, {}, job, directory
                ),
            ),
        ):
            refreshed = process_quality(
                root, [source], QualityConfig(), {"vad": {"fingerprint": "new-backend"}}
            )
            self.assertFalse(any(r["state"] == "failed" for r in refreshed), refreshed)
        self.assertEqual(len(candidate_records(root)), 2)
        self.assertAlmostEqual(summary(root)["unique_candidate_hours"], 6 / 3600)

    def test_actual_waveform_statistics(self):
        from data_fetching.media import run

        audio = self.directory / "audio.wav"
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-i",
                self.video,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "48000",
                audio,
            ]
        )
        result = waveform_check(audio)
        self.assertEqual(result["sample_rate"], 48000)
        self.assertGreater(result["rms"], 0)
        self.assertEqual(result["near_full_scale_fraction"], 0)

    def test_brief_miss_stays_reviewable_and_is_never_measured_visibility(self):
        from data_fetching.media import probe

        detector = self.detector()
        calls = 0

        def detect(*args):
            nonlocal calls
            calls += 1
            return (
                {"faces": [], "poses": []}
                if calls == 8
                else {k: v for k, v in good_sample().items() if k != "time"}
            )

        detector.detect.side_effect = detect
        config = QualityConfig(min_seconds=1)
        scan = scan_source(
            self.video, self.directory, 0, 6, probe(self.video), config, detector
        )
        self.assertFalse(scan["blocked"])
        saved = list(read_frames(self.directory / "frames.jsonl.gz"))
        self.assertEqual(saved[7]["track_state"], "recovered_proposal")
        self.assertIn("detector_gap_review_required", saved[7]["uncertainties"])
        self.assertIsNone(saved[7]["face"])
        self.assertEqual(saved[7]["detections"], {"faces": [], "poses": []})
        scan.update(width=1024, height=768)
        for rows, crop, parent in clean_windows(
            self.directory / "frames.jsonl.gz", scan, config
        ):
            self.assertEqual(len(rows), 150)

    def test_fade_excludes_ramp_not_only_dark_midpoint(self):
        from data_fetching.media import probe, run

        video = self.directory / "fades.mp4"
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
                "color=green:size=1024x768:rate=25:duration=3",
                "-f",
                "lavfi",
                "-i",
                "color=black:size=1024x768:rate=25:duration=1",
                "-f",
                "lavfi",
                "-i",
                "color=green:size=1024x768:rate=25:duration=3",
                "-filter_complex",
                "[0:v]fade=t=out:st=2.25:d=0.75[a];[2:v]fade=t=in:st=0:d=0.75[b];[a][1:v][b]concat=n=3:v=1:a=0[v]",
                "-map",
                "[v]",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                video,
            ]
        )
        scan = scan_source(
            video, self.directory, 0, 7, probe(video), QualityConfig(), self.detector()
        )
        self.assertTrue(
            any(a <= 2.3 and b >= 4.7 for a, b in scan["blocked"]), scan["blocked"]
        )

    def test_static_dim_scene_is_not_automatically_a_fade(self):
        from data_fetching.media import probe, run

        video = self.directory / "dim.mp4"
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
                "color=0x181818:size=1024x768:rate=25:duration=3",
                "-c:v",
                "libx264",
                video,
            ]
        )
        scan = scan_source(
            video, self.directory, 0, 3, probe(video), QualityConfig(), self.detector()
        )
        self.assertEqual(scan["blocked"], [])

    def test_ambiguous_real_people_are_hard_boundaries_not_recovery(self):
        from data_fetching.media import probe
        from copy import deepcopy

        detector = self.detector()
        calls = 0

        def detect(*args):
            nonlocal calls
            calls += 1
            sample = good_sample()
            if calls == 8:
                sample["poses"].append(deepcopy(sample["poses"][0]))
            return {k: v for k, v in sample.items() if k != "time"}

        detector.detect.side_effect = detect
        scan = scan_source(
            self.video,
            self.directory,
            0,
            6,
            probe(self.video),
            QualityConfig(),
            detector,
        )
        self.assertTrue(any(a <= 0.28 and b >= 0.32 - 1e-7 for a, b in scan["blocked"]))

    def test_vfr_and_nonzero_origin_use_pts(self):
        from data_fetching.media import run

        video = self.directory / "vfr.mkv"
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-i",
                self.video,
                "-vf",
                "select='not(mod(n,2))+eq(mod(n,7),0)'",
                "-fps_mode",
                "vfr",
                "-c:v",
                "libx264",
                "-output_ts_offset",
                "5",
                video,
            ]
        )
        timing = [row for _, row in frames(video, 0, 5)]
        self.assertAlmostEqual(timing[0]["time"], 0, places=2)
        differences = {
            round(b["time"] - a["time"], 3) for a, b in zip(timing, timing[1:])
        }
        self.assertGreater(len(differences), 1)
        self.assertTrue(all(row["pts"] >= 5000 for row in timing))


if __name__ == "__main__":
    unittest.main()
