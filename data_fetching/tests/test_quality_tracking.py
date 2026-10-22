"""Regression fixtures for logo false positives, real people, recovery and crops."""

from copy import deepcopy
import unittest

from data_fetching.quality_audio import score_findings, speech_segments
from data_fetching.quality_config import QualityConfig
from data_fetching.quality_text import classify_text
from data_fetching.quality_tracking import (
    SubjectTracker,
    mapped_region,
    recover_rows,
    region_inside,
)
from data_fetching.tests.test_core import good_sample
from data_fetching.quality_worker import offline_inference


def detection():
    sample = good_sample()
    return {"faces": sample["faces"], "poses": sample["poses"]}


class TrackingTests(unittest.TestCase):
    def test_logo_does_not_erase_primary_face_or_split_a_proposal(self):
        d = detection()
        d["faces"].append({"box": [850, 300, 990, 460], "score": 0.55})
        result = SubjectTracker(QualityConfig()).assess(d, 1024, 768, 0)
        self.assertEqual(result["face"], d["faces"][0]["box"])
        self.assertEqual(result["flags"], [])
        self.assertIn("unassociated_secondary_face", result["uncertainties"])
        self.assertEqual(len(result["detections"]["faces"]), 2)

    def test_real_second_body_is_not_ignored_even_with_lower_face_score(self):
        d = detection()
        second = good_sample(shift=450)
        second["faces"][0]["score"] = 0.55
        d["faces"] += second["faces"]
        d["poses"] += second["poses"]
        result = SubjectTracker(QualityConfig()).assess(d, 1600, 900, 0)
        self.assertIn("multiple_or_ambiguous_person_associations", result["flags"])

    def test_strong_face_without_body_remains_uncertain(self):
        d = detection()
        d["faces"].append({"box": [850, 50, 990, 210], "score": 0.95})
        row = SubjectTracker(QualityConfig()).assess(d, 1024, 768, 0)
        self.assertIn("strong_secondary_face_review_required", row["uncertainties"])

    def test_actual_torso_profile_changes_required_region(self):
        torso = SubjectTracker(QualityConfig()).assess(detection(), 1024, 768, 0)
        head = SubjectTracker(QualityConfig(framing_profile="head_chest")).assess(
            detection(), 1024, 768, 0
        )
        self.assertGreater(torso["region"][3], head["region"][3] + 100)

    def test_source_limited_torso_is_explicit_and_never_invented(self):
        d = detection()
        # 320-pixel shoulders, 180 pixels genuinely available below them.
        for point in d["poses"][0]:
            point["y"] += 250
        d["faces"][0]["box"] = [420, 350, 580, 530]
        row = SubjectTracker(QualityConfig()).assess(d, 1024, 760, 0)
        self.assertEqual(row["flags"], [])
        self.assertIn("torso_extent_limited_by_source", row["uncertainties"])
        self.assertLessEqual(row["region"][3], 760)
        # Insufficient real torso is not rescued by clipping/padding the source.
        row = SubjectTracker(QualityConfig()).assess(d, 1024, 650, 0)
        self.assertIn("torso_region_outside_source", row["flags"])

    def test_missing_shoulders_specific_cause_is_retained(self):
        d = detection()
        d["poses"][0][11]["visibility"] = 0.1
        row = SubjectTracker(QualityConfig()).assess(d, 1024, 768, 0)
        self.assertEqual(row["uncertainties"], ["missing_shoulders"])

    def test_projection_is_not_a_second_torso_expansion(self):
        self.assertEqual(
            mapped_region([100, 100, 700, 600], [0, 0, 800, 600], 768, 576),
            [96, 96, 672, 576],
        )
        self.assertTrue(region_inside([96, 96, 672, 576], 768, 576))
        self.assertFalse(region_inside([96, 96, 672, 583], 768, 576))

    def test_recovery_never_bridges_a_boundary_or_long_gap(self):
        tracker = SubjectTracker(QualityConfig())
        left = tracker.assess(detection(), 1024, 768, 0)
        left.update(time=0, end=0.04)
        miss = tracker.assess({"faces": [], "poses": []}, 1024, 768, 0.04)
        miss.update(time=0.04, end=0.08)
        right = tracker.assess(detection(), 1024, 768, 0.08)
        right.update(time=0.08, end=0.12)
        output = list(recover_rows(deepcopy([left, miss, right]), 0.25, 0.15))
        self.assertEqual(output[1]["track_state"], "recovered_proposal")
        miss["boundary"] = True
        output = list(recover_rows([left, miss, right], 0.25, 0.15))
        self.assertIsNone(output[1]["region"])
        miss.pop("boundary")
        self.assertIsNone(
            list(recover_rows([left, miss, right], 0.01, 0.15))[1]["region"]
        )

    def test_invalid_profile_and_strings_cannot_bypass_validation(self):
        for args in (
            {"framing_profile": "nope"},
            {"scene_detector": "nope"},
            {"tracking_gap": 3},
            {"vad_release_threshold": 0.8},
            {"crop_safety": "0.1"},
        ):
            with self.assertRaises(ValueError):
                QualityConfig(**args)


class AudioTextPolicyTests(unittest.TestCase):
    def test_worker_network_guard_is_restored_after_inference(self):
        import socket

        original = socket.getaddrinfo
        with offline_inference():
            with self.assertRaisesRegex(RuntimeError, "Offline inference"):
                socket.getaddrinfo("example.com", 443)
            with socket.socket() as connection:
                with self.assertRaisesRegex(RuntimeError, "Offline inference"):
                    connection.connect(("127.0.0.1", 443))
        self.assertIs(socket.getaddrinfo, original)

    def test_short_offscreen_run_not_hidden_by_good_window_median(self):
        scores = [1.0] * 75
        scores[20:27] = [-1] * 7
        self.assertEqual(len(score_findings(scores, 0, [[0, 3]], 0, 0.2)), 1)
        self.assertEqual(score_findings(scores, 0, [[0, 0.5]], 0, 0.2), [])

    def test_vad_hysteresis_and_brief_silence(self):
        cfg = QualityConfig().to_dict()
        windows = [
            {"start": i / 10, "end": (i + 1) / 10, "probability": p}
            for i, p in enumerate([0, 0.9, 0.9, 0.4, 0.9, 0.1, 0.1, 0.1])
        ]
        result = speech_segments(windows, 0.8, cfg)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0][0], 0.07)
        self.assertAlmostEqual(result[0][1], 0.53)

    def test_text_on_clothes_is_not_automatically_a_caption(self):
        polygon = [[300, 300], [500, 300], [500, 350], [300, 350]]
        result = classify_text(
            polygon, 768, 576, [300, 50, 500, 250], [100, 0, 650, 576]
        )
        self.assertEqual(result["category"], "subject_text_or_overlay_ambiguous")
        self.assertEqual(result["semantic_status"], "unverified")
        caption = [[100, 500], [650, 500], [650, 550], [100, 550]]
        self.assertEqual(
            classify_text(caption, 768, 576)["category"],
            "possible_caption_or_lower_third",
        )


if __name__ == "__main__":
    unittest.main()
