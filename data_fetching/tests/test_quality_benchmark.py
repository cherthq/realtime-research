import unittest

from data_fetching.quality_benchmark import evaluate


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "id": "a",
            "source_id": "source",
            "source": {"sha256": "original"},
            "start": 0,
            "end": 6,
            "qc_disposition": "needs_review",
        }
        self.labels = [
            {
                "source_id": "source",
                "source_sha256": "original",
                "start": a,
                "end": b,
                "label": label,
                "partition": "test",
                "group_id": "source-speaker",
                "reviewer": "fixture",
                "evidence": ["synthetic contract fixture"],
            }
            for a, b, label in [(0, 5, "usable"), (5, 10, "unusable")]
        ]

    def test_time_weighted_and_revisions_do_not_double_count(self):
        report = evaluate([self.record, self.record], self.labels)
        test = report["partitions"]["test"]
        self.assertEqual(test["usable_retained_seconds"], 5)
        self.assertEqual(test["unusable_retained_seconds"], 1)
        self.assertAlmostEqual(test["retained_precision"], 5 / 6)
        self.assertEqual(test["usable_recall"], 1)

    def test_candidates_are_not_eligible_or_approved(self):
        self.assertIsNone(
            evaluate([self.record], self.labels, "eligible")["partitions"]["test"][
                "retained_precision"
            ]
        )
        self.assertEqual(
            evaluate([self.record], self.labels, "approved", ["a"])["partitions"][
                "test"
            ]["unusable_retained_seconds"],
            1,
        )

    def test_leakage_overlap_and_hash_fail_closed(self):
        for change in (
            {"partition": "calibration"},
            {"start": 4},
            {"source_sha256": "different"},
        ):
            labels = [self.labels[0], {**self.labels[1], **change}]
            with self.assertRaises(ValueError):
                evaluate([self.record], labels)
        labels = [
            self.labels[0],
            {
                **self.labels[1],
                "group_id": "mislabelled-group",
                "partition": "calibration",
            },
        ]
        with self.assertRaisesRegex(ValueError, "leakage"):
            evaluate([self.record], labels)


if __name__ == "__main__":
    unittest.main()
