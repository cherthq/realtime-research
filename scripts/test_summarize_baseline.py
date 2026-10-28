"""CPU-only checks for missing, validated and tampered baseline outputs."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_flashtalk_baseline import digest
from scripts.summarize_baseline import collect, summarize


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.assets = self.root / "assets"
        self.assets.mkdir()
        (self.assets / "inputs.jsonl").write_text('{"id":"ref01__test"}\n')
        (self.assets / "suite.json").write_text('{"seeds":[0]}')

    def test_missing_metrics_remain_null(self):
        rows = collect(self.root, self.assets)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["status"] == "not_started" for row in rows))
        self.assertTrue(all(row["generation_fps"] is None for row in rows))
        summary = summarize(self.root, self.assets)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["counts"], {"not_started": 3})
        self.assertTrue((self.root / "outputs.csv").is_file())

    def test_tampered_output_is_never_counted(self):
        case = self.root / "flashtalk/ref01__test__seed0"
        case.mkdir(parents=True)
        video = case / "res_video.mp4"
        video.write_bytes(b"fixture output; no inference performed")
        (case / "metadata.json").write_text(
            json.dumps(
                {
                    "status": "passed_smoke_output_checks",
                    "model_reused": True,
                    "output_sha256": digest(video),
                }
            )
        )
        (case / "metrics.json").write_text('{"generation_fps":4.5}')
        row = collect(self.root, self.assets)[0]
        self.assertEqual(row["status"], "validated_output")
        self.assertEqual(row["timing_condition"], "warm")
        self.assertIsNone(row["lip_sync_score"])
        video.write_bytes(b"changed")
        row = collect(self.root, self.assets)[0]
        self.assertEqual(row["status"], "audit_failed")
        self.assertIsNone(row["output_video"])

    def test_incomplete_record_is_not_a_zero_metric(self):
        case = self.root / "liveact/ref01__test__seed0"
        case.mkdir(parents=True)
        (case / "result.json").write_text('{"status":')
        row = collect(self.root, self.assets)[2]
        self.assertEqual(row["status"], "audit_failed")
        self.assertIsNone(row["worker_wall_seconds"])


if __name__ == "__main__":
    unittest.main()
