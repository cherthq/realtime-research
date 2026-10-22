"""QC contracts and failure policy; no models or network calls."""

import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data_fetching.__main__ import main
from data_fetching.core import write_json, write_jsonl
from data_fetching.metadata import (REQUIRED_CHECKS, check, disposition, interval_id,
    ranges_union, review_binding, subtract_ranges, unique_hours, validate_check,
    validate_review)
from data_fetching.quality_audio import speech_coverage, speech_windows
from data_fetching.quality_backends import invoke, load_registry
from data_fetching.quality_config import QualityConfig
from data_fetching.quality_export import import_discovery
from data_fetching.quality_pipeline import containment
from data_fetching.quality_scan import iou


def record_v2():
    checks = {k: check("pass", "test_fixture_only", [(0, 60)]) for k in REQUIRED_CHECKS}
    return {"schema_version": 2, "id": "fixture", "interval_id": "interval", "processing_id": "processing",
            "parent_span_id": "span", "track_id": "0", "start": 0, "end": 60,
            "source": {"sha256": "source", "media": {"duration": 120}}, "crop": [0, 0, 768, 576],
            "checks": checks, "qc_disposition": "eligible", "artifacts": {},
            "duration_bucket": "long", "speech_bucket": "talking"}


class ContractTests(unittest.TestCase):
    def test_union_and_subtraction(self):
        self.assertEqual(ranges_union([(2, 6), (0, 3), (6, 8)]), [[0, 8]])
        self.assertEqual(list(subtract_ranges(0, 10, [(2, 3), (2.5, 5), (8, 12)])), [(0, 2), (5, 8)])
        for span in ((2, 1), (-1, 5), (0, float("nan")), (False, 4)):
            with self.assertRaises(ValueError):
                ranges_union([span])

    def test_unknown_is_not_pass(self):
        record = record_v2()
        record["checks"]["speech"] = check("unverified", "vad", reason="missing")
        self.assertEqual(disposition(record["checks"]), "needs_review")
        record["checks"]["speech"] = check("error", "vad", reason="crashed")
        self.assertEqual(disposition(record["checks"]), "needs_review")
        record["checks"]["speech"] = check("fail", "vad", [(0, 1)])
        self.assertEqual(disposition(record["checks"]), "rejected")

    def test_check_validation(self):
        for value in ({"status": "pass", "method": "test", "analyzed_ranges": []},
                      {"status": "not_applicable", "method": "test"},
                      {"status": "oops", "method": "test"},
                      {"status": "flag", "method": "test", "score": float("inf")}):
            with self.assertRaises(ValueError):
                validate_check(value)

    def test_unique_hours_do_not_count_revisions_or_overlaps(self):
        a = record_v2()
        b = {**a, "processing_id": "second", "start": 30, "end": 90}
        self.assertAlmostEqual(unique_hours([a, a, b]), 90 / 3600)

    def test_interval_identity_independent_of_processing(self):
        self.assertEqual(interval_id("hash", 0, 100, "1/25"), interval_id("hash", 0, 100, "1/25"))
        self.assertNotEqual(interval_id("hash", 0, 100, "1/25"), interval_id("hash", 1, 100, "1/25"))

    def test_review_binding_and_utc(self):
        record = record_v2()
        review = {"review_binding": review_binding(record), "reviewer": "test",
                  "reviewed_at": "2026-10-22T00:00:00Z"}
        validate_review(record, review)
        for time in (None, "yesterday", "2026-10-22T10:00:00", "2026-10-22T10:00:00+08:00"):
            with self.assertRaises(ValueError):
                validate_review(record, {**review, "reviewed_at": time})
        record["crop"][0] = 2
        with self.assertRaisesRegex(ValueError, "Stale"):
            validate_review(record, review)

    def test_overrides_need_evidence_and_cannot_override_corrupt_render(self):
        record = record_v2()
        record["checks"]["lip_sync"] = check("unverified", "sync", reason="missing")
        record["qc_disposition"] = "needs_review"
        review = {"review_binding": review_binding(record), "reviewer": "tester", "reviewed_at": "2026-10-22T00:00:00Z"}
        with self.assertRaisesRegex(ValueError, "Unresolved"):
            validate_review(record, review)
        review["check_decisions"] = {"lip_sync": {"status": "approved", "reason": "watched entire clip", "evidence": ["review-log"]}}
        validate_review(record, review)
        record["checks"]["render_validation"] = check("error", "ffmpeg")
        review["review_binding"] = review_binding(record)
        review["check_decisions"]["render_validation"] = review["check_decisions"]["lip_sync"]
        with self.assertRaisesRegex(ValueError, "cannot be overridden"):
            validate_review(record, review)

    def test_short_and_listening_are_explicit(self):
        record = record_v2()
        record["duration_bucket"], record["speech_bucket"] = "short", "listening"
        review = {"review_binding": review_binding(record), "reviewer": "tester", "reviewed_at": "2026-10-22T00:00:00Z"}
        with self.assertRaisesRegex(ValueError, "Short"):
            validate_review(record, review)
        review["include_short"] = True
        with self.assertRaisesRegex(ValueError, "Listening"):
            validate_review(record, review)
        validate_review(record, {**review, "include_listening": True})

    def test_profile_rejects_invalid_settings(self):
        for values in ({"width": 767}, {"window_stride": 0}, {"max_seconds": 1},
                       {"ocr_fps": 0}, {"track_iou": 2}, {"head_margin": float("nan")}):
            with self.assertRaises(ValueError):
                QualityConfig(**values)

    def test_full_frame_containment_and_track_iou(self):
        rows = [{"region": [10, 20, 100, 200], "flags": []}]
        self.assertEqual(containment(rows, [0, 0, 768, 576]), 10)
        with self.assertRaises(ValueError):
            containment(rows, [20, 0, 768, 576])
        with self.assertRaises(ValueError):
            containment([{**rows[0], "flags": ["missing_face"]}], [0, 0, 768, 576])
        self.assertEqual(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1)
        self.assertEqual(iou([0, 0, 10, 10], [20, 20, 30, 30]), 0)

    def test_speech_windows_cover_tail_and_leave_pauses(self):
        windows = speech_windows(7.5, [(0.1, 0.2), (7.3, 7.5)])
        self.assertIn((4.5, 7.5), windows)
        self.assertNotIn((3, 6), windows)
        self.assertEqual(speech_coverage([(0, 1), (7, 7.5)], windows), 1)
        self.assertEqual(speech_windows(2, [(0, 2)]), [])

    def test_missing_registry_never_launches_worker(self):
        with tempfile.TemporaryDirectory() as temp, patch("data_fetching.quality_backends.subprocess.run") as launch:
            result = invoke("talknet", {}, {"duration": 10}, Path(temp))
        launch.assert_not_called()
        self.assertEqual(result["active_speaker"]["status"], "unverified")

    def test_invalid_worker_response_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            entry = {"python": "python", "fingerprint": "backend"}
            def launch(command, **kwargs):
                write_json(Path(command[-1]), {"protocol_version": 1, "job_id": "wrong", "backend": "vad", "checks": {}})
            with patch("data_fetching.quality_backends.subprocess.run", side_effect=launch):
                result = invoke("vad", {"vad": entry}, {"duration": 10}, directory)
            self.assertEqual(result["speech"]["status"], "error")

    def test_registry_requires_assets_and_permissions(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "workers.json"
            write_json(path, {"schema_version": 1, "backends": {"vad": {"enabled": True}}})
            with self.assertRaisesRegex(ValueError, "terms"):
                load_registry(path)
            self.assertEqual(load_registry(), {})

    def test_discovery_import_preserves_provenance_without_approval(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "discovery.jsonl"
            row = {"url": "https://youtube.com/watch?v=abcdefghijk", "extra_upstream": "preserve",
                   "clean_intervals": [{"start": 0, "end": 60}, {"start": 0, "end": 60}]}
            write_jsonl(path, [row, row])
            sources, provenance = import_discovery(path)
            self.assertEqual(len(sources), 1)
            self.assertEqual(sum(len(p["proposed_intervals"]) for p in provenance), 1)
            self.assertFalse(provenance[0]["source_approved"])
            self.assertEqual(provenance[0]["upstream"]["extra_upstream"], "preserve")

    def test_quality_dry_run_no_writes_and_no_backends(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest = directory / "sources.jsonl"
            write_jsonl(manifest, [{"id": "fixture", "path": "absent.mp4"}])
            root = directory / "dataset"
            with patch("sys.stdout", new=io.StringIO()), patch("data_fetching.quality_backends.subprocess.run") as launch:
                result = main(["--root", str(root), "process", "--manifest", str(manifest),
                               "--profile", "quality-long", "--workers", "absent.json", "--dry-run"])
            self.assertEqual(result, 0)
            self.assertFalse(root.exists())
            launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
