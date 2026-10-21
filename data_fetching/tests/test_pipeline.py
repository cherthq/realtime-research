from contextlib import ExitStack
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data_fetching.__main__ import main, write_new
from data_fetching.core import Config, Source, file_hash, read_json, write_json, write_jsonl
from data_fetching.pipeline import (build_export, candidate_records, dataset_lock,
                                   fetch_one, fetch_sources, install_models, process_sources, review_template)
from data_fetching.tests.test_core import good_sample


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "dataset"
        self.local = self.base / "input.mp4"
        self.local.write_bytes(b"test-only media stub")
        self.source = Source(id="recording", path=str(self.local), license="unknown")
        self.info = {"width": 1280, "height": 720, "fps": 25, "duration": 10,
                     "video_start": 0, "video_index": 0, "audio_index": 1}

    def fetch(self):
        with patch("data_fetching.pipeline.probe", return_value=self.info):
            return fetch_one(self.root, self.source)

    def processing_mocks(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch("data_fetching.vision.model_paths", return_value=({}, {"face": "f", "pose": "p"})))
        stack.enter_context(patch("data_fetching.vision.runtime_versions", return_value={"mock": "1"}))
        stack.enter_context(patch("data_fetching.pipeline.run", return_value="ffmpeg test\n"))
        stack.enter_context(patch("data_fetching.pipeline.probe", return_value=self.info))
        stack.enter_context(patch("data_fetching.pipeline.validate_clip"))
        stack.enter_context(patch("data_fetching.vision.Detector"))
        scenes = stack.enter_context(patch("data_fetching.vision.scene_intervals", return_value=[(0, 10)]))
        samples = stack.enter_context(patch("data_fetching.vision.sample_clip", side_effect=lambda *a: [good_sample(), good_sample(10)]))

        def previews(path, directory, samples, crop, info):
            result = directory / "preview_first.jpg"
            result.write_bytes(b"preview stub")
            return [result]

        def render(path, directory, start, end, crop, config, info):
            result = {"video": directory / "video.mp4", "audio": directory / "audio.wav",
                      "reference_image": directory / "reference.png"}
            for name, file in result.items():
                file.write_bytes(name.encode())
            return result

        stack.enter_context(patch("data_fetching.vision.save_previews", side_effect=previews))
        rendered = stack.enter_context(patch("data_fetching.pipeline.render_clip", side_effect=render))
        return samples, rendered, scenes

    def candidate(self):
        self.fetch()
        self.processing_mocks()
        process_sources(self.root, [self.source], Config())
        return candidate_records(self.root)[0]

    def approved_review(self, record):
        return {"id": record["id"], "status": "approved", "source_approved": True,
                "license": "permission agreement", "license_evidence": "internal permission record 123",
                "split": "train", "speaker_id": "person1", "conversation_id": "call1"}

    def test_local_fetch_and_resume_hash_validation(self):
        self.assertEqual(self.fetch(), "copied")
        self.assertEqual(self.fetch(), "cached")
        record = read_json(self.root / "sources" / "recording.json")
        self.assertEqual(record["sha256"], file_hash(self.local))
        self.local.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.fetch()

    def test_changed_definition_is_not_overwritten(self):
        self.fetch()
        with self.assertRaisesRegex(ValueError, "definition changed"):
            fetch_one(self.root, replace(self.source, start=1))

    def test_fetch_failures_are_reported(self):
        report = fetch_sources(self.root, [replace(self.source, path=str(self.base / "missing.mp4"))])
        self.assertEqual(report[0]["status"], "failed")
        self.assertFalse((self.root / "sources" / "recording.json").exists())

    def test_lock_does_not_remove_other_writer_lock(self):
        with dataset_lock(self.root):
            with self.assertRaises(RuntimeError):
                with dataset_lock(self.root):
                    self.fail("second writer acquired lock")
            self.assertTrue((self.root / ".pipeline.lock").exists())
        self.assertFalse((self.root / ".pipeline.lock").exists())

    def test_explicit_model_setup_checks_hashes_on_reuse(self):
        with patch("data_fetching.pipeline.urlopen", side_effect=lambda *a, **kw: io.BytesIO(b"model stub" * 1000)) as network:
            manifest = install_models(self.root)
            self.assertEqual(network.call_count, 2)
            self.assertEqual(set(manifest), {"face", "pose"})
            install_models(self.root)
            self.assertEqual(network.call_count, 2)
            (self.root / manifest["face"]["path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "differs"):
                install_models(self.root)
            self.assertEqual(network.call_count, 2)

    def test_dry_run_no_network_or_output_writes(self):
        manifest = self.base / "sources.jsonl"
        write_jsonl(manifest, [self.source.to_dict()])
        with patch("sys.stdout", new=io.StringIO()), patch("data_fetching.pipeline.urlopen") as network:
            for command in ("fetch", "process"):
                self.assertEqual(main(["--root", str(self.root), command, "--manifest", str(manifest), "--dry-run"]), 0)
            self.assertEqual(main(["--root", str(self.root), "models", "--dry-run"]), 0)
        network.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_pending_review_and_explicit_export(self):
        record = self.candidate()
        reviews = self.base / "reviews.jsonl"
        rows = review_template(self.root)
        self.assertEqual(rows[0]["status"], "pending")
        self.assertIs(rows[0]["source_approved"], False)
        write_jsonl(reviews, rows)
        self.assertEqual(build_export(self.root, reviews), [])
        approved = self.approved_review(record)
        write_jsonl(reviews, [approved])
        exported = build_export(self.root, reviews)
        self.assertEqual(exported[0]["split"], "train")
        self.assertFalse(Path(exported[0]["video"]).is_absolute())

    def test_approval_rejects_missing_permission_license_identity_split(self):
        record = self.candidate()
        reviews = self.base / "reviews.jsonl"
        for changes in ({"source_approved": False}, {"license": "unknown"},
                        {"license_evidence": ""}, {"split": None}, {"speaker_id": None},
                        {"conversation_id": ""}, {"status": "oops"}, {"id": "not-a-clip"}):
            write_jsonl(reviews, [{**self.approved_review(record), **changes}])
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                build_export(self.root, reviews)

    def test_resume_and_changed_config(self):
        self.fetch()
        sampled, rendered, scenes = self.processing_mocks()
        process_sources(self.root, [self.source], Config())
        process_sources(self.root, [self.source], Config())
        self.assertEqual(sampled.call_count, 1)
        self.assertEqual(rendered.call_count, 1)
        self.assertEqual(scenes.call_count, 1)
        process_sources(self.root, [self.source], Config(side_margin=0.2))
        self.assertEqual(sampled.call_count, 2)
        self.assertEqual(len(candidate_records(self.root)), 2)

    def test_changed_output_is_not_reused(self):
        record = self.candidate()
        video = self.root / record["artifacts"]["video"]["path"]
        video.write_bytes(b"tampered")
        reviews = self.base / "reviews.jsonl"
        write_jsonl(reviews, [self.approved_review(record)])
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            build_export(self.root, reviews)

    def test_rejected_candidates_cannot_be_exported(self):
        record = self.candidate()
        record["state"] = "rejected"
        write_json(self.root / "clips" / record["id"] / "candidate.json", record)
        reviews = self.base / "reviews.jsonl"
        write_jsonl(reviews, [self.approved_review(record)])
        with self.assertRaisesRegex(ValueError, "Only successful"):
            build_export(self.root, reviews)

    def test_cross_split_leakage(self):
        record = self.candidate()
        duplicate = {**record, "id": record["id"] + "_other"}
        write_json(self.root / "clips" / duplicate["id"] / "candidate.json", duplicate)
        reviews = self.base / "reviews.jsonl"
        write_jsonl(reviews, [self.approved_review(record),
                             {**self.approved_review(duplicate), "split": "test"}])
        with self.assertRaisesRegex(ValueError, "Cross-split leakage"):
            build_export(self.root, reviews)

    def test_duplicate_revision_not_exported_twice(self):
        record = self.candidate()
        duplicate = {**record, "id": record["id"] + "_other"}
        write_json(self.root / "clips" / duplicate["id"] / "candidate.json", duplicate)
        reviews = self.base / "reviews.jsonl"
        write_jsonl(reviews, [self.approved_review(record), self.approved_review(duplicate)])
        with self.assertRaisesRegex(ValueError, "duplicate source interval"):
            build_export(self.root, reviews)

    def test_partially_overlapping_revision_not_exported_twice(self):
        record = self.candidate()
        duplicate = {**record, "id": record["id"] + "_overlap", "start": 5, "end": 15}
        write_json(self.root / "clips" / duplicate["id"] / "candidate.json", duplicate)
        reviews = self.base / "reviews.jsonl"
        write_jsonl(reviews, [self.approved_review(record), self.approved_review(duplicate)])
        with self.assertRaisesRegex(ValueError, "overlapping"):
            build_export(self.root, reviews)

    def test_output_never_overwrites_reviews(self):
        path = self.base / "reviews.jsonl"
        write_new(path, [{"status": "approved"}])
        with self.assertRaises(ValueError):
            write_new(path, [{"status": "pending"}])


if __name__ == "__main__":
    unittest.main()
