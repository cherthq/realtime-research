import json
from pathlib import Path
import tempfile
import unittest

from data_fetching.queue import Queue, seconds
from data_fetching.long_scan import clean_runs
from data_fetching.hub_storage import publication_manifest
from data_fetching.dedup import matches
from unittest.mock import patch
from data_fetching.core import file_hash
from data_fetching.long_scan import scan_source
from types import SimpleNamespace
from data_fetching.hub_storage import publish


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.queue = Queue(self.directory / "discovery")
        self.addCleanup(lambda: self.queue.close())

    def source(self, suffix="aaaaaaaaaaa", duration=1000, digest=None):
        identity = self.queue.add({"url": "https://youtu.be/" + suffix, "duration": duration,
                                   "uploader": "creator", "license": "CC BY 4.0"})
        path = self.directory / (suffix + ".mp4")
        path.write_bytes((digest or suffix).encode())
        self.queue.acquired(identity, path, file_hash(path), {"duration": duration})
        return identity

    def candidate(self, source, start=0, end=600, revision="r1"):
        self.queue.complete_window(source, revision, 0, 1000,
            [{"start": start, "end": end, "framing": "head_upper_chest", "method": "sampled"}])
        return self.queue.db.execute("SELECT id FROM intervals WHERE source_id=? AND revision=?", (source, revision)).fetchone()[0]

    def allow(self, source):
        self.queue.source_review(source, reviewer="reviewer", evidence="permission agreement example", permission="approved", unique=True)

    def approve(self, interval, split="train"):
        self.queue.review({"id": interval, "quality": "approved", "reviewer": "reviewer",
          "notes": "Complete playback inspected", "inspection": "full_video_audio",
          "speaker_id": "speaker", "conversation_id": "recording", "split": split})

    def test_pending_is_never_approved_or_unique(self):
        source = self.source()
        self.candidate(source)
        report = self.queue.report()
        self.assertEqual(report["approved_usable_hours_all_lengths"], 0)
        self.assertEqual(report["unique_raw_acquired_hours_confirmed"], 0)
        self.assertGreater(report["raw_acquired_hours_provisional"], 0)

    def test_overlap_across_revisions_uses_interval_union(self):
        source = self.source()
        self.allow(source)
        self.approve(self.candidate(source, 0, 600))
        self.approve(self.candidate(source, 300, 900, "r2"))
        report = self.queue.report()
        self.assertEqual(report["approved_usable_hours_all_lengths"], 900 / 3600)
        self.assertEqual(report["approved_continuous_hours_gt_300s"], 900 / 3600)

    def test_short_and_exact_thresholds_do_not_inflate_target(self):
        source = self.source()
        self.allow(source)
        self.approve(self.candidate(source, 0, 30))
        self.approve(self.candidate(source, 50, 110, "r2"))
        report = self.queue.report()
        self.assertEqual(report["approved_short_hours_lt_60s"], 30 / 3600)
        self.assertEqual(report["approved_preferred_hours_at_least_60s"], 60 / 3600)
        self.assertEqual(report["approved_continuous_hours_gt_60s"], 0)

    def test_reupload_offset_deduplicates_partial_overlap(self):
        first = self.source(duration=1000)
        second = self.source("bbbbbbbbbbb", duration=1000)
        self.allow(first)
        self.queue.source_review(second, reviewer="reviewer", evidence="manually matched recording",
                                  permission="approved", canonical=first, offset=300)
        self.approve(self.candidate(first, 300, 900))
        self.approve(self.candidate(second, 0, 600))
        report = self.queue.report()
        self.assertEqual(report["approved_usable_hours_all_lengths"], 600 / 3600)
        self.assertEqual(report["unique_raw_acquired_hours_confirmed"], 1300 / 3600)

    def test_exact_hash_alias_is_grouped_on_acquisition(self):
        first = self.source(digest="same")
        self.allow(first)
        second = self.source("bbbbbbbbbbb", digest="same")
        self.assertEqual(self.queue.source(second)["canonical"], first)
        self.assertEqual(self.queue.report()["unique_raw_acquired_hours_confirmed"], 1000 / 3600)

    def test_reopen_and_replay_does_not_duplicate_work_or_reset_review(self):
        source = self.source()
        self.allow(source)
        interval = self.candidate(source)
        self.approve(interval)
        self.queue.close()
        self.queue = Queue(self.directory / "discovery")
        self.candidate(source)
        self.assertEqual(self.queue.db.execute("SELECT COUNT(*) FROM intervals").fetchone()[0], 1)
        self.assertEqual(self.queue.db.execute("SELECT quality FROM intervals").fetchone()[0], "approved")
        self.queue.add({"url": "https://youtube.com/watch?v=aaaaaaaaaaa", "title": "updated title"})
        self.assertEqual(self.queue.source(source)["permission_status"], "approved")

    def test_inspection_permissions_and_split_gates(self):
        source = self.source()
        interval = self.candidate(source)
        with self.assertRaisesRegex(ValueError, "permission"):
            self.approve(interval)
        self.allow(source)
        self.approve(interval)
        other = self.candidate(source, 600, 900, "r2")
        with self.assertRaisesRegex(ValueError, "Cross-split"):
            self.approve(other, "test")
        with self.assertRaisesRegex(ValueError, "Full interval"):
            self.queue.review({"id": other, "quality": "approved", "reviewer": "r", "notes": "preview only", "inspection": "sampled_frames"})

    def test_incomplete_window_rolls_back_atomically(self):
        source = self.source()
        with self.assertRaises(ValueError):
            self.queue.complete_window(source, "r1", 0, 100, [
              {"start": 0, "end": 50, "framing": "head_upper_chest", "method": "sampled"},
              {"start": 90, "end": 150, "framing": "head_upper_chest", "method": "sampled"}])
        self.assertEqual(self.queue.db.execute("SELECT COUNT(*) FROM intervals").fetchone()[0], 0)
        self.assertEqual(self.queue.db.execute("SELECT COUNT(*) FROM windows").fetchone()[0], 0)

    def test_unverified_media_cannot_be_published(self):
        self.source()
        self.assertEqual(publication_manifest(self.queue), ([], []))

    def test_continuous_run_survives_computational_window_boundary(self):
        samples = [{"time": i / 2, "profiles": {"head_upper_chest": True}} for i in range(1500)]
        runs = clean_runs(samples, 0, 750, 2, "head_upper_chest")
        self.assertEqual(len(runs), 1)
        self.assertGreater(runs[0]["end"] - runs[0]["start"], 300)
        samples[600]["profiles"]["head_upper_chest"] = False
        runs = clean_runs(samples, 0, 750, 2, "head_upper_chest")
        self.assertEqual(len(runs), 2)
        self.assertLess(runs[0]["end"], 300)
        self.assertGreater(runs[1]["start"], 300)

    def test_union_does_not_fill_gaps(self):
        self.assertEqual(seconds([(0, 100), (50, 150), (200, 300)]), 250)

    def test_perceptual_reencode_proposes_offset_without_merging(self):
        first = [{"time": i * 5, "full": str(2 ** (i + 10) - 1), "center": str(2 ** (i + 10) - 1)} for i in range(20)]
        second = [{**r, "time": r["time"] - 30} for r in first if r["time"] >= 30]
        result = matches(first, second)
        self.assertEqual(result["suggested_offset_seconds"], 30)
        self.assertEqual(result["status"], "needs_manual_alignment_review")

    def test_scan_resumes_remaining_windows_and_retains_long_span(self):
        source = self.source()
        original = self.directory / "synthetic.mp4"
        original.write_bytes(b"not real media; decoder is mocked for checkpoint test")
        self.queue.db.execute("UPDATE sources SET original=?,sha256=?,media=? WHERE id=?",
          (str(original),file_hash(original),json.dumps({"duration":500}),source))
        self.queue.db.commit()
        called = []
        def sample(path,start,end,info,detector,config):
            called.append((start,end))
            return [{"time": start+i/2,"profiles":{"head_upper_chest":True,"wider_torso_gestures":False}}
                    for i in range(int((end-start)*2))]
        with patch('data_fetching.long_scan.model_paths',return_value=({},{})), \
             patch('data_fetching.long_scan.runtime_versions',return_value={}), \
             patch('data_fetching.long_scan.scene_intervals',return_value=[(0,500)]), \
             patch('data_fetching.long_scan.Detector'), \
             patch('data_fetching.long_scan.sample_window',side_effect=sample):
            with patch('data_fetching.long_scan.time.monotonic',side_effect=[0,10,20]):
                first=scan_source(self.queue,self.queue.source(source),deadline=15)
            self.assertEqual(first['state'],'checkpointed')
            self.assertEqual(called,[(0,120),(120,240)])
            with patch('data_fetching.long_scan.time.monotonic',return_value=0):
                result=scan_source(self.queue,self.queue.source(source),deadline=15)
            self.assertEqual(result['state'],'screened')
            self.assertEqual(called,[(0,120),(120,240),(240,360),(360,480),(480,500)])
        interval=self.queue.db.execute('SELECT * FROM intervals').fetchone()
        self.assertGreater(interval['end']-interval['start'],300)
        self.assertEqual(interval['quality'],'pending')

    def test_upload_verifies_hash_and_preserves_existing_card(self):
        source=self.source()
        self.queue.source_review(source,reviewer='reviewer',evidence='public source license evidence',redistribution='approved')
        original=Path(self.queue.source(source)['original'])
        with patch('huggingface_hub.get_token',return_value='mock-local-auth'), \
             patch('huggingface_hub.HfApi') as api, \
             patch('huggingface_hub.get_hf_file_metadata',return_value=SimpleNamespace(size=original.stat().st_size,etag=file_hash(original))):
            api.return_value.dataset_info.return_value=SimpleNamespace(id='canonical/dataset',sha='parent',siblings=[SimpleNamespace(rfilename='README.md')])
            api.return_value.create_commit.return_value=SimpleNamespace(oid='commit')
            result=publish(self.queue,'old/dataset')
            call=api.return_value.create_commit.call_args.kwargs
            self.assertEqual(call['repo_id'],'canonical/dataset')
            self.assertEqual(call['parent_commit'],'parent')
            self.assertNotIn('README.md',[o.path_in_repo for o in call['operations']])
            self.assertEqual(result['state'],'uploaded_and_verified')
            manifest=json.loads(next(o.path_or_fileobj.decode() for o in call['operations'] if o.path_in_repo=='discovery/sources.jsonl'))
            self.assertNotIn(str(self.directory),json.dumps(manifest))
            self.assertEqual(manifest['training_permission_status'],'pending')

    def test_tampered_original_cannot_be_approved(self):
        source=self.source()
        self.allow(source)
        interval=self.candidate(source)
        Path(self.queue.source(source)['original']).write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'Original'):
            self.approve(interval)


if __name__ == "__main__":
    unittest.main()
