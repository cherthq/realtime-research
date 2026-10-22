"""Speed-path contracts: unchanged coverage, bounded sessions, isolated source jobs."""

import gzip
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from data_fetching.__main__ import main
from data_fetching.core import Source, fingerprint, read_json, write_jsonl
from data_fetching.quality_config import QualityConfig
from data_fetching.quality_pipeline import process_quality
from data_fetching.quality_scan import clean_windows, image_metrics, TemporalMetrics
from data_fetching.quality_sessions import WorkerSession, active_session, worker_session
from data_fetching.quality_worker import cached_model
from data_fetching.vision import Detector


class WindowTests(unittest.TestCase):
    def test_one_gzip_read_for_many_spans_and_identical_parent_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'frames.jsonl.gz'
            scan = {'start': 0, 'end': 8, 'blocked': [[1, 2], [4, 5]], 'width': 1024, 'height': 768}
            rows = [{'time': i / 2, 'end': (i + 1) / 2, 'track_id': '0',
                     'region': [100, 100, 850, 650], 'flags': []} for i in range(16)]
            for row in rows:
                if 1 <= row['time'] < 2 or 4 <= row['time'] < 5:
                    row.update(flags=['blocked'], region=None)
            with gzip.open(path, 'wt', encoding='utf-8') as handle:
                handle.write(''.join(json.dumps(r) + '\n' for r in rows))
            from data_fetching.quality_scan import read_frames
            with patch('data_fetching.quality_scan.read_frames', wraps=read_frames) as reader:
                found = list(clean_windows(path, scan, QualityConfig(min_seconds=1)))
            reader.assert_called_once_with(path)
            self.assertEqual([(r[0]['time'], r[-1]['end']) for r, _, _ in found], [(0, 1), (2, 4), (5, 8)])
            self.assertEqual([p for _, _, p in found],
                             [fingerprint([path.parent.name, a, b])[:24] for a, b in ((0, 1), (2, 4), (5, 8))])

    def test_all_blocked_does_not_read_and_boundaries_are_not_bridged(self):
        scan = {'start': 0, 'end': 8, 'blocked': [[0, 8]], 'width': 1024, 'height': 768}
        with patch('data_fetching.quality_scan.read_frames') as reader:
            self.assertEqual(list(clean_windows(Path('absent.gz'), scan, QualityConfig())), [])
        reader.assert_not_called()

    def test_max_duration_and_track_splits_remain(self):
        rows = [{'time': i, 'end': i + 1, 'track_id': str(i // 5),
                 'region': [100, 100, 850, 650], 'flags': []} for i in range(10)]
        scan = {'start': 0, 'end': 10, 'blocked': [], 'width': 1024, 'height': 768}
        with patch('data_fetching.quality_scan.read_frames', return_value=iter(rows)):
            found = list(clean_windows(Path('scan/frames.gz'), scan,
                                      QualityConfig(min_seconds=1, long_seconds=2, max_seconds=3)))
        self.assertEqual([(r[0]['time'], r[-1]['end']) for r, _, _ in found], [(0, 3), (3, 5), (5, 8), (8, 10)])

    def test_hour_long_fragmented_scan_is_a_single_stream(self):
        rows = [{'time': i, 'end': i + 1, 'track_id': '0',
                 'region': [100, 100, 850, 650], 'flags': []} for i in range(3600)]
        scan = {'start': 0, 'end': 3600, 'blocked': [[i + 8, i + 10] for i in range(0, 3600, 10)],
                'width': 1024, 'height': 768}
        with patch('data_fetching.quality_scan.read_frames', return_value=iter(rows)) as reader:
            found = list(clean_windows(Path('hour/frames.gz'), scan, QualityConfig()))
        reader.assert_called_once()
        self.assertEqual(len(found), 360)
        self.assertTrue(all(r[-1]['end'] - r[0]['time'] == 8 for r, _, _ in found))

    @unittest.skipUnless(importlib.util.find_spec('cv2'), 'OpenCV required')
    def test_shared_metrics_are_exactly_equal(self):
        import cv2
        import numpy as np
        generator = np.random.default_rng(42)
        frames = [generator.integers(0, 256, (64, 96, 3), dtype=np.uint8) for _ in range(3)]
        raw, shared = TemporalMetrics(QualityConfig()), TemporalMetrics(QualityConfig())
        for i, rgb in enumerate(frames + [frames[-1]]):
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            small = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
            self.assertEqual(image_metrics(rgb, [5, 5, 48, 50]),
                             image_metrics(rgb, [5, 5, 48, 50], gray=gray, small=small))
            self.assertEqual(raw.update(rgb, i, i + 1), shared.update(rgb, i, i + 1, small=small))


class SessionTests(unittest.TestCase):
    def test_model_cache_reuses_only_same_configuration(self):
        from data_fetching import quality_worker
        create = Mock(side_effect=object)
        with patch.object(quality_worker, '_MODEL_CACHE', {}):
            a = cached_model('vad', {'weights': 'a'}, create)
            self.assertIs(a, cached_model('vad', {'weights': 'a'}, create))
            self.assertIsNot(a, cached_model('vad', {'weights': 'b'}, create))
            self.assertEqual(create.call_count, 2)
        with patch.object(quality_worker, '_MODEL_CACHE', None):
            self.assertIsNot(cached_model('vad', {}, create), cached_model('vad', {}, create))

    def test_scope_and_nested_scope_close_once(self):
        with patch('data_fetching.quality_sessions.WorkerSession') as session:
            self.assertIsNone(active_session())
            with worker_session():
                self.assertIs(active_session(), session.return_value)
                with worker_session():
                    self.assertIs(active_session(), session.return_value)
            session.assert_called_once()
            session.return_value.close.assert_called_once()
            self.assertIsNone(active_session())

    def test_timeout_discards_child_and_preserves_log(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            process = Mock(pid=123, stdin=io.BytesIO())
            process.poll.return_value = None
            session = WorkerSession()
            with patch('data_fetching.quality_sessions.subprocess.Popen', return_value=process), \
                    patch('data_fetching.quality_sessions.stop_process') as stop, \
                    patch('data_fetching.quality_sessions.time.monotonic', side_effect=[0, 2]):
                with self.assertRaises(subprocess.TimeoutExpired):
                    session.invoke('vad', {'fingerprint': 'test', 'python': sys.executable, 'timeout_seconds': 1},
                                   path / 'request.json', path / 'response.json', path / 'vad.log')
                stop.assert_called()
            # Simulate the owned process having exited after termination.
            process.poll.return_value = 1
            session.close()
            self.assertTrue((path / 'vad.log').exists())
            self.assertTrue(process.stdin.closed)

    def test_real_persistent_protocol_reuses_pid_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            session = WorkerSession()
            entry = {'python': sys.executable, 'fingerprint': 'fixture', 'timeout_seconds': 20}
            try:
                pids = []
                for i in range(2):
                    request, response = directory / f'{i}-request.json', directory / f'{i}-response.json'
                    from data_fetching.core import write_json
                    write_json(request, {'protocol_version': 0, 'job_id': str(i), 'backend': 'vad'})
                    session.invoke('vad', entry, request, response, directory / f'{i}.log')
                    pids.append(session.children[('vad', 'fixture')]['process'].pid)
                    result = read_json(response)
                    self.assertEqual(result['job_id'], str(i))
                    self.assertEqual(result['checks'], {})
                    self.assertIn('Unsupported worker protocol', result['error'])
                self.assertEqual(pids[0], pids[1])
            finally:
                session.close()
            self.assertEqual(session.children, {})


class RuntimeTests(unittest.TestCase):
    def test_gpu_delegate_explicit_and_preserved_on_reset(self):
        options = lambda **kwargs: SimpleNamespace(**kwargs)
        base = Mock(side_effect=options)
        base.Delegate.GPU = 'explicit-gpu'
        vision = SimpleNamespace(FaceDetector=Mock(), PoseLandmarker=Mock(),
                                 FaceDetectorOptions=options, PoseLandmarkerOptions=options,
                                 RunningMode=SimpleNamespace(IMAGE='image', VIDEO='video'))
        python = SimpleNamespace(BaseOptions=base, vision=vision)
        mp = SimpleNamespace(tasks=SimpleNamespace(python=python))
        with patch.dict(sys.modules, {'mediapipe': mp, 'mediapipe.tasks': mp.tasks,
                                     'mediapipe.tasks.python': python,
                                     'mediapipe.tasks.python.vision': vision}):
            detector = Detector({'face': Path('face.tflite'), 'pose': Path('pose.task')},
                                QualityConfig().geometry(), video=True, device='gpu')
            detector.reset()
            self.assertEqual(base.call_count, 4)
            self.assertTrue(all(c.kwargs['delegate'] == 'explicit-gpu' for c in base.call_args_list))
            detector.close()

    def test_gpu_and_worker_settings_dry_run_does_not_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / 'source.jsonl'
            write_jsonl(manifest, [{'id': 'fixture', 'path': 'absent.mp4'}])
            output = io.StringIO()
            with patch('sys.stdout', output), patch('subprocess.Popen') as launch:
                self.assertEqual(main(['--root', str(root / 'dataset'), 'process', '--manifest', str(manifest),
                                       '--profile', 'quality-long', '--jobs', '2', '--detector-device', 'gpu',
                                       '--worker-mode', 'oneshot', '--dry-run']), 0)
            launch.assert_not_called()
            plan = json.loads(output.getvalue())
            self.assertEqual(plan['config']['detector_device'], 'gpu')
            self.assertEqual(plan['jobs'], 2)
            self.assertFalse((root / 'dataset').exists())

    def test_spawn_jobs_preserve_error_reports_and_source_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources = [Source(id='b', path='absent-b.mp4'), Source(id='a', path='absent-a.mp4')]
            rows = process_quality(root, sources, QualityConfig(), {}, jobs=2)
            self.assertEqual([r['source_id'] for r in rows], ['b', 'a'])
            self.assertTrue(all(r['state'] == 'failed' for r in rows))
            self.assertEqual(read_json(root / 'reports/quality-process.json'), rows)
            self.assertTrue((root / 'reports/sources/a.json').exists())

    def test_jobs_and_devices_validate(self):
        for value in (True, 0, 65, 1.5):
            with self.assertRaises(ValueError):
                process_quality(Path('.'), [], QualityConfig(), {}, jobs=value)
        for value in ('cuda', None, True):
            with self.assertRaises(ValueError):
                QualityConfig(detector_device=value)
