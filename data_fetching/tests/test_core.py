from pathlib import Path
import tempfile
import unittest

from data_fetching.core import (Config, Source, assess_samples, fit_crop, intervals,
                                load_sources, within, write_jsonl, youtube_url)
from data_fetching.pipeline import import_talkinghead


def good_sample(shift=0):
    points = [{"x": 500, "y": 350, "visibility": 1, "presence": 1} for _ in range(33)]
    points[11] = {"x": 340 + shift, "y": 330, "visibility": 1, "presence": 1}
    points[12] = {"x": 660 + shift, "y": 330, "visibility": 1, "presence": 1}
    return {"time": 0, "faces": [{"box": [420 + shift, 100, 580 + shift, 280], "score": 0.9}],
            "poses": [points]}


class CoreTests(unittest.TestCase):
    def test_url_canonicalization(self):
        self.assertEqual(youtube_url("https://youtu.be/abcdefghijk?t=10"),
                         "https://www.youtube.com/watch?v=abcdefghijk")
        for url in ("http://youtube.com/watch?v=abcdefghijk", "https://evil.com/abcdefghijk",
                    "https://www.youtube.com/playlist?list=x", "https://u:p@youtube.com/watch?v=abcdefghijk"):
            with self.assertRaises(ValueError):
                youtube_url(url)

    def test_source_validation(self):
        for row in ({"id": "../evil", "path": "x.mp4"}, {"id": "a"},
                    {"id": "a", "path": "x", "start": -1},
                    {"id": "a", "path": "x", "start": 5, "end": 4},
                    {"id": "a", "path": "x", "end": float("nan")},
                    {"id": "a", "path": "x", "split": "small"},
                    {"id": "a", "path": "x", "start": True}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                Source.parse(row, Path.cwd())

    def test_source_nulls_and_wrong_types(self):
        for row in ({"id": "a", "path": 10}, {"id": "a", "url": []},
                    {"id": "a", "path": "x", "start": None},
                    {"id": "a", "path": "x", "license": None},
                    {"id": "a", "path": "x", "split": []}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                Source.parse(row, Path.cwd())

    def test_local_paths_and_duplicate_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / "sources.jsonl"
            row = {"id": "a", "path": "x.mp4"}
            write_jsonl(manifest, [row, row])
            loaded = load_sources(manifest)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].path, str(Path(temp) / "x.mp4"))
            write_jsonl(manifest, [row, {"id": "a", "path": "y.mp4"}])
            with self.assertRaises(ValueError):
                load_sources(manifest)
            write_jsonl(manifest, [row, {"id": "b", "path": "x.mp4"}])
            with self.assertRaises(ValueError):
                load_sources(manifest)

    def test_contained_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(within(Path(temp), "clips/a.mp4"), Path(temp) / "clips/a.mp4")
            with self.assertRaises(ValueError):
                within(Path(temp), "../escape")

    def test_config_validation(self):
        for kwargs in ({"width": 767}, {"height": -2}, {"sample_fps": 0},
                       {"valid_fraction": 1.1}, {"max_seconds": 2}, {"torso_depth": -1}):
            with self.assertRaises(ValueError):
                Config(**kwargs)

    def test_crop_exact_ratio_and_contains_union(self):
        config = Config()
        for box in ([10, 5, 700, 550], [500, 100, 1200, 680], [10, 200, 800, 710]):
            x, y, w, h = fit_crop(box, 1280, 720, config)
            self.assertEqual(w * 3, h * 4)
            self.assertTrue(all(v % 2 == 0 for v in (x, y, w, h)))
            self.assertTrue(x <= box[0] < box[2] <= x + w <= 1280)
            self.assertTrue(y <= box[1] < box[3] <= y + h <= 720)
            self.assertGreaterEqual(w, config.width)
            self.assertGreaterEqual(h, config.height)

    def test_crop_does_not_upscale_or_invent_area(self):
        for box, width, height in (([50, 20, 300, 300], 640, 480),
                                   ([-1, 20, 800, 500], 1280, 720),
                                   ([0, 0, 1280, 720], 1280, 720)):
            with self.assertRaises(ValueError):
                fit_crop(box, width, height, Config())

    def test_scene_boundaries(self):
        self.assertEqual(list(intervals([(0, 4), (4, 40)], 0, 40, Config())),
                         [(4, 19), (19, 34), (34, 40)])

    def test_valid_torso_and_motion(self):
        result = assess_samples([good_sample(-50), good_sample(50)], 1280, 720, Config())
        self.assertIsNone(result["rejection_reason"])
        self.assertIsNotNone(result["crop"])

    def test_missing_shoulders_multiple_faces_closeups(self):
        variants = []
        sample = good_sample()
        sample["poses"][0][11]["visibility"] = 0
        variants.append(sample)
        sample = good_sample()
        sample["faces"] *= 2
        variants.append(sample)
        sample = good_sample()
        sample["poses"] *= 2
        variants.append(sample)
        sample = good_sample()
        sample["faces"][0]["box"] = [200, 10, 900, 500]
        variants.append(sample)
        for sample in variants:
            self.assertIsNone(assess_samples([sample], 1280, 720, Config())["crop"])

    def test_threshold_and_no_samples(self):
        samples = [good_sample() for _ in range(9)] + [{"faces": [], "poses": []}]
        self.assertIsNotNone(assess_samples(samples, 1280, 720, Config())["crop"])
        samples.append({"faces": [], "poses": []})
        self.assertIsNone(assess_samples(samples, 1280, 720, Config())["crop"])
        self.assertIsNone(assess_samples([], 1280, 720, Config())["crop"])

    def test_upstream_id_import_only(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ids.txt"
            path.write_text("abcdefghijk\nabcdefghijk\nlmnopqrstuv\n", encoding="utf-8")
            rows = import_talkinghead(path, "train", 1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["license"], "unknown")
            path.write_text("abcdefghijk_0000,720,1280,0,100,1,2,3,4\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                import_talkinghead(path, "train", 5)


if __name__ == "__main__":
    unittest.main()
