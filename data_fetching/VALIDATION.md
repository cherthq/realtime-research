# Local validation: 2026-10-22

This records a diagnostic replay, not quality certification or permission to train.
No recordings, model weights, private reviews or generated galleries are included.

## Checks performed

- Python 3.11, Windows: the initial 80 automated tests passed; throughput follow-up
  brings the suite to 96 passing tests, including single-pass hour-long selection,
  persistent workers, spawned jobs and timestamp/audio-preserving seeks (see
  [SPEED.md](SPEED.md)). Tests include real FFmpeg/PyAV
  media with mocked learned detections; they do not establish detector accuracy.
- Ruff `E9,F`, dependency checks in the main and OCR environments, CLI help/dry
  run and `git diff --check` passed. GitHub Actions workflow added; remote CI and
  Docker were not executed locally (Docker is unavailable on this machine).
- Real MediaPipe face/pose inference replayed the first 60 seconds of the already
  downloaded first TalkingHead-1KH small-list source, `--Y9imYnfBw`.
- Real Silero v5.1.2 ONNX VAD and PP-OCRv5_mobile_det text detection executed.
  Model manifests retain exact asset SHA-256 values and upstream provenance in
  ignored local storage. PaddleX 3.0.3/Paddle 3.0.0 execute in a separate environment.
- An actual OCR worker invocation succeeded with Python network access blocked
  and an installed local visualization font. Worker network blocking is not an
  OS sandbox. No implicit checkpoint downloads are allowed during inference.
- YuNet was smoke-tested on the 35-second frame, not benchmarked across a corpus.
  Both YuNet and MediaPipe detected the background graphic as an extra face; this
  supports fixing association rather than assuming detector replacement is enough.

## Same-source comparison

| Measurement | Earlier quality-long run | Reliability upgrade |
| --- | --- | --- |
| Source window | 0-60 seconds | 0-60 seconds |
| Decoded frames | 1,799 | 1,799 |
| Detected cuts | 6 | 6 |
| Review-needed candidates | 1 | 3 |
| Candidate duration | 5.572 seconds | 30.665 seconds |
| Preferred crop | Head/upper chest | Torso |
| Speech/text workers | Not configured | Executed |
| Training-approved duration | 0 | 0 |

The final intervals are 0.100-15.883, 34.167-42.676 and 53.120-59.493 seconds.
Every candidate respects the detected cut guards. This is not an independently
labelled measurement of missed cuts. The shot around 35 seconds is retained with
an explicit secondary-face uncertainty, not falsely declared clean.

The final replay's recorded code fingerprint matches the top-level pipeline Python
sources used for these outputs. All candidate artifact hashes were validated and
before/after decoded output frames were visually inspected. All three outputs are
768x576 and remain `needs_review`. OCR flags suspected text; it does not prove
that text is a digital overlay. Brief recovered tracking proposals remain flagged.
The first interval contains 24 recovered-proposal frames, not observed geometry.

Raw candidate-duration improvement is NOT accepted-footage yield. Source/quality
approvals were not granted. The remaining ~281 seconds of the source were not
inspected. The per-shot results do not establish hour-scale drift robustness.

## Remaining release gates

- Commercially appropriate exact active-speaker and sync checkpoints, real
  inference validation, proxy-preprocessing calibration and offset-sign tests.
- Human review of retained and rejected footage, including actual audio playback,
  graphic artifacts, framing, gaze and source rights.
- A disjoint labelled calibration/test corpus and measured accepted precision,
  rejected usable seconds, identity switches, missed transitions and review cost.
- Broader 2-5 hour source pilot, then a staged scale/throughput evaluation before
  targeting hundreds of approved hours. No speedup is inferred from one clip.

## Random-source diagnostic pilot and speed follow-up

The full training-ID list contained 2,872 unique sources. A seed-20261022 random
permutation selected an initial 20 with frozen reserve order. Fifteen sources
were acquired/processed, with up to the first 180 seconds scanned (42m 45.84s total).
Six acquisition attempts failed; the final attempt logged HTTP 429 and a private
video, triggering a safety stop. This was not a completed 20-source evaluation.

The result was 33 unreviewed candidates totalling 449.828 seconds, from 5.005 to
37.920 seconds. None reached one minute; zero training hours were approved.
The source list and opening-window sampling introduce selection biases. This is
not an estimate of full-recording yield, missed cuts or accepted-footage precision.
No generated video, gallery, raw log or local model registry is published.

The original run took about 188 minutes: 115 minutes of source scanning, 36 of
acquisition, 33 of post-scan processing and the remainder of orchestration/previews.
It downloaded ~7.8 hours of full originals (~4.7 GiB), despite bounded scanning.
The local experiment driver remains ignored and machine-specific, not a released
collection scheduler. See [SPEED.md](SPEED.md) for measured stage improvements,
unchanged-frame/window comparisons, and GPU/multi-source validation limits.

## Reproduce locally

Use a new ignored dataset root. Prepare an input JSONL pointing to the preserved
local original, run `fetch` to record/hash it, and explicitly provision detectors
and reviewed local workers as described in [RUNNER.md](RUNNER.md).

```bash
python -m unittest discover -s data_fetching/tests -v
ruff check data_fetching --select E9,F
python -m pip check
python -m data_fetching --root /storage/replay process --manifest /storage/replay/inputs/source.jsonl --profile quality-long --config data_fetching/examples/quality-long.json --workers /storage/replay/inputs/workers.json --limit 1 --scan-seconds 60 --max-clips 20
python -m data_fetching --root /storage/replay gallery --output /storage/replay/results/round1
```

Use the `benchmark` command only with reviewed source-hash-bound labels. See
[RELIABILITY.md](RELIABILITY.md) for the schema and interpretation of its metrics.
