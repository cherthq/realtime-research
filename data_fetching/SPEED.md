# Throughput and long-recording operation

The quality settings and final all-decoded-frame checks are unchanged. These
changes remove repeated work; they do not sample away cutaways or skip difficult
frames. GPU results require their own quality comparison on the runner.

## What changed

- Candidate selection reads the compressed scan once, not once per clean span.
  Candidate buffers remain bounded by `max_seconds`; gaps/cuts/track changes
  still split candidates. Immutable crop geometry is cached rather than rebuilt
  for every frame.
- Late-clip rendering seeks the video input with keyframe preroll, then trims
  exact source timestamps. Audio uses a separate unseeked input to preserve AAC
  sample-clock phase and intentional offsets. This avoids repeated video-prefix
  decoding; audio-prefix decode/demux work remains. Context evidence also seeks
  its short clip instead of decoding its whole prefix.
- Source/output checks reuse grayscale and temporal-analysis images. Scene
  analysis resizes before channel conversion rather than copying full-size BGR.
  Per-frame evidence uses gzip level 1: faster writes, potentially larger files.
- OCR and VAD keep one isolated worker/model per backend for a processing call.
  Every request still checks input/model hashes, runs offline inference and
  validates response identity/coverage. VAD recurrent state resets per clip.
  Timeouts/crashes invalidate the child, never produce a pass. TalkNet/SyncNet
  remain one-shot pending real checkpoint validation.
- `--jobs N` runs distinct sources in spawn-based processes (default 1, maximum
  64). Each source has disjoint scan/render caches; only the parent aggregates
  the report. Completed parallel-source reports also persist individually.
- Scan metadata records decode, scene/bookkeeping, detection/tracking, metrics,
  serialization and recovery timings. Candidate metadata records render/check,
  worker and evidence timings. A requested GPU delegate participates in cache
  identity; CPU results cannot satisfy GPU output-verification caches.

## Measured locally, not a GPU throughput promise

Windows CPU, existing MediaPipe weights and previously downloaded footage:

| Operation | Before | After |
| --- | ---: | ---: |
| Selection from a 180-second interview scan | 2.54 s | 0.56 s |
| Selection from a fragmented 180-second, 60-FPS scan | 6.47 s | 0.43 s |
| Render a 5-second clip at minute 10 of an existing recording | 13.51 s | 2.12 s |
| Full real-model scan, 15-second interview excerpt | 49.73 s | 22.70–45.95 s |

All candidate windows/crops/parent IDs matched on both saved scans. All 449
decoded-frame evidence records matched exactly on the real-model excerpt.
There was one baseline scan and two optimized scans. Large timing variation,
including inside unchanged native pose inference, means these runs do not isolate
the code-only end-to-end speedup. They are not a prediction for every recording.
Pose inference remained dominant. The selection comparison uses the same saved
scan data, not different learned detections.

The late-render comparison produced identical decoded video frames and identical
48-kHz mono WAV files. Regression tests cover nonzero origins, intentional audio
delay, variable timestamps and AAC in a millisecond-timebase container. Seeking
both streams was rejected during testing because it shifted the AAC sample clock.

Real workers on the same ~7-second clip, with two uncached requests per mode:

| Worker | One-shot requests | Persistent cold / warm requests |
| --- | ---: | ---: |
| VAD | 0.72 / 0.62 s | 0.61 / 0.12 s |
| OCR | 10.24 / 8.03 s | 7.67 / 3.10 s |

Worker check objects matched exactly across both requests and both modes.
These are worker/stage timings, not a claim of 5–15x end-to-end speedup. Real
GPU throughput and multi-source resource scaling have not been measured here.

## Runner configuration

Use the Python 3.11 environments, verified assets and storage setup in
[README.md](README.md) and [worker guide](docs/workers.md). The main detector
defaults to CPU for compatibility.
For supported Linux GPU runtimes, explicitly request the MediaPipe GPU delegate:

```bash
python -m data_fetching --root /storage/facetime process \
  --manifest /storage/facetime/inputs/sources.jsonl \
  --profile quality-long --workers /storage/facetime/inputs/workers.json \
  --detector-device gpu --worker-mode persistent --jobs 1 \
  --limit 1 --scan-seconds 3600 --max-clips 1000
```

Run this on one authorized representative recording first, then inspect timings,
coverage, GPU utilization, memory and source/output quality. The installed
MediaPipe 0.10.21 Python delegate supports Linux/macOS, not Windows. Linux needs
a working GPU graphics/EGL/OpenGL environment, not merely a CUDA driver or a
large VRAM allocation. Headless/container support must be smoke-tested on the
actual host. No silent retry on CPU is performed by our adapter. The YuNet
backend is CPU-only here and cannot be combined with `--detector-device gpu`.
See the official [BaseOptions delegate documentation](https://developers.google.com/edge/api/mediapipe/python/mp/tasks/BaseOptions).

GPU settings for OCR/TalkNet/SyncNet remain in their separate worker registry.
Provision and validate the matching backend runtime explicitly; the supplied
Paddle wheel is CPU-only. The detector CLI flag does not switch these workers,
FFmpeg decoding or H.264 encoding to CUDA. VAD deliberately remains on CPU.

After a successful single-source run, compare `--jobs 1` and `--jobs 2` on
different sources before increasing further. Multiple processes create multiple
model copies; VRAM/RAM, disk bandwidth, CPU and native thread pools can become
the bottleneck. On smaller hosts use `--worker-mode oneshot` to release workers
after each clip, accepting slower startup. Do not run multiple CLI writers
against one dataset root; the parent keeps its dataset lock across parallel jobs.

`--scan-seconds 3600` covers at most one hour, not an arbitrarily long video.
Raise it deliberately for longer recordings and inspect `unscanned_tail_seconds`
and `candidate_limit_reached`. `--max-clips 1000` is an explicit budget, not a
guarantee that a source is fully exhausted. Raise `--limit` for larger manifests.
The core CLI is not a persistent collection scheduler. Downloading still
preserves full originals once for reuse;
parallel processing does not add aggressive YouTube download concurrency.

Re-running unchanged inputs validates/reuses caches. Code, model, config and
registry changes may require reprocessing and a new review. Preserve old outputs
and account for revisions when measuring unique footage.
