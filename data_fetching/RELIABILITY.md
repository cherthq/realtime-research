# Reliability upgrade and validation

This is a review-first collection tool, not an artifact-free dataset guarantee.
The real TalkingHead replay is a diagnostic case, not a representative benchmark.
Generated videos, weights, worker registries, reports and HTML belong under
ignored `data/`; nothing here publishes recordings or approves training rights.

## Changes

- `quality_tracking.py` associates faces with shoulder/nose geometry and maintains
  shot-local temporal IoU continuity. It is not biometric recognition or a full
  ByteTrack implementation. Additional unsupported face detections stay explicit
  uncertainties; multiple supported person associations block the interval.
- Every frame retains raw boxes, confidence scores, poses, selected association,
  specific failures, track state and uncertainties. A background detection no
  longer erases the primary face from diagnostic metadata.
- A bounded, two-sided anchor envelope can propose a crop across a brief miss
  (default <=0.25s). No frames are generated/interpolated. Recovered geometry is
  not observed visibility and cannot silently pass continuity or framing. Cuts,
  hard defects, ambiguous person associations and long misses are not bridged.
- `framing_profile` is `torso` by default; alternatives are `head_chest` and
  `gesture` (extra space, not proof that all hands are visible). Use
  `examples/quality-long.json`. Crop selection actually uses the selected region,
  with safety space; not merely a torso label applied after a close face crop.
- Torso depth is a shoulder-width heuristic, not an anatomical measurement.
  A source-limited lower margin is permitted only above `torso_min_depth` and is
  explicitly review-flagged. Head/shoulder requirements cannot be clipped away;
  no missing body, padding, moving zoom, restoration or upscaling is generated.
- Post-render checks project inspected source regions using the actual crop and
  compare independent output head/shoulder geometry. They do not demand another
  torso expansion inside the already cropped image. Detector disagreements stay
  flagged. Output PTS mapping/decode faults remain hard errors.
- Cheap cut detection still sees every frame. Select `scene_detector: adaptive`
  to benchmark PySceneDetect's adaptive method; `content` remains the baseline.
  Static low luminance alone is no longer sufficient to call a shot a fade.
  TransNet V2 is a research comparison, not an installed/validated dependency.
- Near-repeat thumbnail differences are additional possible-freeze evidence,
  not proof of freezing; normal stillness still needs contextual review.
- VAD uses hysteresis, minimum speech/silence and padding, preserving probabilities.
  TalkNet retains frame scores and checks sustained low scores during speech,
  instead of allowing a whole-window median to conceal brief mismatches.
- OCR retains confidence, polygons and text-location categories. Text on clothing
  or in a background is not automatically a subtitle. Burst inspection supplements
  periodic samples; absence in samples does not certify an overlay-free clip.

The default values are starting policies, NOT calibrated thresholds. A secondary
face without visible shoulders can be real. A source-limited region can still be
too tight. No combination of these heuristics establishes eye contact, consent,
speaker identity, naturalness, or freedom from synthetic/beauty-filter artifacts.

## Optional face detector comparison

MediaPipe short-range remains supported. YuNet is an explicit alternative, with
the existing pose model. Use a new dataset root; existing detector manifests are
not overwritten:

```bash
python -m data_fetching --root /storage/yunet-test models --face-backend yunet --dry-run
python -m data_fetching --root /storage/yunet-test models --face-backend yunet
```

The pinned model filename is `face_detection_yunet_2023mar.onnx`, compatible with
our OpenCV 4.x runtime. Analysis is downscaled to at most 960 pixels on its long
edge, with detections mapped back to source pixels. Export pixels are unchanged.
The installer records URLs and hashes; the model directory's official MIT terms
are at <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet>.
This does not establish that YuNet wins on our domain: compare measured errors.

## Worker setup and licensing

Use `requirements-vad-worker.txt` for a minimal CPU Silero ONNX environment.
OCR uses the direct PaddleX text detector, not the full PaddleOCR document/LLM
extras. Keep its environment isolated with `requirements-ocr-worker.txt`.
Registries still require explicit local hashed checkpoints and terms evidence.
Python network connections are blocked during inference. A local visualization
font is selected before importing PaddleX, which otherwise attempts font downloads.
This guard is not a substitute for an OS/container network policy.

- Silero's MIT terms and Paddle's specific Apache-2.0 model card should be retained
  alongside the selected checkpoint revision.
- TalkNet repository code is MIT, but record the exact checkpoint terms too.
- Oxford's SyncNet model page combines research-purpose wording with a CC BY
  reference. Obtain clarity for the intended use; the adapter is not clearance.
- InsightFace public pretrained weights and DOVER/FAST-VQA must not be silently
  added as commercial defaults: their published terms restrict commercial use.

All active-speaker/sync score thresholds need checkpoint AND preprocessing-specific
calibration. Use known-good speech, offscreen/mismatched voices, low-confidence
faces and positive/negative injected delays. Missing assets/calibration stay
unverified. A successful import or synthetic tone is not learned-model validation.

## Labelled benchmark

Create JSONL labels from full interval review, not sparse thumbnails:

```json
{"source_id":"recording_01","source_sha256":"ACTUAL_ORIGINAL_SHA256","start":10,"end":25,"label":"usable","partition":"test","group_id":"reviewed-speaker-and-source-group","reviewer":"reviewer-id","evidence":["review-log-123"]}
```

Labels are `usable`, `unusable`, or `uncertain`. Partitions are `calibration` or
`test`; the same reviewed speaker/source group cannot appear in both. Intervals
cannot overlap. Use exact original hashes and a consistent quality rubric.

```bash
python -m data_fetching --root /storage/facetime benchmark --labels labels.jsonl --stage candidates
python -m data_fetching --root /storage/facetime benchmark --labels labels.jsonl --stage eligible
python -m data_fetching --root /storage/facetime benchmark --labels labels.jsonl --stage approved --reviews reviews.jsonl
python -m data_fetching --root /storage/facetime gallery --output /storage/facetime/results/round1
```

The benchmark reports duration-weighted retention precision/recall separately for
calibration and test. Candidate-stage precision is NOT approval precision. Missing
labels and unknown footage do not certify cleanliness; report coverage and sample
both rejected and retained footage. Overlapping processing revisions do not count
twice. The gallery shows the latest revision per interval and never approves clips.

Before scaling, label a varied small corpus (including long uninterrupted clips)
and compare ablations: baseline, association, recovery, torso crop, audio and OCR.
Report missed cuts, identity switches, subject containment, defect misses, rejected
usable seconds, review effort and compute/storage per approved hour. Keep final
test sources/speakers separate from tuning. A single replay cannot set these rates.

## Execution and remaining scale work

Source scans and renders remain content-hashed and resumable at stage boundaries.
One writer per dataset root is intentional; independent workers need separate roots
and an auditable merge, not concurrent writes through the same lock. A distributed
scheduler, persistent GPU worker pool and robust reupload indexing are not shipped.
Overlapping AV windows currently rerun proxy extraction; benchmark this overhead
before adding large-scale collection. Review model disagreement before merely
raising thresholds. Human approval and source/redistribution permission stay
separate from automated QC.
