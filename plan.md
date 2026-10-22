# FaceTime dataset quality and metadata upgrade plan

Status: code and runner handoff implemented; real OCR/VAD smoke tests and a partial
random-source pilot completed locally. TalkNet/SyncNet checkpoint validation,
labelled calibration and accepted-footage evaluation remain pending on the runner.
Updated: 2026-10-22. See `data_fetching/RUNNER.md` for setup and limitations and
`data_fetching/SPEED.md` for the subsequent throughput changes.

Reliability follow-up: face/body association with retained raw detections, explicit
short-gap proposals, actual torso crop profiles, source-projected output checks,
specific reason codes, adaptive-cut comparison, VAD hysteresis, time-resolved ASD
findings, contextual OCR and a labelled interval benchmark are implemented. See
`data_fetching/RELIABILITY.md`. One real-source replay is not corpus calibration;
commercial checkpoint clearance and the broader labelled pilot remain gates.

This is an implementation record and the remaining validation plan for
`data_fetching/`, not quality certification. The earlier foundation remains in
`docs/data-fetching-plan.md`. This document supersedes that plan's exclusions of
active-speaker checks and its short-clip-only selection strategy for this upgrade.

## 1. Goal and boundaries

Produce reviewable, long, uninterrupted image-and-audio avatar training candidates
with stable 4:3 framing. Remove bad portions rather than discarding every recording
with a cut or defect. Keep full originals for future model-specific processing.

- Keep 768 x 576 landscape output, separate WAV audio and reference images.
- Prefer 60-300 second clean segments. Keep clean 5-59 second remnants in a
  separate short bucket, excluded from the main long-segment export by default.
  Partition longer clean spans into non-overlapping windows of at most 300 seconds;
  retain their parent continuous-span ID so longer contexts can be reconstructed
  from the original, not by concatenating encoded clips.
- Keep the existing 5-15 second smoke-test profile. Introduce an explicit
  `quality-long` profile rather than silently changing old configs or caches.
- Scope: local processing, quality checks, review, metadata and local export.
  No model training, mass collection, scheduler deployment or Hub uploads here.
- Do not inpaint watermarks, generate missing torso, sharpen/restore faces,
  replace backgrounds, stretch footage or silently repair synchronization.
- Automated results are evidence, not proof of identity, gaze, naturalness,
  permission or artifact-free footage. Human training approval remains required.

Original gaps before the quality upgrade: one cut detector; independent face/pose samples at 2 FPS with a 90%
pass rule; fixed crops validated only against those samples; no artifact,
active-speaker or perceptual lip-sync checks. Discovery on a separate scheduler
is not assumed to use this implementation.

## 2. Processing changes

```text
Source manifest + preserved original
  -> timestamp-aware sequential decode and coverage record
  -> cuts / fades / invalid intervals
  -> single-subject tracks and framing checks
  -> artifact evidence and speech / speaker / sync checks
  -> split into clean spans, fit fixed 4:3 crops, render and revalidate
  -> versioned candidates + evidence + review queue
  -> approved-only, deduplicated training and optional Hub-ready export
```

### A. Cuts, fades and temporal continuity

- Use a sequential decoder exposing actual presentation timestamps (PyAV) for
  frame analysis. Preserve the source's first-video-frame-relative clock; store
  original stream offsets and time bases. Do not infer VFR timing from frame/FPS.
  Bound memory with rolling buffers and stream frame metrics to sidecar files.
- Retain PySceneDetect ContentDetector and add ThresholdDetector for fades.
  Keep brief-shot detection enabled; evaluate AdaptiveDetector on the labelled
  pilot before replacing the content baseline. Record detector type and scores.
  If the scene backend's frame indices cannot be reconciled with decoded PTS,
  mark boundary timing unresolved and block automatic eligibility.
- Analyze every decoded frame in the requested scan interval. Record precisely
  what was scanned; a limited scan must never label the entire recording clean.
- Exclude transition intervals and a configurable 0.25 seconds on either side
  of detected cuts/fades. For fades, exclude the full detected transition, not
  only its midpoint. Clamp trims to the source bounds.
- Split at invalid spans, subject-track breaks and confirmed defects. Never
  bridge a bad interval or concatenate across cuts to meet minimum duration.
- Decode errors and unexplained timestamp gaps become explicit unresolved
  intervals. Fail the affected span closed rather than skipping frames silently.

### B. Continuous subject checks and stable 4:3 framing

- Reuse the existing face detector and pose model. Run face detection on every
  decoded frame; use pose VIDEO mode with increasing timestamps, resetting at
  shot boundaries. Associate face/pose geometry over time; keep track IDs local
  to each source/shot. This is tracking, not biometric identity recognition.
- Record missing/extra faces, shoulder confidence, ambiguous associations,
  abrupt position/scale changes and track resets with timestamps. Remove the
  global 90% rule as sufficient evidence for continuous quality.
- Split at confirmed discontinuities. Brief detector uncertainty is a review
  flag, not interpolated proof of visibility. Automatic framing eligibility
  requires valid head/shoulder containment on every decoded output frame;
  missing evidence stays unresolved until explicitly reviewed.
- Support `head_upper_chest` and `wider_torso_gestures` framing labels. Compute
  each label independently from face/shoulder coverage and configured margins;
  a close webcam view must not be rejected merely for lacking waist/hips.
  Wider torso is a separate qualification, not inferred from a face rectangle.
- Fit one fixed, even-aligned 4:3 crop over each clean span's required regions,
  including explicit headroom, both shoulders and the selected chest margin.
  Face boxes do not outline hair; flag near-edge headroom for human inspection.
- No moving zoom/crop in v1. If subject motion prevents a safe fixed crop,
  subdivide and reassess; retain only eligible pieces. No padding or upscaling.
- Validate the final crop against all frame-level regions and recheck rendered
  output. Record crop coordinates, minimum edge clearance, containment failures,
  framing label and coverage. A tracking success is not a gaze/identity label.

### C. Artifact and image-quality screening

- On every frame, measure face-region sharpness, exposure/clipped-pixel ratios,
  black frames, repeated-frame runs, timestamp anomalies and decode failures.
  Use face/foreground metrics so natural background blur is not a rejection.
- Treat freeze detection as a temporal signal: distinguish natural stillness
  from duplicated imagery, especially when audio/mouth activity continues.
  A single repeated frame or low bitrate alone must not cause hard rejection.
- Add a PaddleOCR text-detection adapter, initially sampled at 2 FPS plus shot
  edges and suspicious changes. Save polygons, confidence, timestamps, sampled
  coverage, crop intersection and representative evidence. OCR results are
  suspected text, not a guaranteed watermark/subtitle classification.
- Flag subtitle/lower-third patterns and persistent screen-anchored text for
  review. Text printed on clothing, a book or a background sign is not
  automatically an inserted graphic. Inspect the final crop as well as source:
  graphics fully outside the crop need not invalidate that crop.
- Flag split screens, picture-in-picture and slide layouts using multiple-person
  and persistent-panel/layout heuristics. They are proposals for review; one
  detected face does not establish that the layout is a clean full-frame shot.
- Keep non-text logos, beauty filters, synthetic content and virtual-background
  defects as explicit manual-review checks. OCR and simple quality metrics cannot
  certify their absence. Do not add an unvalidated universal artifact classifier.
- Confirmed contamination is excluded and clean subspans reconsidered. Ambiguous
  content is quarantined for review, not silently accepted or destructively fixed.
- Save raw metrics separately from policy thresholds. Calibrate thresholds on
  labelled footage before treating heuristic scores as hard rejection gates.

### D. Audio, visible speech and lip-sync

- Preserve existing common-clock trimming and duration checks. Add whole-interval
  audio decode, peak/RMS, near-full-scale sample fraction, silence/dropout and
  channel statistics. A 0 dB peak alone is only a possible-clipping flag.
- Use Silero VAD to locate speech, not to infer who speaks. Preserve natural
  pauses within valid continuous footage; label speech ratio and speech intervals.
  Keep non-speaking/listening footage in a separate bucket, not a talking-teacher
  export by default. No transcription or translation is required in this stage.
- Add TalkNet active-speaker scoring on the tracked visible face and SyncNet
  offset/confidence checks. Assess overlapping 3-second windows at 1-second
  strides across speech-bearing spans, including an end-aligned final window.
  Short/unsupported windows remain unresolved; never extrapolate one good score
  across an entire long clip. Adapt window constraints to the pinned backend.
- Produce any required 25 FPS face/audio analysis proxies on one common clock.
  Preserve original output timing; store proxy-to-source mappings and offset
  sign/units. Validate sign with deliberately delayed audio in tests.
- Record window-level active-speaker scores, sync offsets in milliseconds,
  confidence and analyzed speech coverage. Stable high-confidence mismatch
  identifies a bad span; low-confidence results go to review. Initial sync flag:
  absolute estimated lag above 80 ms, subject to held-out calibration.
- Do not automatically shift audio, dub speech or stretch time. Re-score the
  actual rendered outputs so encoding/proxy mistakes cannot inherit a source pass.
- Treat TalkNet/SyncNet as proposed backends, gated on exact checkpoint provenance,
  code/weight terms and a reproducible smoke test. Missing, unsuitable or failed
  backends yield `unverified`/`error`, never a synthetic pass. A human can resolve
  a check with documented evidence; a skipped model alone cannot clear it.

## 3. Metadata, review and export contract

### Versioned records and evidence

- Introduce `schema_version: 2` with separate source, interval, processing and
  review records. Validate finite times, required fields, enum values, bounds,
  referential integrity and dataset-relative artifact paths.
- Source records: canonical URL/provider ID, uploader/title, acquisition time,
  original path/hash, media properties, attribution, license/evidence and separate
  source-permission/redistribution approval. Never infer these approvals from QC.
- Interval records: source ID/hash, half-open `[start, end)` bounds on the source
  video clock, frame/PTS bounds, parent continuous-span ID, local track ID,
  framing/duration buckets, crop and artifact references. Human-assigned speaker,
  conversation and split remain null until reviewed; track ID is not speaker ID.
- Separate stable logical `interval_id` (source hash + frame-aligned bounds) from
  `processing_id` (code/config/runtime/model hashes). A new processing revision
  must not create new footage in hour counts or inherit old crop/QC approvals.
- Every check carries `status`, method/backend/version, actual inspected ranges,
  coverage, thresholds, raw-metric sidecar, findings and evidence references.
  Check states: `pass`, `flag`, `fail`, `unverified`, `error`, `not_applicable`.
  Use null for unavailable measurements, not zero or an empty-string success.
  `not_applicable` requires a reason and cannot bypass required speech checks
  for a talking-teacher sample.
- Findings contain a stable reason code, start/end, severity, confidence when
  meaningful, evidence paths and origin (`model`, `heuristic`, `human`). Record
  OCR as sampled coverage even when decode/tracking covered every frame.
- Keep full per-frame/window records in compressed JSONL sidecars; summaries
  retain worst intervals and coverage, not averages that hide brief failures.
  Store hashes, source PTS references and processing provenance for sidecars.

Example check shape (illustrative values, not an existing inspection result):

```json
{
  "status": "unverified",
  "method": "syncnet",
  "backend_revision": null,
  "analyzed_ranges": [],
  "speech_coverage_fraction": 0.0,
  "offset_ms": null,
  "reason_codes": ["backend_not_installed"],
  "evidence": []
}
```

### Review decisions and deduplication

- Keep processing state (`candidate`, `rejected`, `failed`) separate from
  automatic QC disposition (`eligible`, `needs_review`, `rejected`), human
  quality approval and source-permission approval. `eligible` is not training
  approved. Unresolved required checks force `needs_review`.
- Review records reference exact interval + processing/artifact hashes, reviewer,
  UTC time, per-check decisions and evidence. Preserve model findings when a
  human overrides a flag. Overrides require a reason; unrenderable/corrupt output
  cannot be overridden. Old approvals remain audit history, not current approval.
- Deduplicate exact source aliases/revisions by original hash and interval.
  Detect overlaps, not just identical bounds. Count hours by interval union per
  source; never sum retained processing revisions or overlapping windows.
- Add temporal perceptual-hash matching to propose reupload duplicates. Require
  review before grouping perceptual matches; store a reviewed content-group ID.
  Unknown reupload duplication remains visible in reports, not claimed solved.
- Keep speaker, conversation, source/hash and reviewed duplicate-content groups
  entirely within one split. Do not automatically identify people across videos.
- Report raw/acquired/scanned/candidate/reviewed-approved hours separately, plus
  long/short/listening buckets, distinct reviewed speakers, rejection reasons,
  check coverage, unresolved checks and duplicates. Only approved unique footage
  counts toward the 200-300 hour collection target, not a proven training minimum.

### Compatibility and Hugging Face packaging

- Add an explicit import/migration adapter for current candidate records and
  external discovery manifests, including repeated `clean_intervals` revisions.
  Preserve originals, unknown fields and original upstream records in provenance.
  Collapse duplicate interval proposals; do not promote geometry-only screening
  to quality approval. Old reviews require v2 revalidation.
- Preserve the existing training JSONL required keys: `id`, `reference_image`,
  `audio`, `video`, `split`, `source`, `license`. Add interval/processing IDs and
  a QC-metadata reference; keep paths relative to the dataset root.
- Retain current commands; add a `--profile quality-long` option to `process`,
  explicit legacy/discovery import and local Hub-package export commands. Reject
  conflicting profile/config settings clearly. Dry-run/help never download
  models, scan videos or write files. Allow explicit budgets for long recordings;
  report unscanned tails instead of silently treating them as processed.
- Build Hub-ready packages from approved rendered intervals only, with explicit
  train/val/test directories and joined `metadata.jsonl` using `file_name`.
  Empty/unassigned splits must not be fabricated. Keep raw/rejected recordings
  out of the curated package so default VideoFolder loading cannot include them.
- Test local `load_dataset` loading and metadata joins. Publishing to
  `cherthq/facetime-video`, changing its existing files or coordinating a remote
  scheduler is a separate explicit step, not part of writing/implementing this plan.

## 4. Implementation sequence and operational safeguards

1. **Contract first:** v2 validators, fixture records, interval identity, coverage,
   review migration rules and union-based reporting. Preserve existing tests.
2. **Continuity and crop:** sequential PTS decoder, cuts/fades, per-frame tracks,
   span splitting, framing labels and full-frame crop validation.
3. **Visual QC:** frame metrics, OCR/layout evidence and review queue. Review
   evidence includes worst flagged frames plus short audio/video context, not
   only first/middle/last thumbnails; keep viewers and media Git-ignored.
4. **Audio QC:** waveform/VAD, isolated TalkNet/SyncNet workers, calibrated
   window scores and post-render checks. Do not label this phase complete with
   mocks alone or omit speaker/sync work because it needs a separate environment.
5. **Integration:** legacy/discovery import, approved-only exports and local Hub
   round-trip, documentation and a small reviewed end-to-end pilot.

Keep the main CPU preprocessing environment separate from avatar model training.
Use explicit worker subprocesses and versioned JSON input/output for heavier
OCR/audio models, avoiding incompatible global dependency installs. Pin exact
packages, upstream revisions and checkpoint SHA-256 after compatibility tests;
record code and weight terms separately. Extend explicit model setup and doctor
checks; never execute upstream scripts that silently download models at import.

Retain one writer per dataset root, atomic completion records and resumable
content-addressed stage caches. Invalidate only dependent stages when settings
change; reset temporal model state at reproducible shot boundaries. Log elapsed
time, throughput, device and peak memory before choosing worker counts. Support
CPU diagnostics and mocked adapters locally; benchmark heavy workers on remote
hardware rather than assuming the laptop can process 300 hours economically.
Keep all originals, credentials, weights, evidence and generated packages ignored.
No cleanup of old outputs, remote mutations, push or PR change without a request.

## 5. Verification and acceptance

- Unit tests: schema/state validation, absent vs failed checks, timestamp units,
  interval unions, revision deduplication, overlaps, split leakage, stale reviews,
  configuration-sensitive caches, worker failures and explicit overrides.
- Synthetic video tests: hard cuts, brief cutaways, fades, motion without a cut,
  missing/extra subjects, tracking ambiguity, close/wide framing, head/shoulder
  edge exits, VFR/non-zero origins, corrupt frames and interrupted/resumed runs.
- Artifact tests: digital subtitles vs real shirt/sign text, logos inside/outside
  the crop, split screens, natural background blur vs blurred face, under/over
  exposure, static speaker vs frozen frame, and defects occurring between the old
  2 FPS samples. Heuristic uncertainty must not become a fabricated pass.
- Audio tests: silence, natural pauses, offscreen speech, mismatched voice,
  positive/negative injected delay, drift, clipping, dropouts and unsupported
  short speech. Use actual audiovisual speech fixtures for learned-model tests;
  synthetic tones cannot establish active-speaker or lip-sync correctness.
- FFmpeg tests: exact 4:3 output, no upscale/stretch, safe crop every frame,
  common-clock trimming within one output frame, and post-render verification.
- Export tests: no unapproved/raw/rejected rows, required metadata survives local
  Hugging Face loading, real splits only, duplicate revisions do not inflate
  hours, overlapping footage cannot leak across splits, and missing backends
  block automatic eligibility without breaking resumable diagnostic runs.
- Build a rights-appropriate labelled set of 120 segments (60 clean, 60 adverse)
  across at least 12 speakers. Use 40 for threshold calibration and 80 for held-out
  evaluation (40 clean, 40 adverse), separating source/speaker groups. Include all
  four requested problem categories; independently review severe defects.
- Initial go/no-go targets: zero crossings of known cuts/fades in deterministic
  fixtures; no known crop-containment violations marked automatically eligible;
  at most 5% of held-out adverse segments marked eligible; at least 70% retention
  of clean segments as eligible or reviewable rather than rejected. Report sample
  counts, per-category misses, review burden and runtime. These small-pilot targets
  are not a statistical guarantee of artifact-free production footage.
- Finish with a reviewed 2-5 hour source pilot before scaling collection. Report
  unique usable minutes and failure examples; do not claim a 200-300 hour corpus
  is achievable at a particular cost until measured acceptance/throughput exist.

## Primary implementation references

- [PyAV frame presentation timestamps](https://pyav.basswood.io/docs/stable/api/frame.html)
- [PySceneDetect 0.6.7 detectors](https://www.scenedetect.com/docs/0.6.7/api/detectors.html)
- [MediaPipe pose video mode](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python)
- [PaddleOCR text detection](https://www.paddleocr.ai/latest/en/version3.x/module_usage/text_detection.html)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [TalkNet active-speaker detection](https://github.com/TaoRuijie/TalkNet-ASD)
- [SyncNet synchronization checks](https://github.com/joonson/syncnet_python)
- [Hugging Face video metadata and splits](https://huggingface.co/docs/datasets/video_dataset)

These references guide backend choice, not blanket approval of every checkpoint
or input dataset. Exact runtime/weight pins and measured thresholds belong to the
implementation and calibration artifacts, not untested promises in this plan.
