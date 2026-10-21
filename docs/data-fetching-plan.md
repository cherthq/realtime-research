# FaceTime-style data fetching plan

Status: implemented in `data_fetching/`. A three-YouTube-source visual pilot ran
on 2026-10-21 with real face/pose inference. All nine strict candidates were
rejected, predominantly because the requested torso region extended below these
close-up webcam frames. Separately labelled 4:3 framing previews were generated
for inspection; no training samples were approved or exported. Originals,
detectors, reports and previews are Git-ignored under `data/facetime/`.
See [usage, pilot findings and limitations](../data_fetching/README.md). Offline
tests additionally cover synthetic media and mocked detector outputs.

## Goal and agreed defaults

Add a standalone `data_fetching/` folder for collecting original videos and
producing image-and-audio-driven avatar training candidates. Each clip should
show one person's head, shoulders and upper torso, rather than a tight face crop.

- Default framing: **4:3 landscape (width:height)**, replacing the earlier 3:4
  portrait proposal. Default output resolution: **768 x 576**.
- Inputs: TalkingHead-1KH source IDs plus custom YouTube URLs and local videos.
- Automatically detect faces and body landmarks; manually review candidates
  before admitting them to training exports.
- Keep original media so future model-specific crops do not require downloads.
- Use a separate CPU-capable Python 3.11 preprocessing environment on Windows/WSL
  or Linux; do not modify the avatar models' environments.

The aspect ratio is our product target, not a claim that every FaceTime device or
orientation uses a fixed 4:3 format. Cropping cannot recover a torso that is absent
from the original footage.

## Inputs and fetching

- Provide CLI commands for fetching sources, processing candidates and exporting
  reviewed samples. Start with a small explicit batch limit, not a full-dataset
  download.
- Import TalkingHead-1KH's video IDs as source candidates. Re-detect suitable
  intervals on the downloaded originals instead of reusing its tight face crops
  or assuming its old minute-segment/frame offsets match current downloads.
- Accept custom JSONL inputs containing a URL or local path, optional start/end
  times, and source/license information. Local inputs need no network access.
- Use yt-dlp to download both video and audio, up to 1080p. Preserve source
  metadata, attribution and original media.
- Default to one download worker, bounded retries, resumable downloads,
  deduplication and per-source failure reports. Do not bypass access restrictions.
- Keep fetching separate from processing so failed or changed crops do not
  trigger another download.

## Automatic clip selection and cropping

- Split sources at scene boundaries with PySceneDetect. Generate non-overlapping
  candidate clips of 5-15 seconds within each scene; discard shorter remnants.
- Sample at 2 FPS using MediaPipe face detection and pose landmarks. Require one
  unambiguous face and confident detections of both shoulders in at least 90% of
  sampled frames. Make thresholds configurable and record rejection reasons.
- Compute one fixed crop rectangle per clip from its sampled detections. Include
  the head and both shoulders, with space below them for the upper torso; retain
  arms and gestures where the source and 4:3 frame allow.
- Expand the required region to 4:3, then translate the rectangle within source
  bounds without cutting off the required region. Reject candidates where that
  region cannot fit; never stretch, synthesize missing body regions or silently
  fall back to a face-only crop.
- Resize valid crops to 768 x 576 without upscaling undersized source crops.
  Keep crop bounds even-sized for video encoding.
- Preserve source timing and frame rate. Trim video and audio against the same
  interval and time origin; do not independently reset streams in a way that
  loses their original synchronization.
- Generate first/middle/last-frame previews with detection and crop overlays.
  Review framing, visible speech, camera-facing appearance and lip-sync.

The crop geometry is deterministic given the same detections and configuration.
Pin detector versions and persist detection results and crop coordinates for
reproducibility; do not promise bit-identical neural inference across platforms.
Face/pose detection alone does not prove eye contact, active speech, identity
consistency or genuine torso visibility throughout a clip.

## Outputs and review

- Store originals, processed clips, WAV audio, reference images, previews and
  manifests under ignored `data/facetime/`; commit no downloaded media or weights.
- Record source ID/URL, attribution/license evidence, source and clip timestamps,
  crop coordinates, detector settings, dependency versions and review status.
- Keep candidates separate from training manifests until explicitly approved.
  Export approved samples using the existing contract in `docs/experiments.md`:
  `id`, `reference_image`, `audio`, `video`, `split`, `source` and `license`, with
  paths relative to the dataset root.
- Preserve available source split labels in candidate metadata. Custom samples
  remain unassigned until identity- and recording-aware split review; only
  approved samples with a valid train/val/test assignment enter training exports.
  Check identity overlap even when upstream split labels exist.
- Track source approval separately from visual-quality approval. Membership in
  an upstream list is not automatic permission for commercial model training.
- Resume only validated outputs whose input and processing configuration match;
  interrupted or configuration-mismatched outputs must not be marked complete.

## Implementation boundaries

Include isolated dependency requirements, example manifests, setup/run commands,
upstream attribution and explicit detector-weight download instructions. Keep all
credentials, originals and detector artifacts outside tracked source files.

Do not add model training, fine-tuning, inference changes, distributed scheduling,
automatic active-speaker verification or precise gaze estimation in this version.
Wider or alternate-aspect crops can be generated later from retained originals.

## Verification and pilot acceptance

- Unit tests: input validation, duplicate IDs, crop geometry, boundary clamping,
  aspect ratio, rejection reasons and configuration-sensitive resume behavior.
- FFmpeg integration tests: 4:3 output at 768 x 576, retained audio, no stretching
  or upscaling, and audio/video timing within one video frame of the expected
  common trim interval, including inputs with non-zero stream offsets.
- Detection tests: suitable upper-body subject, tight face close-up, missing
  shoulders, multiple people, subject motion and scene changes.
- First run a small pilot, manually inspect crops and lip-sync, and report
  accepted/rejected counts, rejection reasons and usable hours before scaling.

## Primary references

- [TalkingHead-1KH source, metadata and license notes](https://github.com/tcwang0509/TalkingHead-1KH)
- [yt-dlp setup and downloader options](https://github.com/yt-dlp/yt-dlp)
- [MediaPipe face detection](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector/python)
- [MediaPipe pose landmarks](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python)
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect)
