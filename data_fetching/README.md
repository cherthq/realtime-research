# FaceTime-style data fetching

Review-first tooling for **4:3 landscape, 768 x 576** clips showing a person's
head, shoulders and upper torso. All code, configuration, examples and tests live
in this folder. Originals, model artifacts and generated data stay in the ignored
`data/facetime/` directory. No avatar-model environment is modified.

Small [checked-in pilot samples](pilot_samples/README.md) demonstrate the framing
without needing downloads or the local HTML gallery. Only this curated media
bundle is included in Git; full source recordings and other outputs stay ignored.

See [the discovery/scheduler handoff](DISCOVERY.md) for the 200-300 usable-hour
collection target and longer-segment approach. The handoff describes collection
requirements; the 15-second smoke-test default is unchanged.

The [persistent queue extension](QUEUE.md) now supplies metadata search,
checkpointed whole-recording screening, canonical timeline accounting and
explicit Hugging Face publication. It is opt-in through
`python -m data_fetching.discovery`; recurring collection is not enabled by
installing or importing it. Existing smoke-test processing defaults are unchanged.

**Collection is explicit; inspect the code and inputs before starting a run.** Nothing
downloads just by importing this package, displaying help, or doing a dry run.
The automated tests use synthetic local footage or mocks, not YouTube videos.

## Review the code first

- `core.py`: source/config validation, crop geometry and sample-quality checks.
- `media.py`: timestamp-aware FFmpeg cropping, audio and reference extraction.
- `vision.py`: scene detection, MediaPipe adapters and overlay previews.
- `pipeline.py`: resumable fetching/processing, explicit approvals and export.
- `__main__.py`: CLI. Run it from the repository root, not inside this folder.

Inspect CLI commands without installing the vision libraries:

```bash
python -m data_fetching --help
python -m data_fetching fetch --manifest data_fetching/examples/sources.example.jsonl --dry-run
python -m unittest discover -s data_fetching/tests -v
```

The example manifest intentionally points to a nonexistent local recording.
Replace it with your own authorized source before fetching. Relative local paths
are resolved against the input manifest's directory, not the shell directory.
FFmpeg integration tests skip if its binaries are unavailable. Optional scene
tests skip if their Python dependencies are unavailable. No tests download models.

## Isolated setup (when ready)

Use Python 3.11. Windows PowerShell, from the repository root:

```powershell
py -3.11 -m venv .venv-data
.\.venv-data\Scripts\python.exe -m pip install -r data_fetching/requirements.txt
.\.venv-data\Scripts\Activate.ps1
```

Linux/WSL:

```bash
python3.11 -m venv .venv-data
. .venv-data/bin/activate
python -m pip install -r data_fetching/requirements.txt
```

Install **FFmpeg and FFprobe binaries** on PATH; a Python ffmpeg package is not a
replacement. YouTube fetching also needs yt-dlp's JavaScript support: the default
extra supplies `yt-dlp-ejs`; use Deno or Node >=22 on PATH. Fetching prefers Deno
and explicitly enables Node when Deno is absent. Do not
install both opencv-python and opencv-contrib-python into this environment.

```bash
python -m data_fetching doctor
python -m data_fetching models --dry-run
```

`doctor` is read-only and returns nonzero if requirements or detector artifacts are
missing. It is expected to report missing detectors before explicit setup below.
The requirements pin the direct dependencies; transitive dependencies are not a
cross-platform lockfile. Candidate metadata records the installed detection
package versions, Python, FFmpeg/FFprobe, code fingerprint and detector hashes.

## Inputs

Custom JSONL, one source per line:

```json
{"id":"recording_01","path":"./recordings/call.mp4","start":0,"end":60,"source":"consented recording","license":"unknown","license_evidence":"","split":null,"speaker_id":null,"conversation_id":null}
```

For YouTube, replace `path` with `url`, for example a canonical
`https://www.youtube.com/watch?v=VIDEO_ID` URL. Use a real 11-character video ID.
Only public HTTPS YouTube video URLs are accepted, not arbitrary websites,
playlists, embedded credentials or bypass options. URL input is normalized to one
video; any URL time query is ignored: use explicit `start` and `end` fields.

- `id`: unique letters/digits/underscores/hyphens, at most 100 characters.
- Exactly one of `path` or `url` is required. Start/end are seconds relative to
  the source's first video frame, not absolute container timestamps.
- `end` is optional; `start` defaults to zero. One ID/location per recording;
  select a larger interval instead of duplicating its URL under aliases.
- `split` may be train/val/test or null. Identity and recording grouping must be
  reviewed before export, even when upstream splits exist.
- `license` and `license_evidence` are provenance, not automatic source approval.

### TalkingHead-1KH source IDs

Download and extract the upstream metadata archive yourself when ready. Pass an
`*_video_ids.txt` file, **not** the `*_video_tubes.txt` face-crop table:

```bash
python -m data_fetching import-talkinghead --ids-file /path/to/data_list/small_video_ids.txt --limit 5 --output data/facetime/inputs/small.jsonl
```

Use `--split train` or `--split val` for the appropriate upstream list. The small
subset is not itself a split. Importing writes only a source manifest; it does
not fetch video. Upstream describes its original sources as CC BY 3.0, but this
tool marks imported license status unknown until an individual source is checked.
Original tube coordinates and minute-segment offsets are deliberately not reused.
This is a new torso-framing pipeline, not an exact reproduction of the benchmark.

## Staged collection

Every command defaults to `data/facetime/`. For another root, put the global
argument before the command: `python -m data_fetching --root /storage/facetime ...`.

### 1. Fetch originals only

```bash
python -m data_fetching fetch --manifest data/facetime/inputs/small.jsonl --limit 5 --dry-run
python -m data_fetching fetch --manifest data/facetime/inputs/small.jsonl --limit 5
```

Fetch copies local inputs, or downloads video plus audio with a 1080p cap. It
preserves originals and attribution, retries bounded failures, and resumes partial
YouTube downloads. Downloading is sequential, with two concurrent fragments.
There is no duration cap on downloaded originals: the later scan cap limits
processing, not download size. Check source lengths and available storage first.

Completed source records are SHA-256 checked before reuse. A changed original or
source definition fails without resetting it; choose a new ID or dataset root.
Failed sources are recorded in `reports/fetch.json` and cause a nonzero exit code.

### 2. Install detectors explicitly, then process

```bash
python -m data_fetching models
python -m data_fetching process --manifest data/facetime/inputs/small.jsonl --limit 5 --max-clips 20 --scan-seconds 300 --config data_fetching/default_config.json --dry-run
python -m data_fetching process --manifest data/facetime/inputs/small.jsonl --limit 5 --max-clips 20 --scan-seconds 300 --config data_fetching/default_config.json
```

`models` downloads Google's version-1 BlazeFace short-range and pose-landmarker
lite artifacts. URLs and locally computed SHA-256 hashes are saved and checked
on reuse; hashes are not an independently signed upstream authenticity guarantee.
No automatic model downloads occur inside processing.

By default processing scans only the first 300 seconds from each input's `start`,
attempts up to 20 clips per source and visits at most five sources. Raise these
limits explicitly to expand a run. `--source-id recording_01` selects one manifest
record before applying the source limit. Scene detection still scans the whole
bounded interval, even if the candidate count limit is reached later.

PySceneDetect finds scene boundaries; clips are 5-15 seconds and never joined
across detected cuts. MediaPipe samples approximately twice a second. At least
90% of samples must have a single face/pose with both shoulders confidently
visible and a plausible face/shoulder relationship. Crop bounds contain the
union of valid sampled regions, including configurable space below shoulders.

One fixed, even-aligned 4:3 crop is used for each clip. No stretching, padding,
synthetic body generation, face-only fallback, or upscaling is allowed. Inputs
must already have normal orientation and square pixels; unsupported rotated or
ambiguous-timestamp media fail with an explanation instead of silently warping.
Scene timing uses PySceneDetect/OpenCV: variable-frame-rate sources need special
review of cut boundaries before training use. Export does not force a new FPS.

FFmpeg trims audio and video on a common timestamp origin, preserving audio
delay, then exports H.264/AAC MP4, mono 48 kHz PCM WAV and the first output frame
as a reference PNG. First/middle/last source previews have detection/crop overlays.
Candidate metadata is written last, after output validation and hashing.

Processing caches depend on source, code, configuration, runtime and detector
hashes. Changed settings create new clip IDs; existing valid artifacts are never
replaced by a new configuration. Same-configuration completed outputs are
validated before reuse; interrupted attempts can be retried. Retained revisions
can represent the same time interval: do not approve duplicate versions.

### 3. Review, then export

```bash
python -m data_fetching report
python -m data_fetching review-template --output data/facetime/reviews/round1.jsonl
```

Edit the generated JSONL in your editor. Every candidate starts with
`status: "pending"` and `source_approved: false`. Watch the actual MP4 as well as
the previews; check framing, visible torso, person consistency, camera-facing
appearance, speech/voice match, quality and lip-sync. Previews are not sufficient
to establish quality for every frame.

For an approved candidate set:

- `status`: `"approved"` (or `"rejected"` for a rejected candidate).
- `source_approved`: `true` only after source permissions have been checked.
- `license` and `license_evidence`: the verified basis and permission reference.
- `split`: `"train"`, `"val"` or `"test"`.
- `speaker_id` and `conversation_id`: consistent reviewer-assigned group labels.
  Use the same speaker label across recordings of the same person.

```bash
python -m data_fetching export --reviews data/facetime/reviews/round1.jsonl --output data/facetime/training-round1.jsonl --dry-run
python -m data_fetching export --reviews data/facetime/reviews/round1.jsonl --output data/facetime/training-round1.jsonl
```

Export rejects unknown/duplicate reviews, failed/rejected candidates, missing
approvals, unknown permissions, missing groups/splits, changed artifacts,
duplicate source intervals from retained processing revisions, and cross-split
reuse of speaker, conversation, source ID or original file hash.
It validates hashes and media again. Only approved rows are exported. Paths in
the training manifest are relative to the **dataset root**, regardless of where
the output manifest is saved. Input/review/export commands refuse to overwrite
existing manifests; choose a new output filename for each round.

## Storage and failure handling

```text
data/facetime/
  inputs/          # Your source JSONL files
  originals/       # Downloaded/copied source media and attribution
  sources/         # Successful source definitions, hashes and probe metadata
  models/          # Explicit detector downloads and checksums
  scenes/          # Content-addressed scene caches
  clips/<id>/      # Detections, previews, video, audio, reference, candidate.json
  reviews/         # Human decisions, separate from generated metadata
  reports/         # Latest fetch/process summaries
```

Use one writer per dataset root. A lock prevents concurrent commands from
overwriting state. If a process crashes, verify it has stopped before manually
removing `.pipeline.lock`. No automatic recursive cleanup is performed. Existing
media, private approval records, cookies and credentials must not be committed.

Reports count automatic candidates/rejections and candidate hours, **not** approved
training hours. Counts include retained processing revisions. The export result
is the authoritative list of reviewed training samples.

## First visual pilot (2026-10-21)

Three public YouTube interviews from Creative Commons' Open Culture Voices
series were fetched and processed with the default configuration and real
MediaPipe inference:

- [Mariana Ziku](https://www.youtube.com/watch?v=PKfyqkIJQwY)
- [Julia Pagel](https://www.youtube.com/watch?v=CkV_xSfurrI)
- [Alwaleed Alkhaja](https://www.youtube.com/watch?v=AvXt8nVpYKk)

Each source was scanned from 25 to 70 seconds, producing three 15-second
candidates. The strict filter rejected **9/9**: the main sample-level reason was
`torso_region_outside_source`; some intervals also included a question card or
lost detections. These webcam shots show the head/shoulders/upper chest, not the
deeper torso region demanded by the default `torso_depth: 1.0` heuristic. This
does not establish that every rejected clip is unsuitable for every avatar task.

For visual inspection only, separate full-height 4:3 crops were centered using
the median detected face position on a clean interval from each source. Those
960 x 720 crops were resized to 768 x 576 without padding or upscaling, using the
same timestamp-aware media exporter. They **bypass the torso acceptance test**
and are labelled `preview_only`; none are training approved or exported. The
default acceptance configuration has not been loosened.

Local, Git-ignored results are under `data/facetime/pilot-results/`:
`index.html`, `comparison.jpg`, `three-video-framing-preview.mp4`, individual
15-second clips and `pilot-report.json`. Full strict records remain under
`clips/`. The compilation is standardized to 25 FPS for viewing only; individual
clips retain their source frame timing. This pilot validates execution and shows
a source-selection/torso-threshold limitation, not dataset quality or training
permission. It also exposes source logos/text overlays that need later review.

An approximately 7.3 MiB preview bundle is also committed under `pilot_samples/`,
excluding HTML, originals and detector weights. Its attribution file records the
publisher's explicit CC BY 4.0 statement found after the local pilot. This does
not mark the samples training-approved.

## Limits and attribution

This is a local batch foundation, not a distributed crawler. Source lists do not
prove permission for commercial training. Face and pose models estimate geometry:
they do not guarantee torso visibility, eye contact, active speech or identity.
Short occlusions, scene-detector misses, VFR timing and quality remain review
items. No training, distillation, face identification, precise gaze estimation or
active-speaker model is included.

This is an independent reimplementation inspired by
[TalkingHead-1KH](https://github.com/tcwang0509/TalkingHead-1KH), whose metadata and
scripts are provided under its MIT license. Download its metadata separately and
retain upstream notices. Cite Wang, Mallya and Liu, *One-Shot Free-View Neural
Talking-Head Synthesis for Video Conferencing*, CVPR 2021, when using that dataset.
See upstream for its video-license and removal policies; script licenses and
rights in individual recordings are separate.

Primary implementation references:

- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [PySceneDetect](https://www.scenedetect.com/docs/0.6.7/)
- [MediaPipe face detector](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector/python)
- [MediaPipe pose landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python)
