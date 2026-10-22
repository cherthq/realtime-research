# Talking-person video preparation

A review-first CLI for extracting **768 x 576, 4:3 landscape** clips with a
visible head, shoulders and upper torso. It preserves audio/video timing and
records processing provenance, quality checks and review evidence.

```text
Source manifest -> acquire originals -> scan every frame -> select/crop
-> render and verify -> quality workers -> human review -> approved export
```

The recommended profile is `quality-long`. For compatibility, omitting
`--profile` still selects the older `smoke-test` profile, which uses sparse
detections and 5-15 second clips. Do not use that profile as an all-frame audit.
Nothing downloads on import, help or dry-run.

## Requirements and installation

- Python 3.11, Git, and the FFmpeg/FFprobe executables on `PATH`.
- For public YouTube inputs: Deno or Node.js 22+ on `PATH`.
- Persistent storage for full originals, scans, rendered clips and model assets.
- Optional quality models in separate environments; see [worker setup](docs/workers.md).

From the repository root, on Linux:

```bash
python3.11 -m venv .venv-data
. .venv-data/bin/activate
python -m pip install -r data_fetching/requirements-quality.txt
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv-data
.\.venv-data\Scripts\Activate.ps1
python -m pip install -r data_fetching/requirements-quality.txt
```

Do not mix this environment with avatar-model environments or install a second
OpenCV distribution alongside the pinned `opencv-contrib-python` wheel. Direct
requirements are pinned; they are not a complete cross-platform lockfile.

## 1. Prepare a source manifest

Use JSONL with one source per line. Provide exactly one of `path` or `url`:

```json
{"id":"recording_01","path":"./recordings/call.mp4","start":0,"end":null,"source":"authorized recording","license":"unknown","license_evidence":""}
```

- Relative media paths resolve against the manifest's directory.
- `id` must be unique and contain only letters, digits, `_` or `-` (1-100 characters).
- `start` defaults to zero; `end: null` allows the remainder of the recording.
  Both are relative to the source's first video frame.
- For YouTube, use a public HTTPS video URL with its real 11-character video ID.
  Playlists, credentials and bypass options are not accepted. Use `start`/`end`
  rather than URL time parameters.
- Optional grouping fields are `split`, `speaker_id` and `conversation_id`.
  Source/identity groups must be reviewed before training export.
- License fields record provenance, not automatic permission or quality approval.

Keep manifests under ignored `data/facetime/inputs/` or the configured dataset
root. [The example manifest](examples/sources.example.jsonl) intentionally points
to a nonexistent recording; replace its path before fetching.

### TalkingHead-1KH metadata

Obtain the metadata from the [upstream project](https://github.com/tcwang0509/TalkingHead-1KH).
Pass a `*_video_ids.txt` file, not the face-crop `*_video_tubes.txt` table:

```bash
python -m data_fetching import-talkinghead \
  --ids-file /path/to/data_list/train_video_ids.txt --split train \
  --limit 10000 --output data/facetime/inputs/sources.jsonl
```

The explicit import limit bounds the number of unique IDs written. Importing
does not download videos. Original face crops are not reused, and imported
sources remain permission-unverified until reviewed individually.

## 2. Acquire originals and detector assets

All following examples use the default root, `data/facetime/`. To use persistent
external storage, put `--root /storage/facetime` **before** the subcommand and
adjust manifest/output paths accordingly.

```bash
python -m data_fetching fetch --manifest data/facetime/inputs/sources.jsonl --limit 1 --dry-run
python -m data_fetching fetch --manifest data/facetime/inputs/sources.jsonl --limit 1
python -m data_fetching models --dry-run
python -m data_fetching models
```

Fetch copies local inputs or downloads full video/audio originals up to 1080p.
It records attribution, stream metadata and SHA-256 hashes. Completed originals
are checked before reuse; changed source definitions/files are not overwritten.
Private/deleted sources and acquisition failures are recorded in `reports/fetch.json`.
Processing limits do not bound downloaded duration or file size. Stop and defer
requests if the provider rate-limits acquisition; the CLI is not a global
rate-limit scheduler or an access-control bypass.

`models` explicitly retrieves the default MediaPipe face/pose artifacts and
records their URLs and checksums. Processing does not fetch missing models.
For a separate CPU-only YuNet comparison, use a fresh root with
`models --face-backend yunet`; existing detector manifests are not replaced.

## 3. Process a bounded source batch

Start with one source and five minutes on CPU:

```bash
python -m data_fetching process \
  --manifest data/facetime/inputs/sources.jsonl --profile quality-long \
  --config data_fetching/examples/quality-long.json \
  --limit 1 --scan-seconds 300 --max-clips 20 --jobs 1 --dry-run
```

Remove `--dry-run` to execute. Add `--workers data/facetime/inputs/workers.json`
after explicitly configuring [optional quality workers](docs/workers.md).
Without them, the corresponding checks remain `unverified`; diagnostic
processing is still possible.

| Setting | Default | Meaning |
| --- | ---: | --- |
| `--limit` | 5 | Sources visited, in manifest order |
| `--scan-seconds` | 300 | Maximum scanned seconds from each source's `start` |
| `--max-clips` | 20 | Candidate attempts per source |
| `--jobs` | 1 | Concurrent source processes; maximum 64 |
| `--worker-mode` | `persistent` | Reuse OCR/VAD models; `oneshot` reduces idle memory |
| Detector device | `cpu` | Override with `--detector-device gpu` only on a supported runtime |

Quality-long considers 5-300 second continuous candidates and labels intervals
of at least 60 seconds `long`. It detects cuts/fades, assesses face/body geometry
on every decoded frame, chooses fixed torso-preserving crops, and independently
checks rendered frames. A default 0.25-second transition guard excludes boundary
frames. Brief tracking-gap proposals remain uncertain, not automatically clean.

There is no upscaling, padding, restoration, generated torso or joining of separate
shots. Native orientation and square pixels are required. Framing does not prove
eye contact or identity; heuristics do not certify artifact-free footage.

For an hour-long source, explicitly raise `--scan-seconds 3600` and choose a
sufficient candidate budget. Longer sources need a larger scan budget; a manifest
`end` can also truncate coverage. Inspect `unscanned_tail_seconds` and
`candidate_limit_reached` before treating a recording as exhausted. Full-source
inference remains expensive. [Performance and GPU guidance](SPEED.md) explains
input seeking, caches, device requirements and concurrency tradeoffs.

## 4. Inspect results and review candidates

```bash
python -m data_fetching report
python -m data_fetching gallery --output data/facetime/results/round1
python -m data_fetching review-template --output data/facetime/reviews/round1.jsonl
```

Gallery/review output paths must be new. Watch the entire video with audio, not
only preview frames. Each candidate includes MP4, mono 48-kHz WAV, reference PNG,
QC metadata, frame tracks and available worker/flag evidence. The source scan
retains timestamps, raw detections, reasons, coverage and timing measurements.

Reviews start pending. To approve an interval, record:

- Reviewer and ISO UTC timestamp, plus `status: "approved"`.
- Source approval and verified license/evidence fields.
- Consistent speaker/conversation IDs and `train`, `val` or `test` split.
- A documented decision and evidence for each unresolved quality check.

Render failures cannot be overridden. Approvals bind to exact metadata/artifact
hashes, not just filenames. Short or listening-only samples require explicit
`include_short` / `include_listening` decisions. See [evaluation and review](docs/evaluation.md)
for exclusions, duplicate proposals and labelled measurements.

## 5. Export approved samples

```bash
python -m data_fetching export --reviews data/facetime/reviews/round1.jsonl \
  --output data/facetime/training-round1.jsonl --dry-run
python -m data_fetching export --reviews data/facetime/reviews/round1.jsonl \
  --output data/facetime/training-round1.jsonl
python -m data_fetching report --reviews data/facetime/reviews/round1.jsonl
```

Export validates permissions/review fields, artifact hashes, grouping, duplicate
intervals and overlap. Recorded source/speaker/conversation groups cannot cross
splits. Reviewer-assigned identity groups are not automatic face recognition.

Training rows include `id`, `video`, `audio`, `reference_image`, source/provenance,
interval/revision IDs and split/group labels. Media paths are relative to the
dataset root. The pipeline does not perform model-specific loading or training.

Optional `package-hub` creates a local VideoFolder-style package of approved
samples and requires separate `redistribution_approved: true` decisions. It never
uploads or publishes. See [evaluation and review](docs/evaluation.md).

## Storage, reproducibility and limits

The root contains `inputs/`, `originals/`, `sources/`, `models/`, `quality-scans/`,
`quality-renders/`, `clips/`, `reports/`, and explicitly generated review/results.
Keep these outside Git. Use a persistent volume with adequate free space.

Only one CLI writer may own a dataset root. Use `--jobs` for supported source
parallelism rather than launching competing commands against the same root.
After interruptions, rerun the command to reuse validated completed stages.
This is stage-level caching, not frame-level restart or a durable collection queue.
Do not delete a lock until its recorded process has been verified stopped.
Code, model, configuration or registry changes can create new processing revisions
and require re-review. Preserve originals for later model-specific processing.

No source discovery service, distributed scheduler, automatic training approval,
gaze/identity recognition, reliable semantic watermark detector or universal
reupload detector is provided. Local OCR/VAD inference was exercised; exact
TalkNet/SyncNet checkpoints and their calibration still require validation.
Every-frame inspection does not guarantee detection of every defect.

## Tests and optional tooling

```bash
python -m unittest discover -s data_fetching/tests -v
python -m pip check
```

Tests cover schemas, crop geometry, timing, review/export, public documentation,
single-pass selection, worker protocols and process isolation. Media tests use
synthetic FFmpeg/PyAV fixtures and mocked learned geometry; they do not establish
model accuracy. Optional dependency tests may skip on an incomplete environment.
Tests do not download recordings or checkpoints. Linux CI and local Windows
tests passed for the release; GPU throughput and Docker were not validated locally.

Optional utilities are documented rather than run automatically:

- [Worker setup and checkpoint inventories](docs/workers.md).
- [Review, duplicate proposals and labelled benchmarks](docs/evaluation.md).
- [Performance, GPU operation and long recordings](SPEED.md).

For local-file CPU preprocessing, an optional image is available:

```bash
docker build -f data_fetching/infra/Dockerfile -t chert-data-qc .
docker run --rm chert-data-qc --help
```

Mount persistent input/output storage for actual processing. The image does not
provision GPU support, separate worker environments or a YouTube JavaScript
runtime, and it does not include datasets, weights or credentials.
