# Runner handoff: quality-long preprocessing

Updated: 2026-10-22. Send this file with the repository to the person running it.
This is a local batch tool, not a deployed scheduler or a certified 300-hour corpus.
No commands below publish to Hugging Face or start avatar-model training.

For throughput improvements, explicit GPU selection, persistent workers and
source parallelism, see [SPEED.md](SPEED.md). Start with one source/one job and
benchmark on the actual machine before increasing concurrency.

Read [RELIABILITY.md](RELIABILITY.md) for the subsequent tracking/crop reliability
changes and labelled benchmark. Quality-long now defaults to a torso-driven crop;
head/chest and gesture-space profiles are explicit alternatives. Brief detection
gaps may remain reviewable proposals, not certified visible anatomy.

## 1. What is ready, and what still needs validation

Implemented: full-frame timestamp-aware decode; cut/dark-fade boundaries and
guards; face/shoulder continuity; fixed 4:3 crops and all-frame post-render checks;
image/audio metrics; explicit OCR/VAD/TalkNet/SyncNet worker adapters; versioned
metadata, evidence, human approvals, overlap checks and local Hub packaging.

Real FFmpeg/PyAV integration tests use synthetic footage and mocked geometry.
They do not validate learned-model quality. Heavy worker environments, exact
checkpoints, lip-sync offset sign, thresholds, costs and throughput must be smoke
tested and calibrated by the runner before a large batch. A missing worker
produces `unverified`, not a pass. No weights are included or silently fetched.
An additional local API smoke test ran the existing real MediaPipe face/pose
weights over 30 frames of the previously downloaded pilot: all 30 returned one
face and one pose in VIDEO pose mode. This only confirms basic adapter execution;
it does not establish long-run tracking, quality thresholds or corpus suitability.

Heuristics do not reliably recognize all graphics, split screens, beauty filters,
synthetic footage or dubbing. Full human review is still required. OCR only
samples frames; tracking every frame is not a guarantee that every defect is seen.

## 2. Main environment (CPU-capable)

Use Python 3.11; install FFmpeg, FFprobe and Git. For authorized YouTube acquisition
also install Deno or Node >=22. Run commands from the repository root.

Linux:

```bash
python3.11 -m venv .venv-data
. .venv-data/bin/activate
python -m pip install -r data_fetching/requirements-quality.txt
python -m unittest discover -s data_fetching/tests -v
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv-data
.\.venv-data\Scripts\Activate.ps1
python -m pip install -r data_fetching/requirements-quality.txt
python -m unittest discover -s data_fetching/tests -v
```

The main environment has MediaPipe, OpenCV contrib, PySceneDetect and PyAV. Do not
install a second OpenCV wheel in it. Existing avatar-model environments are separate.
Direct dependencies are pinned; this is not a complete cross-platform lockfile.

Optional CPU container: build from the repository root with
`docker build -f data_fetching/infra/Dockerfile -t chert-data-qc .`.
Mount a persistent dataset directory and supply explicit CLI arguments. Container
and host worker paths are different; workers must exist inside the chosen runtime.
The default container action only prints help, not downloads or processing.
The supplied image is for preprocessing local originals; install a supported
JavaScript runtime separately if using yt-dlp inside the container.

## 3. Explicit local checkpoint preparation

Face and pose setup uses the existing explicit command:

```bash
python -m data_fetching --root /storage/facetime models
```

For extra backends, download approved checkpoints separately after checking the
specific code and weight terms. Keep them outside Git under the dataset's `models/`
directory. Do not execute upstream demo scripts: some download other data/models
or delete temporary directories. Our adapters import inference components only.

- VAD: a Silero ONNX checkpoint with `input`, `state`, `sr` inputs, 512-sample
  chunks at 16 kHz, 64-sample context and state shaped `[2, 1, 128]` (v5-style
  interface). An incompatible checkpoint fails visibly; it is not auto-converted.
- TalkNet: official TalkSet checkpoint plus a clean, pinned TalkNet-ASD checkout.
  Adapter loads `model.*` and `lossAV.*` state dictionaries strictly, without
  upstream's CUDA-only training wrapper or optimizer.
- SyncNet: official compatible SyncNet checkpoint and a clean checkout whose
  `SyncNetInstance` supports `device=` and returns offset/confidence/distances.
  Pin the tested revision, not a moving branch.
- OCR: local `PP-OCRv5_mobile_det` inference directory, including its
  complete configuration/parameter inventory. No recognition model is required.
  The worker uses PaddleX's direct text detector to avoid document/LLM extras.
  It selects an installed local Arial/DejaVu/Liberation font, or registry
  `font_file`; it must not fetch visualization fonts. Python network connections
  are blocked during worker inference (not an OS sandbox).

Official sources (not blanket checkpoint approval):
[Silero](https://github.com/snakers4/silero-vad),
[TalkNet](https://github.com/TaoRuijie/TalkNet-ASD),
[SyncNet](https://github.com/joonson/syncnet_python),
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR).

Separate worker environments:

```bash
python3.11 -m venv .venv-qc-audio
.venv-qc-audio/bin/python -m pip install -r data_fetching/requirements-audio-worker.txt
python3.11 -m venv .venv-qc-ocr
.venv-qc-ocr/bin/python -m pip install -r data_fetching/requirements-ocr-worker.txt
```

For VAD alone, use `requirements-vad-worker.txt` instead of the Torch audio
environment. Preserve the selected model's upstream terms and hashes. On minimal
Linux OCR hosts install `fonts-dejavu-core` or supply a local `font_file`.

Windows: use each environment's `Scripts/python.exe` instead of `bin/python`.
Audio pins give a portable baseline; select/test an appropriate PyTorch CUDA
wheel explicitly for GPU use. OCR pins are a candidate stack, not a remotely
verified lock. Do not mix Paddle and MediaPipe dependencies into one environment.

Copy `data_fetching/examples/workers.example.json` to the ignored dataset's
`inputs/workers.json`. Its entries are deliberately disabled. For each backend:

1. Set absolute interpreter/model/repo paths (recommended). Relative paths are
   resolved against the registry file, not the shell directory.
2. Record the exact upstream commit (`git -C /path/to/repo rev-parse HEAD`). A
   modified/untracked upstream checkout is rejected; keep weights outside it.
   The template pins the source APIs inspected for this implementation: TalkNet
   `6d6821479af485e251c4991487e40573b42181b4` and SyncNet
   `907c0b579c2e2d83f0eae1b2ac9e720cde4e5623`. These are source pins, not a claim
   that checkpoint inference has been validated on the runner.
3. List every model asset and its SHA-256. `python -m data_fetching.asset_manifest
   /path/to/model-or-directory` prints the inventory without downloading anything.
4. Review code/weight terms, set `terms_reviewed: true`, record `license_evidence`,
   then enable that backend. Setting these fields is the runner's explicit decision.
5. Leave calibration fields false/null until real labelled tests establish them.
   The registry fingerprint includes assets, revision and installed worker packages.

```bash
python -m data_fetching --root /storage/facetime doctor --profile quality-long --workers /storage/facetime/inputs/workers.json
```

Doctor is read-only and checks configuration/dependencies/hashes, not inference
accuracy. It returns nonzero if an expected worker is not configured. Processing
can still run diagnostically with workers missing, but results stay review-needed.

## 4. Small batch first

Prepare source JSONL using the existing schema: exactly one of `url`/`path`, a
unique `id`, optional start/end, and actual license/evidence. Discovery is not
approval. For an external discovery/legacy candidate manifest:

```bash
python -m data_fetching import-discovery --input /storage/discovery/sources.jsonl --output /storage/facetime/inputs/sources-v2.jsonl --provenance /storage/facetime/inputs/import-provenance.jsonl
```

The adapter preserves upstream records, collapses duplicate interval proposals and
does not preserve old training approvals. Proposed clean intervals remain
unverified; the source is rescanned rather than trusting sparse geometry checks.

Inspect budgets before downloading. Fetch preserves full originals; scan budgets
limit processing, not download size. Start with one authorized recording:

```bash
python -m data_fetching --root /storage/facetime fetch --manifest /storage/facetime/inputs/sources-v2.jsonl --limit 1 --dry-run
python -m data_fetching --root /storage/facetime fetch --manifest /storage/facetime/inputs/sources-v2.jsonl --limit 1
python -m data_fetching --root /storage/facetime process --manifest /storage/facetime/inputs/sources-v2.jsonl --profile quality-long --workers /storage/facetime/inputs/workers.json --limit 1 --scan-seconds 300 --max-clips 20 --dry-run
python -m data_fetching --root /storage/facetime process --manifest /storage/facetime/inputs/sources-v2.jsonl --profile quality-long --workers /storage/facetime/inputs/workers.json --limit 1 --scan-seconds 300 --max-clips 20
```

For a complete one-hour source, raise `--scan-seconds 3600` explicitly and ensure
the source manifest's `end` does not stop earlier. Full-frame inference is much
more expensive than the old 2 FPS scan. Reports expose unscanned tails and a
candidate-budget warning. Short clean remnants stay labelled `short`, not padded
to one minute. Do not use short pilot manifests to infer full-source coverage.

Existing `--config` under quality-long accepts QualityConfig fields only; smoke
configs using `sample_fps`/`valid_fraction` fail instead of silently mixing policies.
Omitting `--profile quality-long` retains the old smoke-test behavior.

## 5. Inspect evidence and run model smoke tests

Each v2 clip directory contains video/audio/reference, output-tracks JSON,
quality-details, worker request/responses/logs, first/middle/last frames and
context media for flagged checks. `candidate.json` stores checks, actual coverage,
crop, stable interval ID, processing revision and all artifact hashes. Per-frame
source analysis is compressed under `quality-scans/`. Memory is bounded to one
candidate window rather than the full decoded recording.

After configuring a backend, rerun processing: registry changes create a new
processing revision, not new unique footage. To smoke-test one generated request
directly in the relevant worker environment (choose a NEW response filename):

```bash
/path/to/worker/python -m data_fetching.quality_worker --request /storage/facetime/clips/CLIP_ID/syncnet-request.json --response /storage/facetime/reports/syncnet-smoke-response.json
```

Requests only exist for enabled workers. Use a real speech clip with a known
speaker and known sync. Test positive and negative injected audio delays and
offscreen/mismatched speech. Models are not validated by a successful import or a
tone-only synthetic test. SyncNet's raw convention is
`vshift - best_distance_index` at 25 FPS; offset sign must be tested before enabling
`sync_sign_validated`. Score/calibration evidence is attached to the registry.

Heuristic/manual flags are intentionally retained. If a worker times out/crashes,
check its log; it cannot clear itself as passed. Model windows use analysis-only
25 FPS face proxies, not altered training media. Unsupported inputs stay unresolved.
To retry a completed candidate with transient worker errors, add/change a
`retry_token` string in that backend's registry entry. This creates a new auditable
QC revision while reusing unchanged full-frame scans and validated render caches.
Do not hand-edit completed responses or candidate metadata; hashes bind approvals.

## 6. Review, remove bad spans and export

```bash
python -m data_fetching --root /storage/facetime report
python -m data_fetching --root /storage/facetime review-template --output /storage/facetime/reviews/round1.jsonl
```

Watch each entire candidate, not only previews. Fill reviewer, UTC timestamp,
license/evidence, source approval, speaker/conversation and split. Every unresolved
check needs a per-check reason and evidence for a human override. Render failures
cannot be overridden. Speaker IDs are reviewer labels, not face-track numbers.
Geometry does not prove eye contact or speaker identity.

Default export excludes short/listening candidates; set `include_short` or
`include_listening` per review only when intentionally building those buckets.
Approvals bind to exact QC/artifact hashes and cannot transfer across revisions.
Export rejects overlapping source intervals and cross-split source/speaker groups.

Confirmed bad spans can be supplied as JSONL through `process --exclusions FILE`:

```json
{"source_id":"recording_01","start":42.0,"end":48.0,"reason":"B-roll confirmed during review","evidence":["review-log-123"]}
```

Times are relative to the original video's first frame. Reprocessing excludes
these spans and reassesses clean pieces; it does not erase old evidence. Candidate
IDs change, but unique-hour reports do not double-count overlaps/revisions.

```bash
python -m data_fetching --root /storage/facetime duplicates --output /storage/facetime/reports/duplicate-proposals.json
python -m data_fetching --root /storage/facetime export --reviews /storage/facetime/reviews/round1.jsonl --output /storage/facetime/training-round1.jsonl --dry-run
python -m data_fetching --root /storage/facetime export --reviews /storage/facetime/reviews/round1.jsonl --output /storage/facetime/training-round1.jsonl
python -m data_fetching --root /storage/facetime report --reviews /storage/facetime/reviews/round1.jsonl
```

Perceptual duplicate proposals use aligned temporal hashes, not a comprehensive
reupload detector. Human-confirmed matches receive `duplicate_content_group` in
reviews. Match groups cannot cross splits; unseen people/reuploads remain a review
responsibility. Raw candidate sums are not usable-hour counts.
Within a reviewed duplicate group, export allows only one canonical original
recording (multiple non-overlapping intervals from that original are fine), so
known reuploads cannot inflate approved hours.

Optional local package (requires separate `redistribution_approved: true`):

```bash
python -m data_fetching --root /storage/facetime package-hub --reviews /storage/facetime/reviews/round1.jsonl --output /storage/facetime/packages/round1
```

This creates approved-only MP4s, split metadata and companion files. Load locally
with `load_dataset("videofolder", data_dir="/storage/facetime/packages/round1")`
in a separate environment using `requirements-export.txt`. Inspect row counts, real splits and metadata
before any upload. No raw/rejected recordings enter this package. QC references
can point to private evidence; audit metadata for publication before sharing it.
Only the reviewer may approve redistribution. No Hub credentials are needed here.

## 7. Operational limits and rollout gate

- One writer per dataset root; use separate roots for independent workers. The
  lock is not a distributed scheduler. Do not delete a lock until its process
  is confirmed stopped. Re-run the same command to reuse validated stage caches.
- Never commit media, model files, credentials or review/private evidence. All
  generated artifacts belong under ignored data/checkpoints/runs directories.
- Pin and archive the runner's `pip freeze`, FFmpeg version, registry and tested
  upstream revisions alongside the run. Check free disk before fetching originals.
- Before 300 hours: checkpoint smoke tests, labelled threshold calibration and a
  reviewed 2-5 hour source pilot; then a 20-30 hour batch measuring usable yield,
  review minutes per usable hour, CPU/GPU throughput, disk use and defect misses.
- No automated threshold or model score is a production quality guarantee. The
  labelled acceptance set and real throughput benchmark in `plan.md` remain
  runner work, not results produced by this code handoff.
