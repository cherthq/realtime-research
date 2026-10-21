# Persistent long-footage queue

The queue implements the durable state and sampled long-interval discovery
foundation described in DISCOVERY.md. It does not schedule itself. The initial
pilot is capped at 10 acquired raw hours and 50 GB of stored originals across
runs. Recurring collection remains disabled until pilot review. The intended
corpus target is 250 approved hours in intervals at least 60 seconds long.

## Setup and one bounded run

Use the isolated data environment from README.md. Python 3.12 is also supported
for this extension. Install `discovery-requirements.txt` for Hub upload support.
Use Node >=22 or Deno for YouTube. Install detector artifacts explicitly with
`python -m data_fetching models`; no command implicitly installs them.

```bash
python -m data_fetching.discovery search 'original camera-facing educational interview'
python -m data_fetching.discovery add /path/to/discovery-sources.jsonl
python -m data_fetching.discovery run --seconds 1800 --no-search
python -m data_fetching.discovery report
```

Search performs metadata discovery only. Review results for original uploader,
length, framing and suitable acquisition route. Select a source for acquisition
by importing its URL with `acquisition_selected: true`. Source selection does
not assert permission or quality. Example:

```json
{"url":"https://www.youtube.com/watch?v=abcdefghijk","title":"Example only","duration":1800,"uploader":"Original uploader","license":"unknown","source_page":"https://publisher.example/video","acquisition_selected":true}
```

IDs are derived from canonical URLs. Local-file candidates accept `path` instead
of `url`; those files must be authorized originals. New records are pending.
Refreshes merge metadata without resetting existing rights or review decisions.
The current downloader supports public YouTube and local files. Other discovery
sources need an explicit permitted acquisition adapter or local-file import.

The default query rotation is a starting search frontier. Broad keyword results
can be poor: inspect and select results before they consume an acquisition
budget. Search cursors advance per query, and queries rotate across wakeups.
After 200 results, a timestamped new exploration cycle revisits updated results.

## Persistent storage and restart behavior

`data/discovery/queue.sqlite3` is authoritative. It stores source state, search
cursors, scan coverage, review decisions, canonical timeline mappings, upload
receipts and an append-only event history. Atomic JSON sample checkpoints live
under `scans/`. `candidates.jsonl` is a snapshot interchange format; it is not
the mutable queue. All files remain Git ignored. Back up the entire `data/`
directory on persistent storage, including originals and the SQLite database.
Use SQLite's backup API or stop the writer before copying its database/WAL.

Originals, SHA-256 records and detectors use `data/facetime/`. All mutation and
media commands share that dataset's existing single-writer lock. Readers can
inspect SQLite snapshots while collection is running. After a crash, confirm
the old process is stopped before removing a stale `.pipeline.lock`.

Run budgets are cumulative across invocations, with conservative per-source
duration/space preflight checks and a download-byte hook. Failed sources record
errors and retry backoff. Unknown-duration/live streams are refused. No cloud
resources are provisioned. A window or download already in progress may finish
after a nominal wall-clock deadline; checkpoints are the interruption boundary.
Merge temporary files require extra free disk headroom. Local data is never
deleted after an upload.

## Long intervals and quality limits

Scenes are detected over the full recording and cached. Face/pose checks decode
once per 120-second computational checkpoint at 2 FPS. Samples are stored before
coverage is committed. On restart, cached chunks are reused and scanning advances
through remaining scenes. Complete scenes are evaluated across checkpoint
boundaries, retaining spans over five minutes instead of cutting them into
fixed windows. Failed samples split runs. Separate scenes are never joined.

The explicit framing profiles are `head_upper_chest` (0.25 shoulder widths below
the shoulders) and `wider_torso_gestures` (1.0), with the same 4:3/no-upscaling
geometry checks. These are screening heuristics, not verified body visibility.
They do not alter the original pipeline's 5–15-second smoke-test defaults.

Every proposed interval retains `sampled_video_geometry_only` inspection status.
Overlays, captions, logos, filters, compression, natural speech, dubbing, active
speaker matching, VFR boundaries, and between-sample defects require review.
There are no automatic OCR, active-speaker, or full-frame cleanliness guarantees.
No proposal is counted as approved by the scheduler. Short clean intervals below
60 seconds remain outside the main preferred-duration target.

## Source and interval review

```bash
python -m data_fetching.discovery review-template > /path/to/reviews.jsonl
python -m data_fetching.discovery source-review /path/to/source-decisions.jsonl
python -m data_fetching.discovery review /path/to/reviews.jsonl
```

Each source decision contains `id`, `reviewer`, `evidence`, `permission`
(`pending`, `approved`, `rejected`) and `redistribution` (same choices). Training
permission and public redistribution are distinct. Preserve source URL,
uploader, creator credit, license/version and evidence. A public CC label is a
discovery clue; source-specific evidence needs inspection. Do not publish private
agreements in public evidence fields: use a public evidence reference or redacted
summary. Only redistribution-approved originals enter Hub upload plans.

Duplicate review uses either `unique: true` or `canonical: SOURCE_ID` plus an
`offset`: canonical time = this source time + offset. Canonical roots cannot
point through another alias. Never infer speaker identity from detector output.
Exact hashes group automatically. Temporal perceptual hashes propose probable
reencoded/overlapping reuploads, requiring manual alignment. Cropping, short
overlaps, and similar backgrounds can confuse this heuristic. Unresolved sources
stay outside confirmed unique-hour totals, even when no match is proposed.

Interval reviews require `quality`, `reviewer` and `notes`. Approval also requires
`inspection: full_video_audio`, reviewed `speaker_id`/`conversation_id`, a valid
`split`, and prior source permission/duplicate review. Watch/listen through the
complete interval; previews alone do not qualify. Cross-split speaker, recording
and canonical-source reuse is rejected. Source decisions can revoke eligibility;
reports re-evaluate the gates rather than blindly preserving approved totals.

The queue holds source footage and interval decisions. Training-ready crops and
model-specific exports remain a later explicit step through the original
pipeline; queue approval alone does not create training media artifacts.

## Accounting

Reports give provisional raw discovered/acquired duration separately from
duplicate-reviewed unique raw duration. Candidate duration remains provisional
until duplicate review, and every candidate remains quality-pending until
inspection. Counts use interval unions on canonical recording timelines across
reuploads and processing revisions. Gaps are never filled to inflate duration.

The main target counts approved intervals **at least 60 seconds**. The requested
strictly-over-60/120/300-second metrics select actual uninterrupted qualifying
intervals, then deduplicate coverage. They are nested, not additive. Approved
sub-60-second footage is reported separately. Speaker IDs and per-speaker hours
come from reviewed records; source/channel counts are not speaker counts.

## Hugging Face storage

The requested `garyzgao/facetime-video` URL currently redirects to
`cherthq/facetime-video`. The API resolves its canonical repository ID before
upload. Existing dataset visibility and existing cards are preserved.

```bash
hf auth login
python -m data_fetching.discovery publish --repo-id cherthq/facetime-video --dry-run
python -m data_fetching.discovery publish --repo-id cherthq/facetime-video
```

Use a locally configured token with dataset write access; never commit or send
tokens in chat. Uploads include only redistribution-approved full originals,
allowlisted attribution/interval metadata, and a dataset card when absent.
Original filenames are SHA-256 addressed; upload receipts record the commit.
Remote object hashes/sizes are checked before recording upload completion.
Pending footage is explicitly not training approved. No blanket license is
asserted for the entire collection. This uploader does not publish private review
notes, local filesystem paths, credentials, cookies or raw downloader metadata.

An authenticated browser can perform a supervised pilot upload when CLI
authentication is unavailable. Unattended uploads require local API credentials.

## Proposed recurring prompt (not installed)

Every four hours, resume this queue for approximately 30 minutes. Inspect search
results and select only plausible original recordings; record how selection was
performed. Respect the currently approved cumulative raw/storage budgets. Resume
acquisition and unscanned windows. Retain long continuous intervals and full
sources. Publish only sources with recorded redistribution approval. Keep
uninspected quality and unresolved permissions/deduplication pending. Never grant
human review on a scheduled run. Recompute deduplicated progress and stop
acquisition at 250 approved preferred-duration hours. Report meaningful progress,
completion, failure or required user action; remain quiet on unchanged state.
Provide a daily progress report at 9 a.m. America/New_York once enabled.

Before enabling this prompt, review the pilot and choose larger storage/raw
budgets, review ownership and any language/speaker concentration targets. Local
scheduled work requires the host and desktop app to stay running.
