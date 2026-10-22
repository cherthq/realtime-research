# Review, evaluation and optional packaging

These tools inspect preprocessing behaviour and produce approved exports. They
do not run automatically during extraction or grant source permissions.

## Review and exclusions

```bash
python -m data_fetching --root /storage/facetime gallery --output /storage/facetime/results/round1
python -m data_fetching --root /storage/facetime review-template --output /storage/facetime/reviews/round1.jsonl
```

Use new output paths. Review complete footage with audio for crop/torso coverage,
person consistency, gaze, appearance, speaker match and lip-sync. The gallery
displays evidence, not a semantic artifact detector.

Record source approval, license/evidence, reviewer, ISO UTC review time,
speaker/conversation groups, split and per-check decisions. Non-passing checks
require documented human decisions/evidence; invalid renders cannot be overridden.
Set `include_short` / `include_listening` only for intentional inclusion. Reviews
bind to exact QC/artifact hashes.

Confirmed bad spans can be excluded with JSONL:

```json
{"source_id":"recording_01","start":42,"end":48,"reason":"B-roll confirmed during review","evidence":["review-log-123"]}
```

Pass `--exclusions FILE` to `process --profile quality-long` with the usual
manifest/budgets. Times are relative to the source's first video frame. Remaining
intervals are reassessed; previous evidence is not erased.

## Duplicate proposals

```bash
python -m data_fetching --root /storage/facetime duplicates \
  --output /storage/facetime/reports/duplicate-proposals.json
```

Temporal hashes propose possible reuploads; they do not prove identity or find
every duplicate. Confirm matches and assign `duplicate_content_group` in reviews.
Export allows one canonical original per reviewed duplicate group, prevents
recorded source/speaker/conversation/duplicate groups from crossing splits, and
rejects duplicate/overlapping approved intervals from the same original.

## Labelled evaluation

Create JSONL labels from full interval review, bound to the original SHA-256:

```json
{"source_id":"recording_01","source_sha256":"ACTUAL_ORIGINAL_SHA256","start":10,"end":25,"label":"usable","partition":"test","group_id":"reviewed-speaker-source-group","reviewer":"reviewer-id","evidence":["review-log-123"]}
```

Labels are `usable`, `unusable` or `uncertain`; partitions are `calibration` or
`test`. Intervals cannot overlap. A source, original or reviewed group cannot
occur in both partitions. Use a consistent rubric and disjoint test data.

```bash
python -m data_fetching --root /storage/facetime benchmark --labels labels.jsonl --stage candidates
python -m data_fetching --root /storage/facetime benchmark --labels labels.jsonl --stage eligible
python -m data_fetching --root /storage/facetime benchmark --labels labels.jsonl --stage approved --reviews reviews.jsonl
```

Reports use retained duration, merge overlapping revisions, and separate the
partitions. Candidate retention precision is not approved-data precision.
Unlabelled/uncertain footage is not automatically clean. Independently inspect
retained and rejected footage for false acceptance/rejection, missed cuts,
framing defects, review effort and compute/storage cost per approved hour.

## Approved export and local packaging

Follow the [main export workflow](../README.md#5-export-approved-samples).
`report --reviews FILE` separates unique approved hours from raw candidate sums.
Changed processing revisions require revalidation of bound approvals.

```bash
python -m data_fetching --root /storage/facetime package-hub \
  --reviews /storage/facetime/reviews/round1.jsonl \
  --output /storage/facetime/packages/round1
```

Packaging requires v2 reviews with separate `redistribution_approved: true` and
a new destination. It copies approved videos, split metadata and companion
audio/reference/QC files; `COMPLETE.json` marks successful completion. No upload
or login occurs. Audit private evidence references and metadata before sharing.

To verify with Hugging Face Datasets, install `requirements-export.txt` in a
separate environment, then load locally:

```python
from datasets import load_dataset
dataset = load_dataset("videofolder", data_dir="/storage/facetime/packages/round1")
```

Check counts, actual splits and companion paths. Packaging does not establish
permission to redistribute or train on the source footage.
