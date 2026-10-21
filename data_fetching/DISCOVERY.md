# Long-footage discovery handoff

Implementation update: see [QUEUE.md](QUEUE.md) for the opt-in persistent queue,
sampled long-interval screening, deduplicated accounting, and Hugging Face
storage. This handoff remains the collection specification. No schedule is
installed automatically; full quality and source-permission review remain gates.

The collection target is 200-300 hours of **usable, unique, reviewed footage**,
not 200-300 downloaded hours. This is a collection target, not an established
minimum training requirement. Measure rejection rates on a small batch before
estimating the raw hours required.

Keep full original recordings. Prefer clean, continuous 60-300 second segments,
retaining longer uninterrupted sequences when available. Short windows can be
sampled later according to each training implementation's context limits. Never
join separate shots or duplicate footage to satisfy a duration target.

The current default config intentionally makes 5-15 second candidates for smoke
tests. `min_seconds` and `max_seconds` in a separate JSON config control segment
length; changing the discovery target does not silently change processing.
Longer processing also needs explicit `--scan-seconds` and `--max-clips` budgets.
The source `start`/`end` bounds still apply: the pilot manifest covers only 45
seconds per source. Sampled face/pose checks are not a full artifact audit.

## Scheduler prompt

Find and rank original, real-person, camera-facing videos toward a cumulative
200-300 hour reviewed corpus. Resume across runs using persistent records; do
not stop permanently after a fixed count of links or invent progress when a run
budget is exhausted.

Prioritize longer source recordings (approximately 10-60 minutes, or longer when
suitable): webcam interviews, personal updates, educational explainers and Q&As.
Prefer stable framing, clear speech, one visible speaker and visible shoulders
and upper chest or wider torso. Seek speaker, language, environment and appearance
diversity; do not collect most of the corpus from one person.

Identify continuous clean intervals preferably 60-300 seconds. Do not concatenate
shots, repeat footage, stretch time, inpaint overlays, restore faces or synthesize
missing body regions to make an interval pass. Retain shorter good intervals in
a separate duration bucket.

Flag subtitles, logos, lower-thirds, split screens, picture-in-picture, frequent
cuts, heavy compression, beauty filters, virtual-background defects, dubbing,
music over speech and mismatched speakers. Record whether checks used actual
video/audio inspection, sampled frames, metadata only or a thumbnail. Unperformed
checks must be labelled unverified. Models and thumbnails cannot guarantee clean
footage throughout a long interval.

Preserve the original uploader, source URL and permission/license evidence. A CC
label is a discovery clue, not a blanket training approval. Do not bypass access
restrictions. Prefer creator-provided original files through a permitted route
for production acquisition.

Append discovery records to ignored `data/discovery/candidates.jsonl`, separate
from the downloader/training manifests. Include video ID, URL, channel, title,
duration, suggested clean intervals, framing category, observed quality issues,
inspection method, permission evidence and review status. Use framing categories
`head_upper_chest` and `wider_torso_gestures`. New records start pending review.

Deduplicate reuploads and overlapping intervals. Report unique raw hours found,
hours acquired, candidate clean hours, reviewed usable hours, duration buckets
above 1/2/5 minutes, distinct speakers and rejection reasons. Only inspected and
approved unique intervals count toward the usable-hour target. Keep speaker and
recording groups out of multiple train/validation/test splits.

Summarize the strongest candidates, remaining uncertainties and progress since
the previous run. Never invent inspection results, permissions or approvals.

## Implementation boundary

This document is a handoff prompt; the opt-in queue implementation is described
in QUEUE.md and does not run a scheduler on its own.
Discovery records must be reviewed and converted to the source schema in the
parent README before invoking `fetch`. Automatic OCR, watermark detection,
active-speaker scoring and full artifact quality gates are not implemented.
The queue adds perceptual duplicate proposals and canonical timeline mappings,
but unresolved matches require review and are excluded from confirmed totals.
