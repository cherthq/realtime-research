# Experiment conventions

Sequence: unmodified zero-shot baselines -> small matched fine-tuning pilot ->
quality comparison -> select teacher -> investigate streaming/step distillation.
Live Avatar and FlashTalk already contain distillation; further training must
account for that. No model is assumed to support LoRA training out of the box.

## Dataset contract

See [the data pipeline guide](../data_fetching/README.md) for 4:3 landscape,
torso-preserving extraction, quality evidence, review and approved export.

Use JSONL manifests under `data/` with one sample per line. Required fields:
`id`, `reference_image`, `audio`, `video`, `split`, `source`, `license`.
Paths are relative to the dataset root. `split` is `train`, `val` or `test`.
Optional: `prompt`, `transcript`, `speaker_id`, `conversation_id`.

Those required fields describe paired training/real-video evaluation data. For
an image-and-audio-only generation baseline, `video` is optional. See the
[scripted objective-only baseline](zero-shot-baseline.md) and its fixed scenario
pack: no live participant, target video or newly recorded dataset is required.
Use its validation split for model selection, not as an untouched final test.

Keep original media and metadata separate from preprocessed model-specific files.
Split by identity and recording/conversation to prevent leakage. Hold out a fixed
evaluation set before filtering or fine-tuning; include FaceTime-like framing,
long speech, silence and expressive speech. Record audio/video alignment and
crop/resize settings when preprocessing is implemented.

## Runs

Store artifacts at `runs/<model>/<run-id>/`: generated videos, `metadata.json`,
`metrics.json`, and logs. Metadata records source commit, checkpoint ID/revision,
input IDs, prompts, seed, precision, resolution, FPS, steps, hardware, environment
and preprocessing settings. Keep generated data out of Git.

Evaluate the same images/audio across models. Report quality at each model's
recommended settings first, then compare latency/cost at a matched resolution and
duration. Track lip-sync, identity, motion/gesture quality, long-run drift, first
frame latency, end-to-end throughput and peak VRAM. The first scripted milestone
collects objective measures only; add blinded human comparison later. Automatic
scores are proxies, not a substitute for perceived quality, and no performance
claim is established until measured.

The voice dialogue system, streaming TTS, listening motion and interruption tests
come after the standalone avatar baseline is reproducible.
