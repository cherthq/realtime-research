# Experimental listening and speech-onset profile

`AVATARFOREVER_LISTENING_PROFILE=onset_boost` enables the user's selected
quiet-listening profile. The repository default is `off`; this is an opt-in demo
experiment, not a reliable closed-mouth guarantee. Quality varies across portraits and seeds; inspect both silence and speech onset.

## Behavior

- A silent native chunk uses the fixed quiet-listening description.
- The first two consecutive audible chunks use a stronger speaking cue:
  `lerp(quiet.float(), speaking.float(), 1.5)`, cast back to the model dtype.
- Further audible chunks use the normal speaking description. A silent chunk
  ends the onset window. A new response epoch also re-arms it after interruption.
- Cue selection reads valid PCM16 samples only, with absolute magnitude above
  two. Padding and lookahead cannot trigger speaking. This threshold is for clean
  synthesized audio and exactly matches the original experiment, not general VAD.
- Changing cues preserves the original reference, sink chunk, recent visual
  history, audio history and model timeline. No reset or image substitution is
  inserted at a sentence boundary. Session cleanup resets the cue counter.
- The counter advances only after successful generation; a cancelled preview
  does not consume the two-chunk onset window.

The model's native chunk duration is approximately 1–1.28 seconds. This control
is not frame-accurate: a chunk can contain both silence and speech, and generated
motion can persist across its boundary. It does not fix long-call visual drift.

## Configuration and cache safety

Add to the private `realtime/.env` before starting the demo host:

```dotenv
AVATARFOREVER_LISTENING_PROFILE=onset_boost
```

Restart only when no call is active, then warm the GPU before opening a call.
To return to the previous fixed-prompt behavior, set the value to `off` and
restart/warm again. Do not restart the service during someone else's call.

Quiet and speaking embeddings are encoded once while Gemma is resident; the
onset embedding is derived from them. Gemma is then offloaded. There is no text
encoding request for each sentence or chunk. Prompt projections are cached by
**both cue and denoising step**. All three cue banks and bounded history shapes
are warmed before readiness, including when compilation is enabled.

With fixed-prompt caching, this uses approximately three times the projection
cache memory of one prompt. It also extends startup warmup. Check peak VRAM on
the target GPU; do not silently disable caching/compilation or change precision
to make a configuration fit. Changing prompts, weights, dtype or sigmas requires
rebuilding the embeddings and projection caches.

`listening_control` in worker information reports the selected profile and its
experimental status. Per-chunk worker metrics also report the last committed
cue and speech-chunk count. The model-pool health snapshot contains the startup
configuration, not a live cue counter.

## Validation

Unit checks cover cue transitions, clean-PCM thresholds, signed integer overflow,
padding/lookahead exclusion, interruption epochs, cancellation preview, session
reset, separate projection banks, complete cache capture and compiled projection
switching. GPU replay and browser validation results are recorded below when
completed. Private recordings, portraits, voice assignments and access keys are
not part of this document or the public repository.

The implementation checks passed on 2026-11-01: 57 application/worker tests
(one optional integration test skipped), and 14 controller/cache tests in the
GPU environment. Ruff also passed for the changed Python modules. These checks
verify wiring and state handling, not perceptual mouth closure or lip-sync.

A matched 41.96-second replay then exercised the production patch on one H100
PCIe with BF16, four steps, 512×384 native output, 48 compiled transformer blocks
and all three prompt-cache banks. It used the original private portrait, seed
and prerecorded PCM from the onset-boost experiment, preserving history.

- Load and warmup: 296.7 seconds. Prompt-cache tensors: 13.5 GiB; peak PyTorch
  allocation: 68.54 GiB (not total device usage).
- After the first two chunks, median generation time was 0.933 seconds and the
  maximum was 0.991 seconds per 1.28 seconds of native video. This is generation
  throughput, not end-to-end conversational latency.
- Every cue transition matched the expected schedule. The generated contact
  sheet showed quiet lips over most listening interiors and articulation during
  both speech intervals, without an inserted portrait reset.
- With the same provisional FaceMesh threshold as the earlier experiment,
  89.2% of 674 measured silent frames had closed lips. Opening was approximately
  90 ms early for the first reply and 650 ms early for the second. Sustained
  closure occurred at the first audio ending and 2.6 seconds after the second.
- SyncNet confidence was 7.12 / 7.56, with a −40 ms estimated offset for both
  seven-second speech intervals. These are diagnostics, not accuracy percentages.

The compiled/cached path is not pixel-identical to the earlier eager experiment,
which measured 94.2% silent closure for this portrait/seed. This replay confirms
working cues and adequate generation throughput, **not perfect silent-mouth
control**, generalization or long-call stability. Native generation remains
25 FPS for audio alignment; WebRTC publication stays capped at 24 FPS.

Private replay media, measurements and deployment backups are ignored under
`.cache/listening-deploy-20261101/`. The deployment patch contains only this
feature; unrelated local drift research was not copied into the running worker.

The subsequent 92-second browser check passed two completed spoken replies and
one explicitly confirmed speaking interruption, using a synthetic microphone,
the private portrait and its existing reference voice over LiveKit Cloud.
The worker reported `onset_boost`, compiled blocks and prompt caching; publication
remained capped at 24 FPS. It delivered 1,976 observed browser frames with zero
portrait fallbacks, no provider error and no speech-buffer underruns. Mean model
generation rate was 33.34 FPS. No pending speech remained at the final snapshot;
the decoded-batch queue contained one batch.

Playback was not hold-free: metrics recorded 26 held frames, with a maximum
consecutive hold of 1.04 seconds. Server cancellation acknowledgement was 1.63 ms,
which is **not** a physical-speaker or visible-mouth stopping-time measurement.
The test does not establish universally smooth playback or exact lip-sync.

The demo was left running and warm with this opt-in profile, no active session,
continuous motion enabled, ElevenLabs v4 Turbo and the existing 600-second call
limit. The user selected this experimental trial despite the documented closure
failures. No long-call qualification or new training was performed.
