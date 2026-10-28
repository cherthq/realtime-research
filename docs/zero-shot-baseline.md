# Scripted, objective-only avatar baseline

Protocol date: 2026-10-24. The [six-scenario prompt pack](../configs/zero_shot_suite.json)
and [input preparation script](../scripts/prepare_zero_shot_assets.py) are supplied.
Three synthetic portraits and six fixed WAVs are prepared in the local, ignored
input folder. Human listening approval is pending. File-input runners, basic
timing/memory collection and output checks are now supplied; see
[running baselines](running-baselines.md). Wall-clock-paced streaming, cancellation
and calibrated automatic lip-sync/appearance scoring remain pending. Running the
file-input matrix is not a full FaceTime benchmark or a complete quality evaluation.
The active run now requires direct 640x480 output on all three models, as defined
by [the runtime geometry profile](../configs/baseline_geometry.json). This
supersedes the earlier native-first plan without changing the frozen media.

## Inputs and setup

No live user interaction or ground-truth target video is required. This first
baseline evaluates the avatar renderer, not an LLM's answers:

```text
Fixed reply text -> synthesize once -> freeze the WAVs
Same images + same WAVs + same visual prompt -> each avatar model
Generated frames + timing/memory events -> objective results
```

The JSON separates `user_prompt` (scenario context), `segments` (exact spoken
reply and inserted silences), and `avatar_prompt` (visual scene). Conversation
questions are not instructions for an avatar model to generate a spoken answer.
Do not call an LLM or regenerate TTS separately for each avatar model: varying
reply wording, voices, prosody and durations would confound the comparison.

Prepare these assets once:

1. Three permission-cleared or synthetic, camera-facing upper-body images, with
   complete heads/shoulders and neutral mouth poses. Retain high-resolution
   originals, provenance/permission evidence, dimensions and SHA256 hashes.
2. One fixed English TTS voice/provider. The pack pins Kokoro v1.0, `af_heart`,
   kokoro-onnx 0.6.1 and SHA256 hashes for its ONNX model and voices file. The
   model is Apache-2.0 and wrapper MIT; these are upstream component licenses,
   not a blanket license for generated outputs. Use neutral style and speed 1.0,
   except s04 requests 1.25. If the
   provider cannot apply that setting, stop or version the suite explicitly; do
   not silently substitute another condition. Report actual duration/speaking rate.
3. Six master WAVs: mono, 48kHz PCM16. Render the exact speech segments once,
   insert zero-valued PCM for pauses, and record their exact sample ranges.
   Add 0.25-second leading/trailing silence. Never synthesize the pause instruction
   as speech or stretch clips to an arbitrary ten-second duration.
4. Preserve/hash the masters. Produce deterministic model-specific resamples only
   where needed, logging method and both hashes while preserving timing.

Store assets under ignored `data/eval/facetime-zero-shot-v1/`, with `images/`,
`audio/`, `permissions/` and an input manifest. Record `id`, `scenario_id`,
`reference_image`, `audio`, `prompt`, `split`, `source`, `license`, input hashes,
preprocessing, TTS provenance and silence intervals. Paths are relative to the
input root. A target `video` is optional. Missing assets or rights mean not ready.
Use `val` for this model-selection suite; keep its assets out of training.
Public-pretraining overlap may remain unknown.

### Prepared input package and local checks

The local package is `data/eval/facetime-zero-shot-v1/`:

- `images/ref01.png`, `ref02.png`, `ref03.png`: generated 4:3 upper-body originals,
  copied without resizing/cropping. Full prompts are in the public
  [portrait specification](../configs/zero_shot_portraits.json).
- `audio/s01_greeting.wav` through `audio/s06_short_turns.wav`: six shared masters,
  not eighteen separate TTS generations. One voice is deliberately reused across
  all portraits, so this does not test voice/gender or language diversity.
- `assets.json`: dimensions, hashes, durations, TTS settings/dependencies and
  half-open sample timelines. Natural TTS pauses within speech are not forced
  alignments; the explicit inserted zero ranges are exact.
- `inputs.jsonl`: all eighteen image/scenario pairs, split `val`.
- `permissions/provenance.md`: synthetic origin and upstream rights notes.

Media, TTS weights and virtual environments are ignored by Git. A fresh clone
will not contain these assets. Transfer the frozen input directory separately;
do not re-synthesize independently for each model or expect bit-identical audio
from a different runtime. Review all six WAVs before the full GPU suite.

Use a separate Python 3.11 environment for input preparation/validation. It is
not an avatar model's Python/Torch environment:

```bash
python -m venv .venv-eval
# Activate .venv-eval for your shell, then:
python -m pip install -r infra/requirements-eval-inputs.txt
python scripts/prepare_zero_shot_assets.py --validate-only
python -m unittest scripts.test_prepare_zero_shot_assets -v
```

Validation checks hashes/specification binding, 4:3 dimensions, matrix coverage,
PCM format, speech presence, no clipping, timeline continuity and exact pause
samples. It does not judge pronunciation, photorealism or human preference.

To prepare a new version rather than validate the supplied one, first generate
or supply the three images matching its portrait specification. Download the
two files from `audio_preparation.checkpoints` to `checkpoints/kokoro-v1.0/`.
Then run the script without `--validate-only`, with `--output` pointing to a new
versioned directory containing those images. Downloads are manual and their
hashes are verified before model loading. Existing audio is never overwritten;
if synthesis fails, retain the partial folder for diagnosis and use a new one.
Re-generated portraits are new inputs even with the same prompt: assign a new
package version and re-run all compared models against the same new inputs.

### What runs locally versus on Lambda

Build adapter interfaces, command/config validation, replay logic, result schemas,
failure handling and CPU/mocked tests locally. Pin each upstream revision and
keep model dependency environments isolated. Windows tests do not establish
Linux/CUDA compatibility; repeat smoke checks on the target GPU host.

Actual avatar-weight loading and inference run on the GPU host. Collect startup,
new-frame timestamps, throughput, CUDA peak memory and hardware metadata there,
in the same process/session producing the frames. Do not infer time-to-first-frame
from a completed MP4 or benchmark the local laptop as a substitute for Lambda.

After inference, offline file/duration/freeze checks and report assembly can run
locally. GPU-heavy lip-sync/embedding evaluators may run on the server afterward,
with pinned evaluators and controls. Keep their execution separate from timed
generation so evaluator contention does not distort renderer latency or memory.
The prepared package needs no running/paid Lambda instance. No cloud jobs,
avatar checkpoints or new model adapters were started by input preparation.

## Scenarios and run matrix

| ID | Scenario | Main checks |
| --- | --- | --- |
| s01 | Introduction | Startup and normal speech |
| s02 | Clear consonants | Mouth articulation and lip-sync |
| s03 | Appointment and numbers | Short phrases and lip-sync |
| s04 | Faster speech | Rapid articulation and throughput |
| s05 | Speech, two-second pause, speech | Silent-mouth motion and restart |
| s06 | Three replies separated by pauses | Repeated onsets and continuity |

Smoke test: one image, s01 and s05, seed 0, each model = six runs. Stop on setup,
OOM or output-validation failure. No paid run starts automatically.

Full active suite: three images x six scenarios x two seeds (0, 42) x three
models = 108 direct 640x480 outputs. Record actual dimensions, native FPS,
precision and prompts. The geometry override is explicit, not an unmodified
upstream reproduction. Gate expansion on one successful 4:3 greeting per model;
also inspect pause/resume cases. Unsupported geometry stops the run rather than
stretching/interpolating output to claim a match. Prior portrait outputs remain
separate and do not count toward this matrix. Output FPS is not matched yet.
Shared seeds aid repeatability, not equivalent noise across architectures.

Keep hardware-matched comparisons separate from upstream recommended multi-GPU
configurations. Label a 40GB A100 CPU-offloaded result as such; it is not an
equivalent reproduction of a multi-H100/H800 benchmark.

## Objective measurements

| Metric | Measurement | Boundary or limitation |
| --- | --- | --- |
| Runtime, generation FPS, RTF | Generation start to final new decoded frame; unique new frames / generation seconds; generation seconds / actual new-video duration | Report preprocessing, CPU transfer and file encoding separately; exclude overlapping context frames. Playback FPS is not generation speed. |
| Memory | Per-GPU peak allocated/reserved memory, sampled process GPU memory and host RAM | Include every GPU and the collector/sampling interval. |
| Cold/warm startup | Audio available to first usable newly generated decoded frame, not a displayed reference image | Separate loading/compilation. File-completion time cannot substitute for a missing first-frame hook. |
| Lip-sync | Speech-window offset in milliseconds, confidence, valid-window count and coverage | Pin/calibrate evaluator on aligned and deliberately shifted controls; silence and uncertain estimates are not valid measurements. |
| Silent-mouth motion | Normalized mouth opening/motion during annotated silence, with valid face-track coverage | Proxy, not a requirement for a perfectly closed/motionless mouth. Also check speech motion to avoid rewarding frozen video. |
| Appearance/temporal stability | Reference and within-video subject consistency, background consistency, imaging quality and motion smoothness | Proxies do not prove biometric identity, realistic gestures or human preference. Use dimensions supporting custom inputs. |
| Failures and cost | Setup/OOM/decode failures, truncated output, actual audio/video duration, allocated-GPU seconds and optional actual-rate cost | Preserve failed/missing runs and report GPU count; do not invent rates or suppress failures. |

Use a resident worker for warm repeated requests. Restarting a CLI per sample
does not measure warm service latency. For a startup p95 pilot, repeat s01 with
one image for 30 measured warm requests per model/profile, separately from the
two-seed suite. Record sample count and timing boundaries. Collect detailed CUDA
stage profiling separately so synchronization does not serialize streaming.
Input/config/checkpoint mismatches must invalidate resume.

No target video is needed for these raw measures. Do not report PSNR/SSIM against
an invented target, FID/FVD without a reference set, or THEval's full normalized
composite without ground-truth reference statistics. VBench's full leaderboard
score is not a product-quality score for this small custom suite.

## Scripted streaming, cancellation and stress tests

Replay the same audio in 20ms packets at wall-clock pace. Enforce no-future-audio
access at the worker input: no reading unseen samples or precomputing full-file
embeddings. Include chunk buffering/lookahead in startup and output lag. Timestamp
usable decoded frames at the coordinator; validate PTS ordering, bounded queues,
late frames and final partial chunks. Report preloaded-audio inference separately.

Inject cancellation five seconds into s01, discard queued output at the playback
boundary, and measure last stale video/audio where playback exists. This measures
cancellation, not user barge-in detection. Unsupported cancellation is marked
unsupported; without audio playback, audible-stop latency is unavailable.

For ten-minute stress tests, deterministically repeat s06 audio to exactly 600
seconds and label it repeated synthetic test audio. Keep one continuous model
session without resetting history between repetitions. Measure queue/memory
growth, throughput and appearance drift, not conversational understanding.

File-only adapters cannot establish delivered call FPS, transport freezes,
network latency, end-to-end conversational response time or interruption handling.
Missing metrics are unavailable, never zero or an automatic pass.

## Implementation checklist

1. Review the prepared portraits and listen to all six frozen WAVs. Run package
   validation; record approval without modifying frozen audio/specifications.
2. Register LiveAct and pin source/checkpoint/dependency revisions.
3. Build isolated adapters with incremental input/frame hooks where supported.
4. Add objective collectors, input validation and per-run JSON/CSV reporting.
5. Test future-audio isolation, pause boundaries, overlap removal, partial chunks,
   session isolation, bounded memory, failures and hash-bound resume.
6. Pass the three-model geometry gate and inspect pause/resume outputs; approve
   the budget before the 108-output 4:3 suite.
   Human ratings and LLM/TTS comparisons are not part of this first baseline.

Sources: [FlashTalk's released inference loop](https://github.com/Soul-AILab/SoulX-FlashTalk/blob/2ba2e1e801d59a9f526212a5aae5d374db8f2978/generate_video.py),
[VBench custom-video evaluation](https://github.com/Vchitect/VBench#new-evaluate-your-own-videos),
[THEval metrics and normalization](https://github.com/Newbyl/THEval),
[Kokoro model card](https://huggingface.co/hexgrad/Kokoro-82M),
[Kokoro ONNX wrapper](https://github.com/thewh1teagle/kokoro-onnx).
