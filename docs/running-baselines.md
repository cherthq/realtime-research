# 4:3 file-input avatar baselines

These commands explicitly download/install/run software on a Linux CUDA host.
They do not provision an instance or stop cloud billing. Use an already approved
GPU host and budget. Closing SSH does not terminate the instance.

The active matrix is three models x three images x six replies x two seeds = 108
direct 640x480 outputs. All three generate landscape 4:3 video; no portrait-output
stretching, padding or post-generation crop is used. The frozen input package is
unchanged; [the runtime geometry profile](../configs/baseline_geometry.json)
overrides the older native-first plan in its historical suite snapshot. This is
not a live voice agent, realtime transport test or fully scored quality benchmark.

## Recommended gated run

After preparing environments, checkpoints and frozen inputs below, explicitly run:

```bash
python -u -m scripts.run_4x3_baselines \
  --output runs/facetime-4x3 \
  --checkpoints /home/ubuntu/chert-model-cache-20261024 < /dev/null
```

This tests one greeting per model first, validates exact 640x480 square-pixel
output and the video hashes, then runs the full 108-output matrix under
`runs/facetime-4x3/matrix/`. Smoke outputs are extra, not counted among the 108.
Failures stop expansion. Use a tmux session and retain its logs. A partially
completed older portrait matrix must remain separate and is not resumed here.

## Inputs and environments

Transfer the frozen `data/eval/facetime-zero-shot-v1/` package separately; its
media are intentionally absent from Git. Verify input hashes before generation.
See the [input protocol](zero-shot-baseline.md). Keep validation assets out of
training and preserve their original WAVs/images.

Run each model in an isolated Python 3.10 environment. The setup scripts install
dependencies only when explicitly invoked. Default environments are under
`/home/ubuntu/envs/`; each setup script supports a `CHERT_<MODEL>_ENV` override.
The queue runner currently expects the default environment layout.

| Model | Setup | Native condition in this first runner |
| --- | --- | --- |
| FlashTalk | `bash infra/setup-flashtalk.sh` | BF16, direct 640x480/25-FPS, four sampling steps, chunked embeddings from a preloaded file |
| Live Avatar | `bash infra/setup-liveavatar.sh` | Direct 640x480, BF16 conversion, CPU T5/model/KV-cache offload, released LoRA, four Euler steps |
| LiveAct | `bash infra/setup-liveact.sh` | BF16 weights, FP8 KV cache, CPU T5, direct 640x480/24-FPS single-GPU path; no FP8 GEMM or vLLM |

The pinned vendor trees remain unchanged. FlashTalk's runtime adapter changes both
its inference dictionary and its preparation module's target shape. Live Avatar's
entry wrapper registers 640x480 and replaces automatic area/64-pixel rounding
with an exact generation size (both dimensions remain aligned to the model's
16-pixel VAE/patch grid). LiveAct receives `--size 640*480`. This is an explicit
custom geometry experiment, not an upstream-recommended configuration claim.
Only dimensions are matched: native output FPS and precision choices still differ.
On an 80GB H100, Live Avatar exhausted VRAM during its first non-quantized BF16 4:3
forward with all four upstream step caches resident. The active runner explicitly
enables upstream CPU KV-cache offloading rather than reducing dimensions or
silently switching to quantized weights. Cache transfers affect speed; preserve
the failed attempt and compare this as an offloaded configuration.
Its conditioning cache also hardcodes 2,800 tokens, while the 640x480 prefill
requires 2,980. The runtime entry installs a capacity-only guard before upstream
attention: grow K/V storage to the actual prefill length, preserve existing
values/device/dtype and retain the original attention calculation. This adapter
is hash-bound with the run; it does not truncate tokens or modify vendor weights.
If the launcher times out only after upstream reports finished generation/mux,
`scripts.recover_baseline_smoke` can separately validate the saved output's
geometry, audio, duration, full decode and hash. The original failed result is
retained. Such a recovered file may establish a functional smoke check only:
it is not a successful timed measurement and never counts toward the 108 outputs.

Setup is not proof that inference works. LiveAct's LightX2V dependency is pinned
separately. Its import path also imports unrelated framework runners, requiring
extra packages such as pyzmq and matplotlib. LiveAct needs the system SoX shared
library. The script installs SoX through passwordless sudo when absent; otherwise
ask the machine administrator to install it.

The upstream Decord 0.6.0 py3 wheel embeds a cp36 WHEEL tag. FlashTalk setup
reports that exact compatibility warning and verifies import; it does not edit
package metadata or suppress other dependency conflicts. Preserve all environment
freeze files and verify the decoder on the target host before using video inputs.

The FlashTalk setup pins torch/torchvision/xformers to its upstream CUDA 12.8
versions. Other pins are compatibility choices recorded in the run, not a claim
that all upstream projects support identical package versions or precisions.

## Checkpoints

Create a separate lightweight downloader environment with
`huggingface_hub==0.34.4`, then explicitly run:

```bash
python scripts/download_baseline_weights.py flashtalk --root checkpoints
python scripts/download_baseline_weights.py liveavatar --root checkpoints
python scripts/download_baseline_weights.py liveact --root checkpoints
```

Exact HF revisions are in that script and copied into per-model revision records.
Downloads retain available upstream license/readme files and skip demo assets and
the unused fairseq checkpoint. Weight access does not imply training support or
permission for every commercial deployment; audit each component separately.

Keep canonical checkpoints/results on persistent storage. For repeated cold
loads, `bash infra/stage-checkpoints.sh` copies the two non-FlashTalk models to
local SSD without removing persistent originals. It creates a new cache directory
and a READY marker only on complete success. Set `CHERT_MODEL_CACHE` for another
new cache location; it refuses to merge into an existing partial/stale cache.

## Run smoke tests first

From the repository root, using the appropriate model environment:

```bash
python scripts/run_flashtalk_baseline.py \
  --input-id ref01__s01_greeting --seed 0 \
  --output runs/flashtalk/smoke-s01 < /dev/null

python -m scripts.run_native_baseline liveavatar \
  --python /home/ubuntu/envs/liveavatar/bin/python \
  --input-id ref01__s01_greeting --seed 0 \
  --output runs/liveavatar/smoke-s01 < /dev/null
```

Repeat with s05 to exercise pause/resume; LiveAct uses the same native runner's
`liveact` model ID and its own interpreter. Existing output directories are never
overwritten. Each Live Avatar/LiveAct request has a 30-minute limit, preserves failures and
checks that a decoded audio/video file exists with a duration close to its master.
FlashTalk retains its separate 20-minute per-request limit. The offloaded Live
Avatar path needs a larger allowance for the longest replies; timeouts are not
realtime performance targets.

If launched detached, always redirect stdin from `/dev/null`. Otherwise FFmpeg's
interactive terminal input can suspend the background process group. Prefer a
persistent tmux session with logs; avoid treating an active process as success.

## Lower-level matrix commands and resume

Using the FlashTalk environment, from the repository root:

```bash
python -u -m scripts.run_flashtalk_suite \
  --output runs/manual-4x3-matrix/flashtalk < /dev/null
```

That matrix reuses loaded weights but calls upstream preparation for each input,
resetting its reference/prompt, seeded noise generator and motion state. Before
the full set, it performs an A-B-A decoded-frame hash check across two identities.
The repeated control is extra and not counted among the 36 matrix outputs.
Failure stops expansion rather than silently carrying session state forward.

`run_remaining_matrix` waits for a tmux session named `chert-flashtalk-full` to
finish all 36 cases, then runs Live Avatar and LiveAct sequentially. Launch it
from the repository root in a separate session only after arranging that exact
FlashTalk session/output layout and a complete local checkpoint cache:

```bash
python -u -m scripts.run_remaining_matrix \
  --output runs/manual-4x3-matrix \
  --checkpoints /home/ubuntu/chert-model-cache-20261024 < /dev/null
```

The other two models run isolated cold processes per case. Their first greeting
and pause cases come first; any inference/output failure stops their queue. The
queue does not retry with a different precision, shorter audio or resolution and
pretend it is the same experiment.

Bindings capture input/config/checkpoint/worker/environment identities. A resume
skips only successful, hash-matching outputs under the same binding. Failed or
partial attempts are retained; diagnose them and use a separate attempt rather
than deleting evidence or silently mixing changed configurations.

## What the measurements establish

FlashTalk records first new decoded CPU-chunk availability, unique new frames,
generation-loop time, padded versus useful frames, encoding time, CUDA allocator
peaks and sampled process GPU memory. Distinguish the first cold run from reused
model requests. Generation-loop time includes audio loading/embedding, CPU
transfers and any cold compilation; native playback FPS is not generation FPS.

Live Avatar/LiveAct currently record full-process time, sampled device memory,
commands, checkpoint revisions, environment and output checks. First-frame latency
and model-only generation FPS are unavailable for those file-only wrappers, not
zero. Their process time includes loading, compilation, generation and muxing and
cannot be compared directly with FlashTalk's narrower generation-loop time.

All wrappers use preloaded files. No future-audio isolation, delivered call FPS,
network latency, barge-in, calibrated lip-sync score or appearance score is
established by these runs. Quality scoring and wall-clock live replay remain
separate follow-up work. Different native precisions/FPS are recorded; framing is
forced to the shared geometry profile and validated, not hidden by stretching
outputs or re-synthesizing the audio.

Results belong under ignored `runs/`. Save per-case JSON, logs and videos, as
well as global `matrix-progress.json` and FlashTalk's `progress.json`. A complete
matrix means 108 validated output files, not 108 automatically good avatars.

Audit the current files at any time without running inference:

```bash
python -m scripts.summarize_baseline --output runs/facetime-4x3/matrix
```

This writes `summary.json` and `outputs.csv`, including every expected case and
explicit missing/failure states. Successful outputs are rechecked against their
recorded SHA256. Missing measurements remain blank/null, not zero. The remaining
matrix queue also writes this audit when it finishes or stops on a failure.
