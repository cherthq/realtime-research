# Optional quality workers

Workers add speech, text, active-speaker and lip-sync evidence after visual
selection. Missing or uncalibrated checks remain unresolved for review. No
checkpoints are included or silently downloaded during processing.

## Environments

Use Python 3.11 and keep these separate from the main MediaPipe environment.
From the repository root on Linux:

```bash
python3.11 -m venv .venv-qc-vad
.venv-qc-vad/bin/python -m pip install -r data_fetching/requirements-vad-worker.txt
python3.11 -m venv .venv-qc-ocr
.venv-qc-ocr/bin/python -m pip install -r data_fetching/requirements-ocr-worker.txt
```

For optional TalkNet/SyncNet adapters:

```bash
python3.11 -m venv .venv-qc-audio
.venv-qc-audio/bin/python -m pip install -r data_fetching/requirements-audio-worker.txt
```

On Windows, create each environment with `py -3.11 -m venv` and use its
`Scripts/python.exe`. Do not mix Torch/Paddle dependencies with MediaPipe.
The supplied OCR wheel targets CPU; provision a compatible GPU runtime separately.

## Local checkpoints

Obtain trusted checkpoints explicitly after reviewing their exact code/weight
terms. Keep weights outside Git and clean upstream source checkouts. Do not run
upstream demo entrypoints as setup; adapters import inference components only.

| Backend | Required artifacts and interface |
| --- | --- |
| VAD | Silero v5-style ONNX: `input`, `state`, `sr`; 512 samples at 16 kHz, 64-sample context, state `[2,1,128]` |
| OCR | Complete local `PP-OCRv5_mobile_det` inference directory; no recognition model |
| TalkNet | Compatible TalkSet checkpoint and clean pinned checkout; strict `model.*` / `lossAV.*` state dictionaries |
| SyncNet | Compatible checkpoint and clean pinned checkout with `SyncNetInstance(device=...)` and offset/confidence/distances |

Upstream projects: [Silero](https://github.com/snakers4/silero-vad),
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR),
[TalkNet](https://github.com/TaoRuijie/TalkNet-ASD),
[SyncNet](https://github.com/joonson/syncnet_python).
These links do not establish permission for any specific checkpoint/use.

OCR uses PaddleX's direct text detector, not document/LLM extras. Install a local
Arial, DejaVu or Liberation font (`fonts-dejavu-core` on minimal Linux), or set an
absolute `font_file` in its registry entry. Fonts must not download at inference.

## Registry

Copy [workers.example.json](../examples/workers.example.json) to the ignored
dataset's `inputs/workers.json`. Example backends are disabled deliberately.

1. Set interpreter, weight/model-directory and optional upstream-repository paths.
   Absolute paths are recommended. Interpreter/model/repository and asset paths
   otherwise resolve against the registry directory; use an absolute custom font.
2. Inventory local checkpoint files with the read-only helper:

   ```bash
   python -m data_fetching.asset_manifest /path/to/checkpoint-or-model-directory
   ```

3. Copy every path/SHA-256 into `assets`. Weights must be listed, and a model
   directory must match its inventory exactly.
4. Pin TalkNet/SyncNet `revision` to the checkout commit and keep it clean,
   including untracked files. Example revisions are source API pins, not evidence
   that checkpoint inference has been validated. Keep weights outside the checkout.
5. After terms review, record `license_evidence`, set `terms_reviewed: true`, then
   `enabled: true`. This is an operator decision, not automatic clearance.
6. Keep calibration flags false/null until tested on labelled real footage.
   `timeout_seconds` must be an integer from 1 to 86400.

```bash
python -m data_fetching --root /storage/facetime doctor \
  --profile quality-long --workers /storage/facetime/inputs/workers.json
```

Doctor validates paths, hashes, dependencies and fields, not inference accuracy.
It returns nonzero when any expected backend is unconfigured. Diagnostic
processing can still run; missing checks remain `unverified`, never a pass.

## Execution and calibration

Pass `--workers FILE` to `process --profile quality-long`. OCR/VAD reuse isolated
workers by default; `--worker-mode oneshot` releases them after each clip.
TalkNet/SyncNet remain one-shot. Their registry `device` and runtime choices are
separate from `--detector-device` for the main face/pose models.

Each request checks input/model hashes and response identity/coverage. Python
network connections are blocked during inference, not by an OS-level sandbox.
Failed or timed-out workers cannot clear checks as passed. VAD state resets
for every clip even when its model stays loaded.

To exercise an already generated request, choose a new response filename:

```bash
/path/to/worker/python -m data_fetching.quality_worker \
  --request /storage/facetime/clips/CLIP_ID/ocr-request.json \
  --response /storage/facetime/reports/ocr-smoke-response.json
```

Real local VAD/OCR execution was tested. Exact TalkNet/SyncNet checkpoints and
calibration remain unvalidated. Use known-good speech, offscreen/mismatched voices
and injected positive/negative delays. AV models use analysis-only 25-FPS face
proxies, not altered training media. Record calibration evidence before setting
`score_validated` or `sync_sign_validated`; establish SyncNet's offset convention
for the chosen preprocessing. A successful import does not validate model quality.

To retry a completed candidate after a transient worker error, change a
`retry_token` string in that backend's registry entry. This produces an auditable
QC revision and can reuse unchanged scans/renders. Do not edit completed responses
or candidate metadata; artifact hashes bind the review.
