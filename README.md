# Chert realtime avatar research

Minimal workspace for comparing image + audio driven avatar models, selecting a
quality baseline, then investigating fine-tuning and distillation.

## Start here

Local checkout: `D:\realtime-research` (WSL: `/mnt/d/realtime-research`).
The orchestration script uses only Python's standard library (Python 3.10+).

```bash
python scripts/workspace.py list
python scripts/workspace.py sources all --dry-run
python scripts/workspace.py sources all
python scripts/workspace.py doctor
```

`sources` clones official upstream repositories into ignored `vendor/` directories
at the commits in `configs/models.json`. Existing checkouts are verified, never
reset or automatically updated. Model downloads and installs are explicit.

## Layout

- `configs/models.json`: source commits, weight locations, upstream setup links.
- `scripts/workspace.py`: list models, fetch pinned sources, check prerequisites.
- `docs/experiments.md`: baseline protocol and data/experiment conventions.
- `infra/lambda.md`: GPU host setup and environment boundaries.
- `infra/ssh_config.example`: SSH template; keep your key outside this repository.
- `vendor/`, `data/`, `checkpoints/`, `runs/`: local artifacts, excluded from Git.

## Model set

| ID | Model | Initial purpose |
| --- | --- | --- |
| `liveavatar` | Live Avatar (14B family) | Distilled streaming baseline |
| `flashtalk` | SoulX-FlashTalk-14B | Distilled streaming/gesture comparison |
| `hunyuan_avatar` | HunyuanVideo-Avatar (~13B backbone) | Offline quality comparison |

These are avatar renderers: voice understanding, dialogue, TTS and interruption
handling belong to the later conversation pipeline. Open weights do not imply
released training code. Verify training support and checkpoint compatibility
before scheduling fine-tuning. Live Avatar and FlashTalk are already distilled;
fine-tuning them may require retaining their few-step/streaming objectives.
Hunyuan has its own community license; do not assume Apache terms for all models.

Keep one Python/CUDA environment per upstream repository. Follow its pinned
README for installation; there is deliberately no combined ML requirements file.
Local 4 GB VRAM is for orchestration checks. GPU inference runs on Lambda.

See [Lambda setup](infra/lambda.md) and [experiment protocol](docs/experiments.md).
