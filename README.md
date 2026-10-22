# Chert realtime avatar research

Research tooling for image-and-audio-driven avatar models and review-first
preparation of talking-person video data.

The repository contains two independent workflows:

- **Data preparation:** extract fixed 4:3 upper-body clips, preserve audio timing,
  record quality evidence, review candidates and export approved samples.
- **Model research:** inspect and retrieve pinned upstream avatar-model sources
  for baseline comparisons, fine-tuning feasibility and distillation research.

No collection, checkpoint download, training or deployment starts automatically.
Datasets, model weights, generated media, credentials and private reviews are not
included in the checkout.

## Get started

```bash
git clone https://github.com/cherthq/realtime-research.git
cd realtime-research
```

For data preparation, follow the [data pipeline README](data_fetching/README.md).
It documents Python 3.11 setup, source manifests, processing limits, review and
export. See [performance and GPU setup](data_fetching/SPEED.md) before processing
long recordings or increasing concurrency.

For the model workspace, Python 3.10+ is sufficient for the standard-library
orchestration script:

```bash
python scripts/workspace.py list
python scripts/workspace.py sources all --dry-run
python scripts/workspace.py doctor
```

To explicitly retrieve registered upstream sources:

```bash
python scripts/workspace.py sources all
```

Sources are cloned into ignored `vendor/` directories at revisions recorded in
`configs/models.json`. Existing checkouts are verified, not reset or silently
updated. Checkpoint downloads and environment installation are separate steps.

## Repository layout

| Location | Purpose |
| --- | --- |
| `data_fetching/` | Preprocessing CLI, quality checks, examples and tests |
| `configs/models.json` | Registered model sources, revisions and weight references |
| `scripts/workspace.py` | Model workspace inspection and source retrieval |
| `docs/experiments.md` | Experiment protocol and dataset conventions |
| `infra/` | Remote-host setup and SSH configuration template |

Generated artifacts belong in ignored `data/`, `checkpoints/`, `runs/` and
`vendor/` directories, or on an explicitly configured external storage volume.
Keep private keys and service credentials outside tracked files.

## Registered model baselines

| Model ID | Model |
| --- | --- |
| `liveavatar` | Live Avatar |
| `flashtalk` | SoulX-FlashTalk-14B |
| `hunyuan_avatar` | HunyuanVideo-Avatar |

Registration does not establish training support, checkpoint compatibility or
hardware capacity. Use a separate environment for each upstream project and
follow its pinned documentation. These models render avatars; speech recognition,
dialogue, voice generation and interruption handling are separate components.

See the [experiment protocol](docs/experiments.md) and
[remote-host guide](infra/lambda.md) for the model research workflow.

## Validation and responsible use

```bash
python -m unittest discover -s data_fetching/tests -v
```

Install the data pipeline prerequisites first. Tests use synthetic local media
or mocks, with optional dependency checks; they do not download recordings or
model weights. See the pipeline README for the tested scope and limitations.

Preprocessing produces candidates, not automatic training approval. Source
permission, quality review and redistribution approval are separate decisions.
Upstream code, checkpoint and source-media terms must be checked independently;
inclusion in a source list does not grant permission for an intended use.
