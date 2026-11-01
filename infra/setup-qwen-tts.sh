#!/usr/bin/env bash
# Separate environment: Qwen's Transformers version must not alter AvatarForever.
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv --python 3.11 --seed .venv-qwen-tts
uv pip install --python .venv-qwen-tts/bin/python torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-qwen-tts/bin/python \
  'faster-qwen3-tts @ git+https://github.com/andimarafioti/faster-qwen3-tts@b35fdc7d71eb3e4d19e63f4b4a330a1aef31d682' \
  'transformers==5.15.1' 'qwen-tts-hf==0.1.1.post1' 'fastapi>=0.115,<1' 'uvicorn>=0.34,<1'
.venv-qwen-tts/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', revision='0c0e3051f131929182e2c023b9537f8b1c68adfe', local_dir='checkpoints/qwen-tts')"
mkdir -p runs/qwen-tts/setup
uv pip freeze --python .venv-qwen-tts/bin/python > runs/qwen-tts/setup/requirements.lock.txt
