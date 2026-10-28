#!/usr/bin/env bash
set -euo pipefail
workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
environment="${CHERT_LIVEAVATAR_ENV:-/home/ubuntu/envs/liveavatar}"
cd "$workspace"
python3 scripts/workspace.py sources liveavatar
if [[ ! -f "$environment/bin/python" ]]; then
    python3 -m venv "$environment"
fi
"$environment/bin/python" -m pip install pip==25.2 wheel 'setuptools<81'
"$environment/bin/python" -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
"$environment/bin/python" -m pip install -r infra/requirements-liveavatar.txt
MAX_JOBS=8 "$environment/bin/python" -m pip install flash-attn==2.8.3 --no-build-isolation
# Run GPU import/smoke checks separately; setup can overlap another model's run.
printf 'Live Avatar installation finished. Inference validation still required.\n'
