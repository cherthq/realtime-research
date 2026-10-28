#!/usr/bin/env bash
# Explicit Linux/GPU environment installation; never executes on import or clone.
set -euo pipefail
workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
environment="${CHERT_AVATARFOREVER_ENV:-$workspace/.venv-avatarforever}"
cd "$workspace"
python3.11 scripts/workspace.py sources avatarforever
if [[ ! -f "$environment/bin/python" ]]; then
    if command -v uv >/dev/null 2>&1; then
        uv venv --python 3.11 --seed "$environment"
    else
        python3.11 -m venv "$environment"
    fi
fi
# Editable build backends resolve their executable through PATH.
export PATH="$environment/bin:$PATH"
"$environment/bin/python" -m pip install 'pip==25.2' 'uv_build==0.9.8'
"$environment/bin/python" -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
"$environment/bin/python" -m pip install -r infra/requirements-avatarforever.txt
"$environment/bin/python" -m pip install --no-deps --no-build-isolation \
    -e vendor/avatarforever/packages/ltx-core -e vendor/avatarforever/packages/ltx-pipelines
"$environment/bin/python" -m pip check
mkdir -p runs/avatarforever/setup
"$environment/bin/python" -m pip freeze > runs/avatarforever/setup/requirements.lock.txt
"$environment/bin/python" -c 'import torch; from ltx_pipelines import ARA2VidDistilledPipeline; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
# Weights and inference are separate explicit commands.
