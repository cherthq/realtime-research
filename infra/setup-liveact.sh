#!/usr/bin/env bash
set -euo pipefail
workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
environment="${CHERT_LIVEACT_ENV:-/home/ubuntu/envs/liveact}"
cd "$workspace"
if ! command -v sox >/dev/null 2>&1; then
    sudo -n apt-get update -qq
    sudo -n apt-get install -y --no-install-recommends sox libsox-dev libsox-fmt-all
fi
if [[ -d "$workspace/vendor/liveact/.git" ]]; then
    git -C "$workspace/vendor/liveact" config --local core.excludesFile "$workspace/infra/vendor-generated.gitignore"
fi
python3 scripts/workspace.py sources liveact
git -C "$workspace/vendor/liveact" config --local core.excludesFile "$workspace/infra/vendor-generated.gitignore"
if [[ ! -f "$environment/bin/python" ]]; then
    python3 -m venv "$environment"
fi
"$environment/bin/python" -m pip install pip==25.2 wheel 'setuptools<81'
"$environment/bin/python" -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
"$environment/bin/python" -m pip install -r infra/requirements-liveact.txt
MAX_JOBS=8 "$environment/bin/python" -m pip install flash-attn==2.8.3 --no-build-isolation
# Freeze LightX2V separately instead of following its moving default branch.
dependency="$workspace/vendor/lightx2v"
revision=7b6cd61ae4694b962c62745a3b73a9e67a4870e6
if [[ ! -d "$dependency" ]]; then
    git init "$dependency"
    git -C "$dependency" remote add origin https://github.com/ModelTC/LightX2V.git
    git -C "$dependency" fetch --depth 1 origin "$revision"
    git -C "$dependency" checkout --detach FETCH_HEAD
fi
[[ "$(git -C "$dependency" rev-parse HEAD)" == "$revision" ]]
"$environment/bin/python" -m pip install -e "$dependency" -c infra/requirements-liveact.txt
printf 'LiveAct installation finished. Single-GPU inference validation still required.\n'
