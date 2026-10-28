#!/usr/bin/env bash
# Explicitly invoked remote setup, never run on clone/import.
set -euo pipefail
workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
environment="${CHERT_FLASHTALK_ENV:-/home/ubuntu/envs/flashtalk}"
cd "$workspace"
python3 scripts/workspace.py sources flashtalk
if [[ ! -f "$environment/bin/python" ]]; then
    python3 -m venv "$environment"
fi
"$environment/bin/python" -m pip install 'pip==25.2' wheel 'setuptools<81'
"$environment/bin/python" -m pip install torch==2.7.1 torchvision==0.22.1 xformers==0.0.31 --index-url https://download.pytorch.org/whl/cu128
"$environment/bin/python" -m pip install -r infra/requirements-flashtalk.txt
MAX_JOBS=8 "$environment/bin/python" -m pip install flash_attn==2.8.0.post2 --no-build-isolation
if ! dependency_check="$("$environment/bin/python" -m pip check 2>&1)"; then
    # Upstream's py3 wheel embeds a cp36 WHEEL tag despite using a ctypes API.
    # Do not modify package metadata or suppress unrelated dependency failures.
    if [[ "$dependency_check" != 'decord 0.6.0 is not supported on this platform' ]]; then
        printf '%s\n' "$dependency_check"
        exit 1
    fi
    printf 'Known upstream Decord wheel-tag warning: %s\n' "$dependency_check"
    "$environment/bin/python" -c 'import decord; print(decord.__version__)'
fi
"$environment/bin/python" -c 'import torch, flash_attn, xformers; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name()); assert torch.cuda.is_available()'
cd vendor/flashtalk
"$environment/bin/python" -c 'import flash_talk.inference; print("FlashTalk imports passed")'
