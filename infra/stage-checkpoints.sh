#!/usr/bin/env bash
# Persistent canonical weights -> disposable local SSD copy for faster reloads.
set -euo pipefail
workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cache="${CHERT_MODEL_CACHE:-/home/ubuntu/chert-model-cache-20261024}"
if [[ -e "$cache" ]]; then
    printf 'Cache already exists; refusing to merge into a partial or stale cache: %s\n' "$cache"
    exit 1
fi
mkdir -p "$cache"
for directory in Wan2.2-S2V-14B Live-Avatar LiveAct chinese-wav2vec2-base; do
    cp -a "$workspace/checkpoints/$directory" "$cache/$directory"
done
for model in liveavatar liveact; do
    cp "$workspace/checkpoints/$model-revisions.json" "$cache/"
done
touch "$cache/READY"
printf 'Local SSD checkpoint cache is complete: %s\n' "$cache"
