#!/usr/bin/env bash
# Optional NVIDIA MPS for AvatarForever and Qwen sharing one Linux GPU.
set -euo pipefail
workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_MPS_PIPE_DIRECTORY="$workspace/.cache/realtime-mps/pipe"
export CUDA_MPS_LOG_DIRECTORY="$workspace/.cache/realtime-mps/log"
command -v nvidia-cuda-mps-control >/dev/null

case "${1:-}" in
  start)
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    chmod 700 "$workspace/.cache/realtime-mps" "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    if ! printf 'get_server_list\n' | nvidia-cuda-mps-control >/dev/null 2>&1; then
      nvidia-cuda-mps-control -d
    fi
    ;;
  run)
    shift
    if [[ $# -eq 0 ]]; then echo 'Supply the service command after run.' >&2; exit 2; fi
    printf 'get_server_list\n' | nvidia-cuda-mps-control >/dev/null
    exec "$@"
    ;;
  status)
    printf 'get_server_list\n' | nvidia-cuda-mps-control
    ;;
  stop)
    servers="$(printf 'get_server_list\n' | nvidia-cuda-mps-control)"
    for server in $servers; do
      clients="$(printf 'get_client_list %s\n' "$server" | nvidia-cuda-mps-control)"
      if [[ -n "$clients" ]]; then
        echo 'Stop the avatar and Qwen services before stopping MPS.' >&2
        exit 1
      fi
    done
    printf 'quit\n' | nvidia-cuda-mps-control
    ;;
  *)
    echo 'Usage: bash infra/realtime-mps.sh start|status|run COMMAND...|stop' >&2
    exit 2
    ;;
esac
