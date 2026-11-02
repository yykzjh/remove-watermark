#!/usr/bin/env bash
# Connect-only defaults; deployment remains a separate action.
set -euo pipefail
cd "$(dirname "$0")/.."
uv_bin=$(command -v uv || true)
uv_bin=${uv_bin:-"$HOME/.local/bin/uv"}
if [[ -z "${UV_PROJECT_ENVIRONMENT:-}" && -x "$HOME/.cache/uv-envs/remove-watermark/bin/python" ]]; then
  export UV_PROJECT_ENVIRONMENT="$HOME/.cache/uv-envs/remove-watermark"
fi
exec "$uv_bin" run --no-sync watermark-studio \
  --url http://127.0.0.1:8765 \
  --first-pass-mode complete \
  --preserve-background \
  --context 0.45 \
  --input-resolution 1024 \
  --repair-blending harmonic \
  --quality-check \
  --quality-batch-size 1 \
  --quality-retry-resolution 1536 \
  "$@"
