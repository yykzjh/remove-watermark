#!/usr/bin/env bash
# Connect-only: original-based ROI editing, strict QA, and at most one evidence-backed retry.
set -euo pipefail
cd "$(dirname "$0")/.."
uv_bin=$(command -v uv || true)
uv_bin=${uv_bin:-"$HOME/.local/bin/uv"}
if [[ -z "${UV_PROJECT_ENVIRONMENT:-}" && -x "$HOME/.cache/uv-envs/remove-watermark/bin/python" ]]; then
  export UV_PROJECT_ENVIRONMENT="$HOME/.cache/uv-envs/remove-watermark"
fi
exec "$uv_bin" run --no-sync watermark-batch \
  --first-pass-mode complete \
  --preserve-background \
  --context 0.45 \
  --input-resolution 1024 \
  --repair-blending harmonic \
  --quality-check \
  --quality-batch-size 1 \
  --quality-auto-retry \
  --quality-retry-localized \
  --quality-max-attempts 2 \
  --quality-retry-resolution 1024 \
  --remove-logo \
  "$@"
