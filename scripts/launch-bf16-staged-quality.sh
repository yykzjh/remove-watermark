#!/usr/bin/env bash
# Experimental full-precision recipe, not the default throughput configuration.
# ~38 GiB pinned CPU model weights per worker (about304 GiB for8 workers).
# Verified20 reused development images: zero OOM, no first-pass quality advantage.
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/launch-throughput-quality.sh \
  --quantization none \
  --transformer-phase-offload \
  --quality-phase-offload \
  --batch-size 1 \
  --large-batch-size 1 \
  "$@"
