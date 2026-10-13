#!/usr/bin/env bash
# Mixed-shape throughput preset. FP8 changes numerical results; see docs/deep-tuning.md.
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/launch-server.sh \
  --quantization float8 \
  --prompt-encoder lean \
  --encoder-batch-size 2 \
  --compile \
  --compile-scope projections \
  --compile-dynamic \
  "$@"
