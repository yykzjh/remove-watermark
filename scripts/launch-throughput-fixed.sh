#!/usr/bin/env bash
# Fixed inference shape/batch preset. New shapes and singleton tails can recompile.
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/launch-server.sh \
  --quantization float8 \
  --prompt-encoder lean \
  --encoder-batch-size 2 \
  --compile \
  --compile-scope blocks \
  "$@"
