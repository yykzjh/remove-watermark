#!/usr/bin/env bash
# Separately loaded 8B candidate, SAME base weights as the image encoder.
# Cross-model adjudication is disabled by the weight-identity audit.
# Download the verifier and install its extra first.
# Inherits every explicit deployment option and the eight-GPU default from the base recipe.
# Small edits (<=512 budget) may batch by two; large edits keep the base cap of one.
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/launch-throughput-quality.sh \
  --quality-model "$HOME/.cache/watermark-models/models/Qwen--Qwen3-VL-8B-Instruct/snapshots/master" \
  --quality-quantization nf4 \
  --small-batch-size 2 \
  "$@"
