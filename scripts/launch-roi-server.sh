#!/usr/bin/env bash
# Automatic source-area ROI sampling. Candidate until its frozen benchmark is accepted.
# General GPU default stays 0-7; pass --gpus 0,1,2,3 for the currently reserved devices.
# To use a global schedule instead: --roi-steps 0 --uniform-steps N.
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/launch-throughput-quality.sh \
  --uniform-steps 0 \
  --roi-steps 12 \
  --roi-step-min-area 0.14 \
  "$@"
