#!/usr/bin/env bash
# Complete eight-L40 deployment example. Trailing CLI options override values below.
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/start-server.sh \
  --config config/deployment.toml \
  --host 127.0.0.1 \
  --port 8765 \
  --backend qwen \
  --model Qwen/Qwen-Image-2.1-Turbo \
  --model-source modelscope \
  --revision master \
  --cache-dir "$HOME/.cache/watermark-models" \
  --parallelism data \
  --gpus 0,1,2,3,4,5,6,7 \
  --batch-size 2 \
  --batch-window 4 \
  --batch-bucketing \
  --oom-split-batch \
  --replicas-per-gpu 1 \
  --load-parallelism 1 \
  --dtype bfloat16 \
  --attention sdpa \
  --compile-mode default \
  --offload none \
  --vae-tiling \
  --vae-slicing \
  --kv-cache \
  --tf32 \
  --cpu-threads 1 \
  --allowed-roots "$HOME" /mnt \
  --state-dir "$HOME/.local/state/watermark-studio" \
  "$@"
