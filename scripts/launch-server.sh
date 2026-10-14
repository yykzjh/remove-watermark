#!/usr/bin/env bash
# Complete eight-L40 deployment example. Trailing CLI options override values below.
# Omitted booleans (compile, compile-dynamic, fuse-qkv, cudnn-benchmark,
# vae-channels-last) remain false.
# To disable an enabled boolean, remove its line or call start-server.sh directly.
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
  --load-parallelism 8 \
  --dtype bfloat16 \
  --attention sdpa \
  --attention-backend native \
  --compile-mode default \
  --compile-scope blocks \
  --prompt-encoder stock \
  --encoder-batch-size 1 \
  --quantization none \
  --float8-granularity tensor \
  --offload none \
  --vae-tiling \
  --vae-tile-size 256 \
  --vae-tile-stride 192 \
  --vae-slicing \
  --kv-cache \
  --tf32 \
  --cpu-threads 1 \
  --expandable-segments \
  --shape-sort \
  --scan-workers 1 \
  --report-interval 64 \
  --allowed-roots "$HOME" /mnt \
  --state-dir "$HOME/.local/state/watermark-studio" \
  "$@"
