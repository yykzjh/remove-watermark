#!/usr/bin/env bash
# Eight-L40 throughput candidate measured at 1024 input budget, original-size output.
# VAE tiling is omitted intentionally. Use launch-throughput.sh when tiling is needed.
# All booleans remain opt-in; scalar options at the end override this recipe.
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
  --batch-size 1 \
  --batch-window 4 \
  --batch-bucketing \
  --oom-split-batch \
  --replicas-per-gpu 1 \
  --load-parallelism 8 \
  --dtype bfloat16 \
  --attention sdpa \
  --attention-backend native \
  --compile \
  --compile-dynamic \
  --compile-mode default \
  --compile-scope projections \
  --prompt-encoder lean \
  --encoder-batch-size 1 \
  --quantization float8 \
  --float8-granularity tensor \
  --offload none \
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
