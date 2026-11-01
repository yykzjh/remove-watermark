#!/usr/bin/env bash
# Independent quality VLM adds VRAM; validate GPU memory and throughput for your input budget.
# VAE tiling is omitted intentionally. Use launch-throughput.sh when tiling is needed.
# All booleans remain opt-in; scalar options at the end override this recipe.
# Adaptive verifier residency reserves ~8.27 GiB host RAM per worker for a BF16 mirror.
# Source-claim B4 changed 3/414 primary decisions in the invariance probe: keep B1.
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
  --large-batch-size 1 \
  --small-batch-size 2 \
  --batch-window 4 \
  --adaptive-dispatch \
  --detection-batch-size 4 \
  --source-vet-batch-size 1 \
  --batch-bucketing \
  --oom-split-batch \
  --replicas-per-gpu 1 \
  --load-parallelism 8 \
  --quality-verifier \
  --quality-rectangle-guard \
  --protection-detail-scan \
  --protection-core-feather \
  --protection-physical-vet \
  --separate-vae-decode \
  --quality-phase-offload-above 1024 \
  --protected-document-resolution 1536 \
  --retain-rejected-first \
  --quality-backend independent \
  --quality-model "$HOME/.cache/watermark-models/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master" \
  --quality-revision master \
  --quality-quantization none \
  --dtype bfloat16 \
  --attention sdpa \
  --attention-backend native \
  --compile \
  --compile-dynamic \
  --compile-mode default \
  --compile-scope projections \
  --prompt-encoder lean \
  --encoder-batch-size 4 \
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
