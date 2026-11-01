#!/usr/bin/env bash
# Complete conservative eight-L40 deployment example; see launch-throughput-quality.sh for the measured quality recipe.
# Trailing scalar options override these values. Every boolean is opt-in.
# To disable an enabled boolean, remove its line or call start-server.sh directly.
# Available disabled boolean switches (uncomment only after validating the combination):
# --quality-cross-full-context: Experimental full-result physical-carrier context; all vetoes remain.
# --transformer-phase-offload
# --protect-source-rims: Experimental source photo preservation, two distinct proofs and OCR; independent quality backend and quality-ocr extra required.
# --protect-denoising-context: Experimental full-frame flow context preservation; no extra VAE/diffusion forwards; incompatible with prefill-overlays.
# --prefill-original-semantics: With prefill-overlays, retain the original full-frame VLM reference; initialize VAE only. Experimental, disabled by default.
# --prefill-overlays: Experimental source-only harmonic conditioning for focused or protected whole editing; requires trusted cores for whole mode; no additional generations
# --adaptive-dispatch: Shrink worker windows near job tail to balance remaining work
# --quality-verifier: Enable post-removal review using the selected quality backend
# --retain-rejected-first: Keep an immutable first rejected candidate for audit; adds disk use only for rejected first attempts
# --protection-physical-vet: Vet small physical text and reconcile broad outdoor masks with distinct source detectors and digital/carrier proofs
# --protection-detail-scan: Complete source-only fine codes for whole-image protected editing; never crop diffusion inputs
# --protection-core-feather: Fade only the source detector safety margin; keep the complete watermark core opaque
# --separate-vae-decode: Decode pipeline latents after per-call prefix KV caches are released; exact pinned decode path, offload none
# --quality-phase-offload: Park independent unquantized verifier weights in host RAM during editing; restore before QA; no SSH transfer
# --quality-rectangle-guard: With quality verification, review coherent multi-side panel tone steps; CPU-only, no extra model call
# --compile: Compile transformer; first inference and new shapes can take longer
# --compile-dynamic: Compile dynamic shapes; requires --compile to take effect
# --fuse-qkv: Fuse transformer Q/K/V linear projections; exact arithmetic structure, rounding may differ
# --cudnn-benchmark: Search cuDNN convolution algorithms; first/new shapes cost more and may use more memory
# --vae-channels-last: Convert VAE Conv2d/Conv3d weights to channels-last layouts; benchmark on target GPU
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/start-server.sh \
  --config config/deployment.toml \
  --host 127.0.0.1 \
  --port 8765 \
  --model Qwen/Qwen-Image-2.1-Turbo \
  --revision master \
  --model-source modelscope \
  --cache-dir "$HOME/.cache/watermark-models" \
  --gpus 0,1,2,3,4,5,6,7 \
  --parallelism data \
  --batch-size 2 \
  --large-batch-size 0 \
  --small-batch-size 0 \
  --batch-window 4 \
  --batch-bucketing \
  --detection-batch-size 1 \
  --oom-split-batch \
  --replicas-per-gpu 1 \
  --protected-document-resolution 0 \
  --source-vet-batch-size 1 \
  --quality-phase-offload-above 0 \
  --quality-backend resident \
  --quality-model Qwen/Qwen3-VL-4B-Instruct \
  --quality-revision master \
  --quality-quantization none \
  --dtype bfloat16 \
  --attention sdpa \
  --attention-backend native \
  --compile-mode default \
  --compile-scope blocks \
  --prompt-encoder stock \
  --encoder-batch-size 1 \
  --quantization none \
  --float8-granularity tensor \
  --vae-tile-size 256 \
  --vae-tile-stride 192 \
  --expandable-segments \
  --shape-sort \
  --scan-workers 1 \
  --report-interval 64 \
  --offload none \
  --vae-tiling \
  --vae-slicing \
  --kv-cache \
  --tf32 \
  --cpu-threads 1 \
  --load-parallelism 8 \
  --allowed-roots "$HOME" /mnt \
  --state-dir "$HOME/.local/state/watermark-studio" \
  --backend qwen \
  "$@"
