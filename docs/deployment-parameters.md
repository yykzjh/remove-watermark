# 完整部署参数表

由当前DeploymentConfig和CLI帮助生成。默认布尔开关均关闭；启动配方会显式启用所需项目。

|参数|默认|含义|
|---|---|---|
|--host|127.0.0.1|Loopback bind address: 127.0.0.1 or ::1|
|--port|8765|API port, 1024..65535|
|--model|Qwen/Qwen-Image-2.1-Turbo|Model repository ID or a local model directory|
|--revision|main|Model repository revision (main on HF, master on ModelScope, or pinned commit)|
|--model-source|huggingface|Model download provider; ignored for a local model directory|
|--cache-dir|~/.cache/huggingface/hub|Model download cache directory|
|--gpus|[0, 1, 2, 3, 4, 5, 6, 7]|Comma-separated physical GPU indices, e.g. 1,2,3|
|--parallelism|data|single: exactly one worker; data: independent image workers on selected GPUs|
|--batch-size|2|Maximum independent edits per GPU forward, 1..16|
|--prefill-overlays|False|Experimental source-only harmonic conditioning for focused or protected whole editing; whole mode requires trusted large-card cores. Small groups remain original for Qwen removal. No extra generation.|
|--prefill-original-semantics|False|With prefill-overlays in protected whole mode, encode the unchanged full photograph and baseline prompt; initialize only the VAE reference. No extra image encoding or generation; experimental, not in default recipe.|
|--protect-denoising-context|False|Experimental full-frame flow context preservation using already encoded original VAE latents. Same official sigma schedule and full-frame forwards. Incompatible with prefill-overlays; not enabled in recipe.|
|--large-batch-size|0|Optional cap for input budgets >1024; 0 inherits batch-size, 1 avoids large-frame B2 peaks|
|--small-batch-size|0|Separate cap for model input budgets <=512; 0 inherits batch-size, 1..16 overrides it. Does not raise the raw-image concurrency/window. Exact shapes and OOM learned caps still apply. Quality recipe explicitly selects 2.|
|--batch-window|4|Images gathered per worker = batch size times window, 1..8|
|--batch-bucketing|False|Group edits by identical inference dimensions within the window|
|--adaptive-dispatch|False|Shrink worker windows near job tail to balance remaining work|
|--detection-batch-size|1|Maximum compatible VLM detection images per forward, 1..4; OOM splits|
|--oom-split-batch|False|Recursively split a batch on CUDA OOM; singleton failure is reported|
|--replicas-per-gpu|1|Full model copies per GPU, 1..4; multiplies VRAM use|
|--quality-verifier|False|Enable post-removal review using the selected quality backend|
|--protected-document-resolution|0|Optional full-frame document budget, 0 disables; protected whole requests >=1024 only, never downsizes chosen budget|
|--retain-rejected-first|False|Keep an immutable first rejected candidate for audit; adds disk use only for rejected first attempts|
|--source-vet-batch-size|1|Independent editor-side source claims per forward, 1..4; verifier proofs remain per candidate. Experimental until GPU speed and frozen-reference invariance are measured.|
|--protection-physical-vet|False|Before protected whole editing, retain small physical text only with two distinct proofs. Outdoor masks >=35% additionally require two agreeing tighter detections and digital identity proofs; disagreement preserves original pixels and forces review.|
|--protection-detail-scan|False|Complete source-only fine codes for whole-image protected editing; never crop diffusion inputs|
|--protection-core-feather|False|Fade only the source detector safety margin; keep the complete watermark core opaque|
|--separate-vae-decode|False|Decode pipeline latents after per-call prefix KV caches are released; exact pinned decode path, offload none|
|--quality-phase-offload|False|Park independent unquantized verifier weights in host RAM during editing; restore before QA; no SSH transfer|
|--quality-phase-offload-above|0|Swap verifier weights only above this model-input budget; 0 disables adaptive swapping|
|--quality-cross-full-context|False|Experimental full-result context in physical-text dispute localization and proofs. No guard, source overlap, pixel, dual-model or residual-scan condition is waived.|
|--quality-rectangle-guard|False|With quality verification, add CPU-only review of coherent multi-side panel tone steps; records boundary evidence, cannot approve images or override existing vetoes. No extra model call. Quality recipe explicitly enables it; see [round 2 validation](first-pass-round2-2026-11-01.md).|
|--quality-backend|resident|resident: reuse image encoder; independent: separate instruction VLM; launch-throughput-quality.sh explicitly selects independent|
|--quality-model|Qwen/Qwen3-VL-4B-Instruct|Independent quality VLM repository or local path; adds GPU memory|
|--quality-revision|main|Independent quality VLM revision (master for ModelScope)|
|--quality-quantization|none|Independent verifier language layers: none, int8 or nf4; requires quality-quantization extra. Vision layers and lm_head retain selected dtype; a failed precision audit prevents startup.|
|--dtype|bfloat16|Model weight and inference dtype|
|--attention|sdpa|sdpa; flex is rejected on the pinned stack after measured compiler failures|
|--attention-backend|native|Unmasked/image attention kernel; text causal masks keep native. Hub kernels require kernels extra|
|--compile|False|Compile transformer; first inference and new shapes can take longer|
|--compile-mode|default|default or max-autotune-no-cudagraphs; CUDA graph modes remain rejected|
|--compile-scope|full|full transformer, repeated blocks, or only attention projections and MLP|
|--compile-dynamic|False|Compile dynamic shapes; requires --compile to take effect|
|--prompt-encoder|stock|stock or lean (omit unused LM logits, hidden-state history and VLM KV cache)|
|--encoder-batch-size|1|Independent lean VLM prefill across worker window; decoupled from diffusion batch, 1..16, OOM splits|
|--fuse-qkv|False|Fuse transformer Q/K/V linear projections; exact arithmetic structure, rounding may differ|
|--quantization|none|none, transformer FP8 dynamic activation/weight, or INT8 weight-only; requires quantization extra|
|--float8-granularity|tensor|FP8 scaling per tensor or per row; only active with --quantization float8|
|--cudnn-benchmark|False|Search cuDNN convolution algorithms; first/new shapes cost more and may use more memory|
|--vae-channels-last|False|Convert VAE Conv2d/Conv3d weights to channels-last layouts; benchmark on target GPU|
|--vae-tile-size|256|VAE square tile edge in pixels; only active with --vae-tiling|
|--vae-tile-stride|192|VAE tile stride in pixels, <= tile size; overlap = size - stride|
|--expandable-segments|False|PyTorch expandable CUDA allocator segments; set before CUDA initialization to reduce fragmentation|
|--shape-sort|False|Sort scanned images by inference shape to improve homogeneous batching|
|--scan-workers|1|CPU workers for image verification, 1..16|
|--report-interval|1|Persist summary every N completed images; final/control state always flushed|
|--offload|none|CPU offload: none, model, sequential; incompatible with compile|
|--vae-tiling|False|Tile VAE to reduce activation memory|
|--vae-slicing|False|Slice VAE batch to reduce activation memory; transformer batch is unchanged|
|--kv-cache|False|Reuse prefix KV within the current image's denoising steps|
|--tf32|False|Allow TF32 for eligible FP32 CUDA matrix multiplications|
|--cpu-threads|1|PyTorch CPU threads per worker, 1..64|
|--load-parallelism|8|Concurrent model-loading workers, 1..16; does not set job concurrency|
|--allowed-roots|['~', '/mnt']|Replace allowed directory roots; space-separated paths|
|--state-dir|~/.local/state/watermark-studio|Directory for job summaries and per-image result logs|
|--backend|qwen|qwen: real model; mock: tests only, copies images without removal|


## 单独加载8B检验候选（同一基础权重）

在Linux开发机安装可选依赖（由开发机直接下载）：

~~~bash
uv sync --frozen --extra server --extra quantization --extra quality-quantization --no-dev
uv run --no-sync python scripts/download-quality-model.py --model Qwen/Qwen3-VL-8B-Instruct --workers 8
bash scripts/launch-throughput-quality.sh \
  --quality-model "$HOME/.cache/watermark-models/models/Qwen--Qwen3-VL-8B-Instruct/snapshots/master" \
  --quality-quantization nf4
~~~

启动配方默认8卡。其他任务占卡时用 --gpus 和 --load-parallelism 显式选择可用卡。
NF4降低语言层驻留显存，不保证比BF16更快或更准确；需要同时比较正确结果误拒、
缺陷漏检和端到端耗时。当前Transformers使用完整模块名匹配量化排除项，
因此显式排除model.visual；加载后再次检查视觉模块不存在Linear4bit/Linear8bitLt。
resident复用去水印编码器，health如实报告::text_encoder；它不提供两个独立模型意见，
不能启用双模型争议改判。none/int8/nf4是字符串选择项，不是bool开关。

当前官方8B检查点与图像模型text_encoder的全部750个权重张量指纹相同。independent表示单独加载的实例，不保证权重独立。
服务会进行规范化张量指纹审计；同一基础权重或身份未知时cross_review_supported=false，API与桌面禁止双模型改判。
8B/NF4仍可作单独检验角色；不要向该配方的客户端传--quality-cross-review。4B等不同权重也必须先通过身份核验。
