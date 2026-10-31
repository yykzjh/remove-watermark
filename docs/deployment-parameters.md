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
|--small-batch-size|0|Separate cap for model input budgets <=512; 0 inherits batch-size, 1..16 overrides it. Does not raise the raw-image concurrency/window. Exact shapes and OOM learned caps still apply.|
|--batch-window|4|Images gathered per worker = batch size times window, 1..8|
|--batch-bucketing|False|Group edits by identical inference dimensions within the window|
|--adaptive-dispatch|False|Shrink worker windows near job tail to balance remaining work|
|--detection-batch-size|1|Maximum compatible VLM detection images per forward, 1..4; OOM splits|
|--oom-split-batch|False|Recursively split a batch on CUDA OOM; singleton failure is reported|
|--replicas-per-gpu|1|Full model copies per GPU, 1..4; multiplies VRAM use|
|--quality-verifier|False|Enable post-removal review using the selected quality backend|
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
|--encoder-batch-size|1|Independent VLM images per lean encoder forward; 1..16|
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


## 独立8B检验候选

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
