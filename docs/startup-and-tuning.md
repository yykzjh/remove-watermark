> 最新千张首轮与CPU暗框检查见[第二轮验证](first-pass-round2-2026-11-01.md)。检验、自动返工及完整部署参数见 [质量流程](quality-review.md)、[完整参数表](deployment-parameters.md) 和 [最新修补验收实测](review-repair-2026-11-01.md)（此前[大样本测试](quality-performance-2026-10-31.md)）。

> 当前界面见 [工作流说明](workflows.md)，新一轮整图吞吐实验见 [深度调优报告](deep-tuning.md)。

追加部署组合实验见 [第二轮调优报告](tuning-second-pass.md)。

# 服务启动、动态凑批与调优

在 Linux 开发机 ~/project/inference/remove-watermark 执行，使用 uv。
完整可执行脚本：[launch-server.sh](../scripts/launch-server.sh)。
历史区域实验见 [batch-tuning-results.md](batch-tuning-results.md)，不可与当前整图吞吐直接比较。

## 启动

1024输入预算的新增实测配方：[launch-throughput-untiled.sh](../scripts/launch-throughput-untiled.sh)；
8卡、FP8、动态projections、batch1、不分块VAE，concurrency推荐32。
257张同输入热任务比原FP8动态投影配方耗时降低7.7%，峰值分配33501 MiB。
需要分块控制显存时可追加 --vae-tiling。完整对照和限制见第二轮报告。

吞吐优先可选 [launch-throughput.sh](../scripts/launch-throughput.sh)（FP8动态投影编译），
固定形状大任务可选 [launch-throughput-fixed.sh](../scripts/launch-throughput-fixed.sh)（FP8分块编译）。
以下 launch-server.sh 为保守BF16配方；两种FP8脚本复用完整参数清单。
实测、质量取舍与万张估算见[深度调优报告](deep-tuning.md)。


```bash
uv sync --frozen --extra server --extra quantization --no-dev
bash scripts/launch-server.sh --print-config
# 确认任务结束后重启：
bash scripts/stop-server.sh
bash scripts/launch-server.sh
uv run --no-sync python scripts/check-health.py
```

starting 仅表示进程创建，/health stage=ready 才表示所有模型加载完成。
.run/deployment.json 保存最终参数，.run/server.log 保存服务日志。
有活动服务时显式传新参数会报错，不会悄悄忽略参数或中断现有任务。

非布尔参数优先级：命令行 > TOML > 内置值。
**所有 bool 开关不传为 false，传 --参数名 为 true**。不提供 --no-*，不接受 true/false 字符串。
启动 CLI 的布尔默认 false 会覆盖旧 TOML 中的 true。完整脚本显式传入需要启用的开关；
要关闭脚本中已列出的开关，删除该行，或直接使用 start-server.sh 列出需要启用的开关。
标量重复时最后一次生效；GPU/路径列表整体替换。

## 全部部署参数

| 参数 | 含义、默认或完整脚本配置 |
| --- | --- |
| --config | 基础 TOML，默认 config/deployment.toml |
| --host | 127.0.0.1 或 ::1，仅监听回环 |
| --port | 默认 8765，1024–65535；客户端与隧道同步修改 |
| --model | 仓库 ID 或本地模型路径；默认 Qwen/Qwen-Image-2.1-Turbo |
| --model-source | huggingface / modelscope；脚本用 modelscope，依赖和模型由开发机下载 |
| --revision | 脚本 master；此分支可变，严格复现应固定不可变版本 |
| --cache-dir | 脚本 ~/.cache/watermark-models |
| --parallelism | data / single；single 限制一个 GPU 一个副本，但仍可在该卡凑批 |
| --gpus | 默认 0,1,2,3,4,5,6,7，启用全部 8 卡 |
| --replicas-per-gpu | 默认 1，范围 1–4；复制完整权重，L40 不建议多副本 |
| --batch-size | 默认 2，范围 1–16；每 GPU 单次模型调用最多独立整图/自定义区域数 |
| --batch-window | 默认 4，范围 1–8；每 worker 候选图片上限=batch_size×batch_window |
| --batch-bucketing | bool，按相同推理宽高分桶；完整脚本显式启用 |
| --oom-split-batch | bool，CUDA OOM 递归拆批；单图仍失败则报告且不写半成品 |
| --load-parallelism | 默认 8，1–16；仅控制同时加载模型的进程数，不控制推理并发 |
| --dtype | bfloat16 / float16；BF16 为默认，精度变更需重新验收画面 |
| --attention | sdpa / flex；当前固定依赖 Flex 实测编译失败，启动校验会拒绝，使用 sdpa |
| --compile | bool，编译 transformer；首次和新形状有额外编译开销 |
| --compile-mode | default / max-autotune-no-cudagraphs；后者增加冷启动搜索耗时。reduce-overhead / max-autotune 因 CUDA Graph 错误拒绝 |
| --attention-backend | native（默认）/ native_flash / native_cudnn / native_efficient / flash_hub / sage_hub；仅切换图像 attention，文本因果部分保持 native；Hub 需 kernels extra、网络及兼容二进制 |
| --compile-scope | full（默认）/ blocks / projections；blocks 编译重复 Transformer 块，projections 仅编译 attention 线性投影与 MLP，避开 KV/mask 分支 |
| --compile-dynamic | bool，动态形状编译；不能保证完全消除不同宽高比/尾批重编译 |
| --prompt-encoder | stock（默认）/ lean；lean 跳过无用 LM logits、全层 hidden history 和生成 KV |
| --encoder-batch-size | 默认 1，1–16；仅 lean 生效，独立图像联合编码的上限；与去噪 batch-size 分开 |
| --fuse-qkv | bool，合并 Transformer Q/K/V 线性层；本机没有稳定收益，保持关闭 |
| --quantization | none（默认）/ float8 / int8；仅量化 Transformer 块线性层，需 quantization extra；可能改变生成细节 |
| --float8-granularity | tensor（默认）/ row；仅 FP8 生效，分别用整张量/逐行 scale；硬件支持、性能和数值误差需实测 |
| --cudnn-benchmark | bool，搜索卷积算法，固定形状可能加速；新形状首次更慢，可能增加临时显存，不保证确定性 |
| --vae-channels-last | bool，仅转换 VAE Conv2d/Conv3d 权重的内存布局；可能加速卷积，也可能因转换开销变慢 |
| --vae-tile-size | 默认 256，128–1024、32 倍数；启用 vae-tiling 时的像素块边长 |
| --vae-tile-stride | 默认 192，64–1024、32 倍数且不大于块边长；重叠宽度=tile-size−stride |
| --expandable-segments | bool，CUDA 初始化前启用可扩展 allocator，减少碎片；无法解决真实容量不足 |
| --shape-sort | bool，扫描后按实际推理宽高排序，帮助整图同形状凑批；不改变输出相对路径 |
| --scan-workers | 默认 1，1–16，验证图像的 CPU 线程数；缓存热的小文件头单线程更快 |
| --report-interval | 默认 1，1–1024；每 N 个结果落盘 summary，结果 JSONL 仍逐图记录，结束/取消总会刷新；崩溃时 summary 可能滞后 |
| --offload | none / model / sequential；卸载换显存，不能与 compile 同时启用 |
| --vae-tiling | bool，VAE 分块节省高分辨率峰值显存 |
| --vae-slicing | bool，VAE 分片降低 batch 解码峰值；去噪仍批量执行 |
| --kv-cache | bool，同一批去噪步骤复用前缀 KV，不跨照片共享内容 |
| --tf32 | bool，允许符合条件的 FP32 矩阵乘使用 TF32；BF16 权重不改变 |
| --cpu-threads | 每 worker 的 PyTorch CPU 线程，默认 1，1–64 |
| --allowed-roots | 允许输入/输出目录根，默认家目录和 /mnt，空格分隔 |
| --state-dir | 任务结果、错误和参数记录，默认 ~/.local/state/watermark-studio |
| --backend | qwen 真实模型 / mock 测试复制，默认 qwen |
| --mock | bool 开关，backend=mock 的快捷方式 |
| --print-config | bool，打印最终配置并退出 |
| --help | 显示所有启动参数 |

完整脚本中的开关组合才是调优运行配置；裸 start-server.sh 的所有 bool 均关闭。
查看每个选项的当前实际值，用 launch-server.sh --print-config。

## concurrency 的新含义

工作进程数 W = GPU 数 × 每卡副本数。

**concurrency 是全服务同时在途的原图数上限**：包含检测、等待凑批、推理和写入。
它不再表示工作进程数，也不是每卡 batch 大小。调度器保证已分配图片总数不超过此值。

- 8 卡、batch_size=2、concurrency=16：通常每卡领取两张图。
- concurrency=32/64：提供更多凑批候选，但每次模型调用仍最多 batch_size 张整图（自定义模式为区域）。
- 最大在途容量 = W × batch_size × batch_window；超出时 API 拒绝，GUI 限制输入。
- 批量模式每张整图对应一次编辑；单张自定义模式的一图多框产生多个区域编辑。
- 目录扫描完成后才处理，候选队列已知；尾批立即执行，不人为增加 batch-wait-ms 延迟。
- 尺寸不兼容或图片不足时用较小 batch，不为凑批强制拉伸原图。

批量模式对每张整图独立编码，再将相同推理尺寸的图像组成去噪 batch。
Diffusers 原生 image 列表仍是共享参考图；项目适配层独立编码各图并验证 token 布局。
不同长度提示词使用正确 padding mask；大块 padding attention 按样本剔除无效 key，
投影和 MLP 仍使用批维度。该优化不跨图混合参考像素。

## 原图尺寸与质量

批量和单张自动模式均对整图推理；单张自定义区域先处理选区，再合并原图。
模型输入 resolution 是面积预算 N²，宽高按原图比例取整到 32；不是最终输出尺寸。
默认输出存储宽高与原文件一致，并保留 EXIF 方向；显式 output_width/output_height 在合成完成后调整整张图。
当前输出调整使用 Lanczos 重采样，不是 AI 超分，不能补回低输入预算缺失的真实细节。
原图不覆盖，输出目录保持相对结构。生成内容是自然补全，无法保证还原被遮挡的真实细节。
整图生成也可能改变非水印区域；需要严格保留其它像素时使用单张自定义区域。

## 全部任务参数

API 请求还需 input_dir、output_dir，均为开发机目录，不能相同或相互包含。
完整示例：[job.example.json](../config/job.example.json)。

| options 字段 | 默认与作用 |
| --- | --- |
| concurrency | 32，1–1024 且不超过缓冲容量；GUI 推荐 W×batch_size×min(2,batch_window) |
| scope | 批量由服务强制 full_image；单张 auto 也使用 full_image，单张 custom 使用 regions |
| preset | auto / custom 为当前 UI 两种模式；gray/blue 仅旧 API 兼容，不用于批量 |
| regions | custom 必填，最多 8 个；kind=panel/logo/custom，box 为 0..1 坐标；批量不用 |
| detection_resolution / remove_logo | 旧区域检测 API 参数；当前整图提示词总包含左右两类水印，批量不依赖这两个字段 |
| whole_prompt_profile | 默认neutral中性材质保真；scene增加受限原图事实并只对纸张/无植被土石场景补充约束；baseline为历史对照。检验不变，见[提示词实测](whole-prompt-tuning-2026-11-01.md)。 |
| quality_retry_resolution | 默认1024；整图默认返工不降至768。可显式选择其它预算。 |
| resolution | 默认 1024；UI 为 512/768/1024/1280/1536/2048 面积预算档位，API 允许 256–2048 的 32 倍数 |
| output_width / output_height | 同为 0 保留每张原图尺寸；否则两者都填，调整完整输出画布；最多 6400 万像素 |
| context | 默认0.6，0.1–2；精细首轮或单张重建的上下文系数。Linux桌面/批处理CLI可显式传`--context`；不在批量页面提供统一水印区域。历史裁切第二轮采用0.45；整图直接路径不裁切，因此该值不改变整图编辑范围。 |
| feather | 12，0–64；仅自定义区域合成过渡宽度 |
| seed | 42；结合相对路径与区域序号；不同精度、编译或批大小仍可能有数值差异 |
| overwrite | false；不开启时跳过已存在输出 |
| extra_prompt | 空，最多 2000 字符；场景补充提示词 |

```bash
uv run --no-sync watermark-batch /home/zjh/images /home/zjh/images-clean \
  --concurrency 32 --input-resolution 1024
```

## 调优与复现

固定整图输入 1024、输出原尺寸、Turbo 原生 8 步、完整提示词后再比较吞吐。
本轮复现脚本、具体硬件、冷/热启动定义、各项失败和参数取舍见 [深度调优报告](deep-tuning.md)。
不要把降低输入分辨率的速度提升当作同质量加速，也不要把单卡外推当作八卡实测。
并发数主要保证 GPU 不断粮，超过服务容量不会增加计算能力；更大的 batch 也可能降低吞吐。

每图 results.jsonl 包含 window_seconds、window_images、batch_sizes、peak_allocated_mib、
peak_reserved_mib、oom_retries、output_size。窗口指标重复记录到各图，不能按图片简单求和。
PyTorch 峰值不包含其它进程/CUDA 上下文全部开销。只通过 SSH 返回小型状态摘要，测试图片与输出留在开发机。

## 客户端与同步

先在开发机运行完整启动脚本，然后在 WSL 使用桌面客户端连接：
```bash
uv run --no-sync watermark-studio --ssh-host cdu_dev
```
桌面在 WSLg 运行，窗口显示在 Windows 桌面。图像和模型均在开发机处理；
SSH 隧道以目录/状态/参数为主；单张打开预览时可按需返回最多 64 KiB JPEG。更改 GPU/batch 容量后重开客户端。
自定义端口用 --remote-port 和 --port；IPv6 需要相应 IPv6 转发。

代码必须本地 Linux 环境开发、git commit/push，开发机 git pull --ff-only。
不通过 SSH 传源码包或模型；用户允许偶尔少量测试图片，批量图像由用户在开发机准备。

参考：[官方 QwenImage21Pipeline](https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage21)。


## 整图输入与逐图保护的新配方

当前推荐采用完整原图输入，并显式启用原图保护；定位放大图不进入扩散生成。完整对照、独立千张、失败候选留存和限制见[整图保护报告](whole-protected-scale-2026-11-01.md)。

~~~bash
# 默认8卡；仅使用空闲卡时显式覆盖列表和加载并行度。
bash scripts/launch-throughput-quality.sh --gpus 1,2,3,4,5 --load-parallelism 5
# 客户端在WSL/Linux，路径指向开发机；只预选设置，不自动提交任务。
uv run --no-sync watermark-studio --ssh-host cdu_dev --poll-seconds 15 \
  --preserve-background --first-pass-mode whole --whole-prompt-profile neutral \
  --input-resolution 1024 --quality-check --quality-cross-review \
  --quality-auto-retry --quality-batch-size 2 --quality-retry-resolution 1536
~~~

- 1024扩散batch2，大于1024的帧用large-batch-size=1；source编码凑批与扩散凑批分开，encoder-batch-size=4。
- separate-vae-decode在扩散调用结束后按相同算式解码，释放局部prefix KV；它不是额外生成或降分辨率。
- quality-phase-offload-above=1024仅对更大预算使用主机镜像切换。每worker约8.27GiB额外主机内存，非量化独立检验器；普通1024避免每窗口搬运。
- protected-document-resolution=1536只对较大文档且基础预算>=1024升档；0关闭。512/768的明确快速选择不被覆盖，输出尺寸仍单独设置。
- protection-detail-scan补足小码，protection-core-feather只柔化定位安全边距，protection-physical-vet核对小panel的原图实物载体。原图内容分歧不盲目重绘。
- retain-rejected-first保留首份未通过候选及SHA，避免返工掩盖首轮问题；会增加未通过项的磁盘占用。
- source-vet-batch-size默认1；2..4为独立的编辑侧载体判断凑批，独立验证器的载体证明仍逐项。必须检查相同冻结候选的标志和保护决定不变，且端到端更快，才应选择更大值。
- 所有布尔参数仍省略=false、指定=true。已在配方中启用的开关，不能用末尾省略来关闭；使用start-server.sh或复制配方删除该行。

完整保守示例scripts/launch-server.sh覆盖当前全部部署字段，关闭的布尔开关以注释列出；--print-config只输出有效配置、不加载模型。不要从前向次数减少或单个kernel更快推断整个批任务耗时降低。千张验收需同时记录首次及最终自动通过、异常、独特源文件SHA、原始输出宽高、实际模型预算、全部返工及总墙钟时间。
