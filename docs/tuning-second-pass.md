# 第二轮吞吐调优：FP8 缩放与卷积实现

本轮在既有八卡独立实例、动态凑批、FP8、编译优化之上继续筛选。
核心指标是同一批输入从提交到所有文件保存完毕的墙钟耗时。
输入预算保持 1024、Turbo 8 步、完整去水印提示词、最终文件保持各自原图尺寸。

## 官方实现与候选范围

- [Qwen 模型卡](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)：Turbo 固定检查点推荐的 8 步 schedule，CFG=1，前缀 KV 在当前图像的去噪步骤间复用。不能用跨图片复用条件编码来制造吞吐提升。
- [Diffusers 加速文档](https://huggingface.co/docs/diffusers/main/en/optimization/fp16)：编译范围、动态形状和内存布局均可影响速度；示例是其它 pipeline，不能直接视为 Qwen 的已验证配方。
- [TorchAO FP8 配置](https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.quantization.Float8DynamicActivationFloat8WeightConfig.html)：支持 PerTensor / PerRow 缩放。新增 --float8-granularity tensor/row，默认保持 tensor。量化误差和内核支持取决于硬件；逐行不必然更快。
- [PyTorch 卷积布局转换](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.convert_conv3d_weight_memory_format.html)：新增 --vae-channels-last，仅转换 VAE 的 Conv2d/Conv3d 权重，不对全部五维/线性参数盲目调用 channels_last。
- [cuDNN 算法搜索](https://docs.pytorch.org/docs/2.14/backends.html#torch.backends.cudnn.benchmark)：新增 --cudnn-benchmark，搜索当前卷积形状的算法；首次延迟、临时显存及非确定性都需要记录。
- [Attention 后端](https://huggingface.co/docs/diffusers/main/en/optimization/attention_backends)：Flash/Sage 的架构和二进制限制仍适用。L40 不使用只针对 SM90/Blackwell 的选项，文本因果掩码仍保留。

新增两个 bool 开关仍遵守不传=false、传入=true。全部可通过启动 CLI 设置，
health 状态返回实际生效值。完整参数表见 [启动说明](startup-and-tuning.md)。

## 实验方法

第一阶段：8 张 L40 分别筛选计划中的 20 个候选配置，每配置首次调用 + 4 次热调用；
覆盖 tensor/row、Conv 布局、cuDNN、batch 1/2/3、blocks/projections、autotune，
并追加量化后关闭 VAE tiling/slicing 和改变块大小的组合。
微基准跨 GPU 同时运行，有 CPU/功耗竞争，只用于候选筛选。

第二阶段：停止本项目空闲正式服务，独占其八个 GPU worker 做完整服务对照；
每次加载时间单列，同一进程重复任务。原本已有的外部 GPU0 进程不改动。
测试结束恢复正式服务。比较统计记录在本文件和对应 JSON 中，最终验收见文末。

仅有用户提供的两张实拍图。混合形状压力集由原图硬链接重复及填充变体构成，
可以检验执行稳定性和调度，但不能替代上万张不同真实照片的画质验收。
“冷”指新进程首次推理，保留已有磁盘编译缓存。热性能不代表首次部署总耗时。

## 部署选项的作用阶段

这里的部署调优指在启动服务时决定后续推理实现，而非只加快加载：

| 部署选项 | 影响 |
| --- | --- |
| gpus / parallelism / replicas-per-gpu | GPU 拓扑与独立模型副本数量 |
| quantization / float8-granularity / dtype | 计算精度、权重和激活显存、矩阵乘内核 |
| compile / compile-scope / compile-dynamic / compile-mode | 编译边界、可复用形状、内核选择 |
| attention-backend / kv-cache / fuse-qkv | Attention 执行与每张图片内部缓存 |
| vae-tiling / vae-slicing / tile-size / tile-stride / vae-channels-last | 图像编解码计算和峰值显存 |
| cudnn-benchmark | 卷积算法搜索及后续调用性能 |
| batch-size / batch-window / batch-bucketing | 每卡一次推理数量、待凑批窗口与形状兼容性 |
| prompt-encoder / encoder-batch-size | 图像条件编码路径和编码批大小 |
| expandable-segments / cpu-threads / shape-sort / report-interval | allocator、CPU 工作、调度和记录开销 |

运行时 concurrency 只决定全局在途图片上限，模型输入分辨率影响 token 数；
它们与上面的部署选项分开记录。加载并发数不等于推理并发数。
部署首调用所需的编译/算法搜索仍算成本；“ready”也不意味着所有形状已编译完毕。

## 单卡候选实测

每个成功配置为 1 次首次调用 + 4 次热调用；热秒数取四次中位数。
冷调用包含编译，保留磁盘缓存。并行筛选的微小差距不代表端到端稳定收益。

| 配置 | batch | 首次秒 | 热秒/批 | 张/秒 | 峰值 MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| v2-autotune-b1 | 1 | 69.90 | 6.843 | 0.1461 | 27862 |
| v2-autotune-b2 | 2 | 125.83 | 14.076 | 0.1421 | 31425 |
| v2-blocks-b1 | 1 | 21.18 | 5.919 | 0.1689 | 29517 |
| v2-blocks-b2 | 2 | 22.12 | 12.331 | 0.1622 | 30740 |
| v2-control-b2 | 2 | 15.47 | 13.617 | 0.1469 | 31425 |
| v2-cudnn-b2 | 2 | 24.96 | 13.767 | 0.1453 | 31862 |
| v2-large-tiles-b2 | 2 | 13.47 | 13.419 | 0.1490 | 31425 |
| v2-layout-b2 | 2 | 15.27 | 13.289 | 0.1505 | 31425 |
| v2-layout-cudnn-b2 | 2 | 27.73 | 13.394 | 0.1493 | 31425 |
| v2-no-slicing-b2 | 2 | 14.82 | 13.078 | 0.1529 | 31425 |
| v2-no-tiling-b1 | 1 | 8.04 | 6.126 | 0.1632 | 33501 |
| v2-no-tiling-b2 | 2 | 13.14 | 13.027 | 0.1535 | 35919 |
| v2-projections-b1 | 1 | 8.50 | 6.426 | 0.1556 | 27862 |
| v2-projections-b3 | 3 | 21.78 | 22.026 | 0.1362 | 34988 |
| v2-row-b1 | 1 | 601.28 | 6.142 | 0.1628 | 27867 |
| v2-row-b2 | 2 | 34.48 | 13.318 | 0.1502 | 31430 |
| v2-row-b3 | 3 | 20.12 | 20.149 | 0.1489 | 34993 |
| v2-row-layout-b2 | 2 | 514.14 | 12.937 | 0.1546 | 31430 |

计划 20 个微基准配置，实际 18 个完成。v2-row-blocks-b1 首次调用在进程运行超过 15 分钟后仍未完成，
主动终止该候选；同一队列后续 v2-row-blocks-b2 未执行。两者没有热吞吐成绩，不能宣称不支持或更快。
所有成功配置保持模型原生 8 步，未降低输入预算。

逐行 FP8 的首次成本非常依赖编译路径：row-b1 首次 601.3 秒，row-layout-b2 首次 514.1 秒，
而 tensor 对照首次分别约 8.5 / 15.5 秒。不能只凭热调用几百分点差异开启所有组合。
在万张任务中应使用“准备时间 + 图片数 / 热吞吐”比较，并考虑新形状的额外编译。

相同 seed 的第一个样例相对 v2-control-b2，8-bit 像素 MAE：
layout=1.2710、row=1.8737、cuDNN=1.3589、关闭 tiling=7.6653。
这些数字描述输出变化，不是画质分数。关闭 tiling 后的两张小预览中未见左下信息栏或右下相机文字残留，
但人体、植物、地面生成细节会变化。两张预览分别 45,939 / 61,062 bytes，原图和完整输出均留在开发机。

## 模型副本数量的筛选

在物理 GPU6 单独比较 1 / 2 个 FP8 实例、batch=1。单实例完成两次 8 张任务；
双实例在当前加载路径下、load-parallelism=1 时仍加载 OOM，未获得双实例吞吐。
此结论限定于当前组件精度、加载路径和显存缓存状态，不代表其它量化/卸载实现必然无法容纳。
部署仍采用每卡一个独立模型实例。

## 多卡部署与可用优化的边界

[Diffusers 分布式推理指南](https://huggingface.co/docs/diffusers/main/en/training/distributed_inference)
区分独立数据并行、device_map 组件分配和 Context Parallelism。
在固定依赖的 transformer_qwenimage21.py 中可以确认 repeated_blocks 声明，
但没有该类专用的 _cp_plan / _tp_plan。其它模型的 Ring/Ulysses/TP 示例不能视为
当前 Qwen21 + 独立图像 batch + 前缀 KV 组合已经验证的启动开关。
本项目继续用每卡独立实例分配照片；不将模型拆到八卡就默认等同于吞吐更高。

当前评估保持 CFG=1、检查点的 8-step sigmas。
[Qwen21 API](https://huggingface.co/docs/diffusers/main/en/api/pipelines/qwenimage21)
允许显式 sigmas 覆盖采样网格；这会改变模型生成轨迹和质量，本轮没有靠减少步骤取得加速。
num_inference_steps 在 sample_sigmas 生效时不是一个有效的减步开关。
输出文件尺寸保持原图；推理输入预算、原始细节保留能力和最终输出尺寸是三个不同概念。

每个新任务的 summary.json 保存完整 deployment 配置；/health 中的 job 摘要不重复传输该配置，保持 SSH 状态通信量小。

## 八卡完整服务对照

任务时间包含扫描、读取、推理、合成和保存，不包含模型加载。首轮与热轮分列；
同一行在同一服务进程重复，所有输入、提示词、种子规则、输出编码与尺寸策略一致。

### 混合形状，65 张

| 配方 | 首轮秒 | 热轮 1 秒 | 热轮 2 秒 | 热中位秒 | 峰值 MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp8-projections | 62.949 | 61.694 | 61.717 | 61.706 | 31425 |
| fp8-layout-no-slicing | 63.015 | 61.657 | 60.135 | 60.896 | 31425 |
| fp8-projections-b1 | 61.905 | 60.897 | 60.177 | 60.537 | 27862 |
| fp8-projections-b1-no-tiling | 57.524 | 57.316 | 56.304 | 56.810 | 33501 |

### 固定形状，64 张

| 配方 | 首轮秒 | 热轮 1 秒 | 热轮 2 秒 | 热中位秒 | 峰值 MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp8 | 63.947 | 55.078 | 54.743 | 54.910 | 30740 |
| fp8-b1 | 90.686 | 55.597 | 55.296 | 55.447 | 29656 |

混合形状基线为已有 FP8 tensor + dynamic projections + batch2 + VAE tiling/slicing。
layout-no-slicing 只改 VAE 内存布局和切片；b1 只改 batch 上限；b1-no-tiling 再关闭 VAE 分块。
四组共 780 张输出全部成功、8 卡参与、尺寸错误 0、OOM 0。
热中位比较：layout-no-slicing 减少约 1.3%，b1 减少约 1.9%，b1-no-tiling 减少约 7.9%。
前两者收益小，不能凭单轮差异认定稳定优势。

固定形状两组共 384 张输出全部成功、8 卡参与、尺寸错误 0、OOM 0。
FP8 blocks 的 batch1 热中位反而略慢于 batch2（55.447 对 54.911 秒），首轮更慢。
因此保留 launch-throughput-fixed.sh 的 batch2；单卡微基准排名不能直接替代完整服务排名。

## 使用实测吞吐配方

完整参数脚本：[launch-throughput-untiled.sh](../scripts/launch-throughput-untiled.sh)。
以 1024 输入预算为目标，默认全部八卡、每卡一实例、FP8 tensor、dynamic projections、
batch1、window4、concurrency 建议32、原尺寸输出。量化/编译/VAE策略在服务部署时设置。

~~~bash
cd ~/project/inference/remove-watermark
# 先预览全部生效参数，不启动模型
bash scripts/launch-throughput-untiled.sh --print-config
# 上一个任务结束后切换部署配方
bash scripts/stop-server.sh
bash scripts/launch-throughput-untiled.sh
uv run --no-sync watermark-batch /home/zjh/input /home/zjh/output \
  --concurrency 32 --input-resolution 1024
~~~

输入预算更高或需要 VAE 分块降低显存时，可在该脚本末尾追加 --vae-tiling，
或使用原 launch-throughput.sh。输入预算改变后需重新测 batch 和显存。
新的两个 bool 开关（cudnn-benchmark / vae-channels-last）均不默认启用；
float8-granularity 默认保持 tensor。本机没有足够端到端证据支持默认叠加它们。
动态凑批功能仍支持 batch2/3/4 等配置；b1 是当前 untiled 吞吐配方，不代表通用硬件上 b1 总是最快。

## 复测命令

完整服务测试使用同一输入集，先结束现有任务并停止本项目服务，避免占用相同 GPU。
输出均留在开发机 .run/throughput/，精简指标见 [JSON 数据](tuning-second-pass.json)。
测试脚本每任务默认上限 21600 秒（6 小时），可以通过 --job-timeout-seconds 修改；
生产 watermark-batch CLI 没有任务总时限，HTTP 请求的 10 秒超时不是任务时限。

~~~bash
uv run --no-sync python scripts/benchmark-throughput.py \
  --profiles fp8-projections fp8-projections-b1-no-tiling \
  --images 257 --pattern mixed --repeats 2 --tag verify-257
# 完整万张推理会长时间占用 GPU，以下仅是复测命令，并非本轮已执行的成绩：
uv run --no-sync python scripts/benchmark-throughput.py \
  --profiles fp8-projections-b1-no-tiling \
  --images 10000 --pattern mixed --repeats 2 --job-timeout-seconds 21600 --tag verify-10k
~~~

脚本中的万张数据仍是样例重复/形状变体。真实数据集应通过 watermark-batch 提交实际目录，
并结合任务 summary.json 的 started_at/finished_at、成功数和保存的 deployment 参数计算耗时。

## 257 张相同输入复测与最终选择

| 配方 | 首轮秒 | 热轮秒 | 热吞吐 张/秒 | 峰值分配 MiB |
| --- | ---: | ---: | ---: | ---: |
| 原 FP8 dynamic projections / batch2 / tiling | 230.844 | 227.745 | 1.1285 | 31425 |
| FP8 dynamic projections / batch1 / untiled | 213.194 | 210.115 | 1.2231 | 33501 |

四个任务均 257/257 成功，8 卡全部参与，尺寸错误 0、OOM 0。
同输入热轮耗时减少 **7.74%**，吞吐提高 **8.39%**；
峰值分配增加 2076 MiB（约6.6%）。所有逐窗口峰值均不超过各自表中上限。
这不是任意输入/任意时长下的显存保证，峰值分配也不等于 nvidia-smi 的总占用。

本轮累计 **26 个成功端到端任务、2336 次推理与输出保存**。
另有 18 个完成微基准配置（合计165次图像推理）、1个首次调用超时主动终止、
1个未执行配置，以及双实例加载 OOM。失败/未完成候选未被计入成功吞吐。
采用 untiled 配方后，推荐 concurrency 在8卡、batch1/window4时为32；仍受服务 capacity 限制。
固定形状 blocks 配方继续保持 batch2，VAE channels-last、cuDNN benchmark、FP8 row 均保留可选而非默认启用。

仅按上述热吞吐线性外推一万张，原配方约 **147.7 分钟**，
untiled 配方约 **136.3 分钟**，差约 **11.4 分钟**。
另加模型加载、未命中的编译缓存和实际存储开销。
这不是一万张真实独立照片的实测承诺：样例重复使文件系统缓存偏热，真实内容和宽高比分布可能改变结果。

基准使用模型缓存中的 ModelScope master 快照，软件由 uv.lock 固定；
master 是可变名称，严格重现实验时还需固定模型快照/权重。
当前包元数据：PyTorch2.14.1、Diffusers0.41.0.dev0、Transformers5.19.0、TorchAO0.18.0；
NVIDIA驱动580.105.08，8×L40，每卡46068 MiB。

## 正式服务最终验收

代码提交 6c348b7 同步后，以 launch-throughput-untiled.sh 恢复 8765 服务，八个 worker 均 ready。
scripts/smoke-workflows.py 验收全部通过：

| 流程 | 成功数 | 耗时秒 | 输出尺寸 |
| --- | ---: | ---: | --- |
| 批量全图（八卡全部参与） | 16 | 18.06 | 1706×1279、4000×3000，与各原图一致 |
| 单张自定义两个区域 | 1 | 14.04 | 1920×1440，符合指定分辨率 |
| 单张自动识别 | 1 | 8.03 | 4000×3000，与原图一致 |

以上18张为功能验收，不计入26个性能测试任务的2336张统计。
WSL Linux环境下52项测试通过，Ruff、shell语法及diff检查通过。
验收结束八卡空闲，未遗留性能测试进程；原有外部GPU0进程保持运行。
