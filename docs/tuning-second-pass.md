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

第一阶段：8 张 L40 分别运行 20 个候选配置，每配置首次调用 + 4 次热调用；
覆盖 tensor/row、Conv 布局、cuDNN、batch 1/2/3、blocks/projections、autotune，
并追加量化后关闭 VAE tiling/slicing 和改变块大小的组合。
微基准跨 GPU 同时运行，有 CPU/功耗竞争，只用于候选筛选。

第二阶段：停止本项目空闲正式服务，独占其八个 GPU worker 做完整服务对照；
每次加载时间单列，同一进程重复任务。原本已有的外部 GPU0 进程不改动。
测试结束恢复正式服务。比较与验收统计将记录在本文件和对应 JSON 中。

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
