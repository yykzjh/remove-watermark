# Qwen-Image-2.1-Turbo 整图吞吐深度调优

## 目标与测量约定

目标是在不漏图、不省略任何水印类别、保留原始输出尺寸的前提下，缩短大量图片的总完成时间。
批量模式对整张照片推理，不使用固定水印区域；提示词包含左下信息面板和右下今日水印/相机/真实时间/防伪码等附属块，
并要求自然补全人体、泥土、草地、树木和植物。

本轮平台为开发机 8 × NVIDIA L40（每卡 46068 MiB），128 CPU、约 1 TiB RAM。
GPU 0 有一个约 1.1 GiB 的既存进程，本轮未改动它。
软件固定于 uv.lock：PyTorch 2.14.1+cu130、Diffusers 0.41.0.dev0
（916a5791ed5f29d966200d2eee53f06862ec4434）、Transformers 5.19.0、TorchAO 0.18.0。
硬件、内核、依赖变化后应重新测量。

- 固定 1024 面积预算、Turbo 原生 8 步、相同完整提示词；尺寸变化实验单列。
- 只有两张用户实拍图，压力测试用硬链接重复。混合形状测试额外生成横/竖/方形填充版本。
- 每张仍独立编码和去噪，没有跨图片的 embedding 或输出缓存。重复数据会让文件系统缓存变热。
- 这能验证调度、吞吐、内存与稳定性，不能替代一万张不同场景的质量验收。
- 微基准为一轮冷调用和两轮热调用；在不同 GPU 同时筛选，CPU/功耗共享会影响小幅差异。
- 最终参数依据完整八卡服务的相同输入对照；冷任务包含编译，热任务不包含首次编译。
- 所有输出逐文件检查成功数和存储宽高；默认不缩小最终输出。1024 模型输入并不等于保留原始细节。
- 一万张目录扫描不等于一万张 GPU 推理；任何一万张耗时预测都需要标明外推。

## 调研结论及实现选择

### 模型自身的约束

[模型卡](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)和
[官方 pipeline](https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage21)
说明 Turbo 使用检查点的 sample_sigmas 和前缀 KV cache。该快照是 8 步；
随意传 num_inference_steps 不能覆盖 sample_sigmas，本项目没有提供一个实际无效的“减步加速”按钮。

Diffusers 的 image 列表是同一次生成的参考图列表，不能直接当作独立照片 batch。
项目适配层逐图构造条件、正确补齐 mask，再对布局相同的照片批量去噪。
前缀缓存只在该图片的去噪步骤间复用，不跨照片复用。
图像条件与 VLM token/latent 布局耦合，不应擅自把视觉编码器输入缩小一档而保持其它布局不变。

多卡独立实例不需要 GPU 间的梯度同步或张量通信，适合大量独立照片。
模型切分/张量并行的主要价值是单模型容量或单请求延迟；在每卡都能装下模型时，
不假定其吞吐比八个独立实例更高。每卡一份模型，窗口内动态凑批仍保留。

### 编译与精度

[Diffusers 编译指南](https://huggingface.co/docs/diffusers/main/optimization/fp16)
提供全模型和重复块编译。QwenImage21TransformerBlock 在当前源码中标为重复块。
新增 compile_scope=blocks，降低编译范围；compile_dynamic 可尝试减少形状变化导致的重编译。
编译仍可能遇到 Python 分支、不同 token 长度、尾批大小，不能保证只编译一次。
另提供 projections 范围：只编译 attention 的 Q/K/V/输出 Linear 和每块 MLP，
使可变 KV cache、mask 和 segment Python 分支留在编译外。
它的热吞吐可能略低，但冷启动和宽高比变化时的代价可能更小，适合作为稳定性候选。

max-autotune-no-cudagraphs 可搜索矩阵乘内核而不启用 CUDA Graph；
reduce-overhead/max-autotune 在此前实测发生缓存生命周期错误，仍在参数校验阶段拒绝。
首次冷编译可能需要数分钟，不应从测试总耗时中悄悄扣除。
编译产物会缓存在开发机本地，版本/代码/形状变化可使缓存失效。

[TorchAO 集成](https://huggingface.co/docs/diffusers/quantization/torchao)和
[FP8 配置](https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.quantization.Float8DynamicActivationFloat8WeightConfig.html)
支持动态激活/权重 FP8。本机 L40 支持该计算路径。实现仅量化 Transformer 块内 Linear，
输入/输出投影与外部调制保持模型精度；不量化 VLM/VAE。
FP8 的转换与 scale 计算需要编译融合才表现出明显收益。
INT8 weight-only 节省权重显存，但不能据此断言运行更快或可以任意增大 batch。
FP8 是有损计算，需看输出而非只看速度。

lean 编码器跳过模型未使用的 LM logits、所有层 hidden history、生成 KV cache。
本机单图 embedding 与官方路径逐元素完全一致；联合编码两张会发生 BF16 舍入差异。
编码器在整个耗时中占比不大，不把“删除无用工作”直接当作显著端到端加速。

### Attention、显存与 CPU

[Attention backend 文档](https://huggingface.co/docs/diffusers/main/optimization/attention_backends)
允许选择 native/Flash/cuDNN/efficient/Sage 等内核。本项目保留文本的因果 mask，
仅为非因果图像部分选择内核；不能为了加速丢掉 mask。
Hub 内核需要通过官方初始化接口验证并下载，直接传名称不会完成初始化。
Hub 二进制能否下载/支持当前 Torch/CUDA 组合，需要实测，不能仅按产品名称宣称可用。
初次 Hub 测试暴露出缺少官方初始化的实现问题，已修正并重测；
随后 Sage 下载报网络不可达，未获得可用内核，所以不列为已验证加速选项。
FP8 编译 batch3/4 的追加微测没有完成，不将其宣称为已验证的推荐方案。

[VAE 内存指南](https://huggingface.co/docs/diffusers/main/optimization/memory)
区分 tiling 和 slicing。tiling 主要减少空间激活，slicing 主要减少 VAE batch 峰值，
都不意味着 Transformer batch 显存自动降低。新增 tile size/stride 允许调节重叠成本。
CPU offload 换容量但增加传输，在本机模型可驻留的吞吐任务中不是首选。

expandable_segments 用来减少变形状/批大小时的 allocator 碎片。
它可以让此前因碎片失败的配置跑通，但不能越过权重+激活的物理容量。
QKV 融合也可能改变编译/显存行为；本轮没有足够收益支持默认开启。

## 单卡筛选记录

下列秒数均是两轮热调用的中位数，batch=2、1024，除非另注。
它们是筛选数据，微小差异不代表统计显著优势。

| 变量 | 实测结果 | 取舍 |
| --- | --- | --- |
| 官方 stock BF16 | 17.72 秒/2张 | 对照 |
| lean 串行编码 | 18.29 秒/2张 | 无显著吞吐收益，保留可选 |
| lean 两张联合编码 | 18.27 秒/2张 | 无显著吞吐收益，保留可选 |
| BF16 batch 1/2/3 | 0.118/0.114/0.110 张/秒 | 增大 batch 未必更快 |
| BF16 batch 4/6/8 | OOM | 不采用 |
| VAE tile 256/384/512/768 | 17.49/17.79/17.95/17.56 秒 | 保留 256/192 |
| 不分块 VAE，batch2 | OOM | 保留 tiling |
| 不分块 VAE，batch1 | 8.40 秒，峰值 40177 MiB | 显存余量更小 |
| QKV 融合 | 17.90 秒，batch3/4 OOM | 默认关闭 |
| BF16 块编译 | 16.18 秒 | 小幅候选 |
| BF16 动态块编译 | 16.11 秒 | 混合形状再验证 |
| BF16 块编译 autotune | 15.66 秒 | 冷启动成本需折算 |
| FP8 eager | 16.32 秒 | 单独量化收益有限 |
| INT8 weight-only | 18.53 秒 | 更慢，默认关闭 |
| FP8 + 块编译 | 11.69 秒，首次 338.58 秒 | 重点端到端验证 |
| PyTorch CPU threads 1/2/4/8 | 17.44/18.00/18.15/17.91 秒 | 保持 1 |
| native / flash / cuDNN / efficient | 16.90/17.60/18.15/19.00 秒 | 保持 native |
| 开启 allocator 后 BF16 full compile | 15.81 秒，峰值 42029 MiB | 能跑通但余量较小 |
| 开启 allocator 后 FP8 eager batch4 | 35.69 秒/4张 | 增大 batch 未改善吞吐 |
| 开启 allocator 后 BF16 blocks batch1 | 首次408.50秒，热7.90秒/张 | 尾批的新编译代价明显 |
| 开启 allocator 后 BF16 blocks batch3 | 首次217.63秒，热24.33秒/3张 | 热吞吐0.123张/秒 |
| FP8 blocks batch1 | 首次624.75秒，热5.69秒/张 | 比batch2稍快，但冷启动更长 |
| FP8 projections | 首次56.94秒，热12.67秒/2张 | 缩小编译范围 |
| FP8 dynamic projections | 首次33.43秒，热13.03秒/2张 | 继续测试混合形状 |

分辨率独立实验：batch2 的 512/768/1024 分别约 4.49/9.92/18.17 秒；
1280/1536/2048 的 batch2 在该组合下 OOM。低输入预算会影响画面，不能归为同质量加速。

单图 CUDA event 分段测量：总墙钟 9.18 秒，其中 Transformer 7.01 秒（8 次调用），
VLM 0.31 秒，VAE 编码 0.285 秒、解码 0.655 秒。事件测量用于定位瓶颈，不与吞吐测试混用。

一万张硬链接文件扫描：单线程 1.90/1.76 秒，4 线程 3.78 秒，8 线程 3.57 秒。
单张 4000×3000 JPEG 解码约 0.15 秒；质量97保存约0.116秒，质量95约0.104秒，
不足以支持为了几毫秒降低默认输出质量。

## 质量检查方法

同 seed 与相同输入比较 stock、lean 和量化结果，查看两类水印残留以及人体/植被/道路结构。
像素 MAE 是数值差异指标，不是质量评分，也不能证明更接近被覆盖的真实背景。
已测 stock 对 lean 串行 MAE=0；lean batch2 MAE≈0.54；FP8 eager MAE≈7.78（8-bit像素）。
编译和量化可能使生成细节变化，尤其是人体、细小枝叶和长直线，需在真实场景样本上验收。
对两张实拍样例的受限预览检查：stock 和 FP8 编译结果都未见左下信息栏及右下相机水印残留；
草地、泥路及人物细节存在差别。预览不能证明原尺寸下无细微残留，也不能证明被遮挡内容真实还原。
仅回传四张各不超过 64 KiB 的预览，原始推理结果保留开发机。

## 复现与文件位置

所有脚本在 Linux 运行，代码通过 git commit/push → 开发机 pull 同步。
本地 WSL 不安装 GPU 推理依赖；开发机自行 uv sync 下载。
scripts/benchmark-deep.py 为单卡矩阵（lane 指定配置组，gpu 指定空闲卡）；
scripts/profile-stages.py 为阶段事件测量；
scripts/benchmark-scale.py 为一万文件扫描和 JPEG I/O；
scripts/benchmark-throughput.py 为八卡完整服务相同输入比较。
八卡测试独占本项目 GPU worker，请先正常停止服务，不能与生产任务同时跑。

原始实验记录留在开发机 .run/deep-tuning/ 与 .run/throughput/，
每次服务任务还保留 ~/.local/state/watermark-studio/<job>/results.jsonl。
输出图留在开发机，仅小型指标与受限预览返回本地。
全部参数含义见 [启动参数说明](startup-and-tuning.md)。
