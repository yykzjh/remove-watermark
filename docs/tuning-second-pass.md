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

第一阶段：8 张 L40 分别运行 16 个候选配置，每配置首次调用 + 4 次热调用；
覆盖 tensor/row、Conv 布局、cuDNN、batch 1/2/3、blocks/projections、autotune。
微基准跨 GPU 同时运行，有 CPU/功耗竞争，只用于候选筛选。

第二阶段：停止本项目空闲正式服务，独占其八个 GPU worker 做完整服务对照；
每次加载时间单列，同一进程重复任务。原本已有的外部 GPU0 进程不改动。
测试结束恢复正式服务。比较与验收统计将记录在本文件和对应 JSON 中。

仅有用户提供的两张实拍图。混合形状压力集由原图硬链接重复及填充变体构成，
可以检验执行稳定性和调度，但不能替代上万张不同真实照片的画质验收。
“冷”指新进程首次推理，保留已有磁盘编译缓存。热性能不代表首次部署总耗时。
