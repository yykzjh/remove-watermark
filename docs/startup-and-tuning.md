> 当前界面与分辨率语义见 [工作流说明](workflows.md)。下文区域批量测试属于历史路径。

# 服务启动、动态凑批与调优

在 Linux 开发机 ~/project/inference/remove-watermark 执行，使用 uv。
完整可执行脚本：[launch-server.sh](../scripts/launch-server.sh)。
实测结果见 [batch-tuning-results.md](batch-tuning-results.md)。

## 启动

```bash
uv sync --frozen --extra server --no-dev
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
| --batch-size | 默认 2，范围 1–16；每 GPU 单次模型调用最多编辑区域数 |
| --batch-window | 默认 4，范围 1–8；每 worker 候选图片上限=batch_size×batch_window |
| --batch-bucketing | bool，按相同推理宽高分桶；完整脚本显式启用 |
| --oom-split-batch | bool，CUDA OOM 递归拆批；单图仍失败则报告且不写半成品 |
| --load-parallelism | 默认 8，1–16；仅控制同时加载模型的进程数，不控制推理并发 |
| --dtype | bfloat16 / float16；BF16 为默认，精度变更需重新验收画面 |
| --attention | sdpa / flex；当前固定依赖 Flex 实测编译失败，启动校验会拒绝，使用 sdpa |
| --compile | bool，编译 transformer；首次和新形状有额外编译开销 |
| --compile-mode | default / reduce-overhead / max-autotune；后两种当前实测 CUDA Graph 错误，开启编译时校验拒绝；使用 default |
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
- concurrency=32/64：提供更多凑批候选，但每次模型调用仍最多 batch_size 个区域。
- 最大在途容量 = W × batch_size × batch_window；超出时 API 拒绝，GUI 限制输入。
- 一图多个水印时会生成多个修复区域，因此实际 batch 区域数不等于原图数。
- 目录扫描完成后才处理，候选队列已知；尾批立即执行，不人为增加 batch-wait-ms 延迟。
- 尺寸不兼容或图片不足时用较小 batch，不为凑批强制拉伸原图。

每张图独立检测和文本编码，再将相同推理尺寸的区域组成去噪 batch。
Diffusers 原生 image 列表仍是共享参考图；项目适配层独立编码各图并验证 token 布局。
不同长度提示词使用正确 padding mask；大块 padding attention 按样本剔除无效 key，
投影和 MLP 仍使用批维度。该优化不跨图混合参考像素。

## 原图尺寸与质量

最终输出始终在原图画布上局部合成，**输出存储尺寸与原文件严格一致，同时保留 EXIF 方向**。
4000×3000、竖图、奇数宽高和非 32 倍数均支持。EXIF 旋转照片内部按正常观看方向处理，保存前反变换，存储宽高和正常显示方向均与原图一致。
resolution 是局部生成预算，不是最终输出宽高。PNG 未修改区域像素保持一致；
JPEG 重新编码会有压缩差异。原图不覆盖，输出目录保持相对结构。
生成内容只是自然补全，无法保证还原被水印遮挡的真实细节。

## 全部任务参数

API 请求还需 input_dir、output_dir，均为开发机目录，不能相同或相互包含。
完整示例：[job.example.json](../config/job.example.json)。

| options 字段 | 默认与作用 |
| --- | --- |
| concurrency | 32，1–1024 且不超过缓冲容量；GUI 推荐 W×batch_size×min(2,batch_window) |
| preset | auto；gray/blue 为固定样例模板，custom 使用 regions；常规批量用 auto |
| regions | []，custom 必填，最多 8 个；kind=panel/logo/custom，box 为 0..1 坐标 |
| detection_resolution | 1280，512–2048，32 倍数；检测预览长边，不改变输出大小 |
| remove_logo | false；**需要完整去除今日水印及附属块时显式勾选或传 true** |
| resolution | 1024，256–2048，32 倍数；局部生成的面积分辨率预算 |
| context | 0.6，0.1–2；周围场景上下文扩展比例 |
| feather | 12，0–64；原图像素单位的外侧过渡宽度，区域内部完整替换 |
| seed | 42；结合相对路径与区域序号；BF16 批量与单图可能有少量数值差异 |
| overwrite | false；不开启时跳过已存在输出 |
| extra_prompt | 空，最多 2000 字符；场景补充提示词 |

```bash
uv run --no-sync watermark-batch /home/zjh/images /home/zjh/images-clean \
  --concurrency 32 --input-resolution 1024
```

## 调优与复现

先固定质量预算：自动检测 1280、生成 1024、BF16、每卡一副本。
分别测 batch、concurrency、KV、VAE、线程、编译、offload，记录吞吐和显存峰值。
不通过漏清理 logo、降低检测分辨率或跳过文件来制造吞吐提升。
路径、端口、seed、区域、提示词等参数不是可以独立按速度最大化的开关。
load_parallelism 仅比较启动时间，不算入热身后的推理吞吐。
真实照片由用户准备；当前合成测试不能代替人体/植被等细节的真实画质验收。

```bash
# 单卡脚本须选择空闲 GPU；不可与同卡服务副本同时运行
uv run --no-sync python scripts/benchmark-batches.py --gpu 0 --profile optimized
uv run --no-sync python scripts/benchmark-batches.py --gpu 0 --profile compile
uv run --no-sync python scripts/benchmark-batches.py --gpu 0 --profile offload

# 八卡服务 ready 且空闲时执行
uv run --no-sync python scripts/benchmark-service.py --mode custom
uv run --no-sync python scripts/benchmark-service.py --mode auto
```

测试图片和输出保留在开发机 .run/tuning；仅小型结果摘要通过 SSH 返回。
每图 results.jsonl 包含 detection_seconds、window_seconds、window_images、batch_sizes、
peak_allocated_mib、peak_reserved_mib、oom_retries、output_size。
窗口指标重复记录到各图，不能按图片简单求和。PyTorch 峰值不含其他进程/CUDA 上下文全部开销。

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
