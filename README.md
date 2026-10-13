# Watermark Studio

使用 Qwen-Image-2.1-Turbo 在 Linux 开发机批量修复工地/户外照片的数字水印。提供 Linux 原生 PySide6 桌面窗口、GPU 推理服务与命令行。环境统一由 uv 管理。

图像由用户提前放在开发机。服务在开发机递归读取输入目录，过滤非图像及损坏/多帧图像，并按相同相对路径写入输出目录。桌面窗口通过少量 JSON 控制请求操作；没有图片上传/下载 API。

## 环境与安装

需要 Linux、Python 3.12、NVIDIA CUDA 驱动；权重和依赖由开发机直接下载。项目的 uv.lock 固定依赖版本及 Diffusers Git 提交。模型是新发布的 Turbo 版本，需要包含 sample_sigmas 支持的 Diffusers 与 transformers>=5.17。

```bash
# 初次安装 uv（已安装则略过）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 开发机
mkdir -p ~/project/inference
cd ~/project/inference
git clone https://github.com/yykzjh/remove-watermark.git
cd remove-watermark
uv sync --frozen --extra server --no-dev
bash scripts/start-server.sh
uv run --no-sync python scripts/check-health.py
```

首次启动先下载权重，再加载 GPU 工作进程；/health 的 stage 依次为 downloading、loading、ready 或 failed。服务日志在 .run/server.log。主批处理窗口仅在 ready 后展示。编译采用 PyTorch 的延迟编译，第一张图片仍可能有明显编译开销。

## 桌面启动

**在有 Linux 图形桌面的开发机上：**

```bash
uv sync --frozen --extra server --extra desktop
uv run --no-sync watermark-studio
```

这会先启动本机服务，模型就绪后打开桌面批处理窗口。若服务已由脚本启动，使用 `watermark-studio --url http://127.0.0.1:8765`，避免重复启动。

**在 WSL2 / WSLg 中使用开发机服务：**

```bash
uv sync --frozen --extra desktop
uv run --no-sync watermark-studio --ssh-host cdu_dev
```

需要 WSL 内的 ssh cdu_dev 免密连接。启动器调用开发机 scripts/start-server.sh，并建立只传控制信息的 SSH 隧道。默认项目目录 ~/project/inference/remove-watermark，可用 --remote-dir 调整。目录浏览与输入/输出路径都指向开发机，不是 WSL 本地路径。默认每 5 秒查询一次简短状态；可用 --poll-seconds 增大间隔。

GUI 需要 Linux Qt 系统库；Ubuntu 常用依赖：
```bash
sudo apt-get install libegl1 libgl1 libxkbcommon0 libxcb-cursor0 libxcb-xinerama0 fonts-noto-cjk
```

## 批处理

窗口可设置输入/输出目录、并发图片数、修复区域分辨率、蓝/灰水印预设、自定义区域、上下文扩展比例、过渡边缘和补充提示词。支持暂停、继续与取消；已提交的 GPU 推理会完成后停止派发新图。关闭远程客户端不会取消开发机任务，重新连接可查看当前任务。

输入/输出必须互不包含；不跟随目录/文件符号链接。默认跳过已有输出，实现重启后继续处理；需要重新修复已有结果时勾选覆盖。逐图错误互不影响，输出采用原子写入。保存格式、名字和目录结构与输入对应，非图像不复制，空目录保留。

命令行：
```bash
uv run --no-sync watermark-batch /home/zjh/photos /home/zjh/photos-clean \
  --concurrency 1 --preset auto --resolution 1024
```

每次任务在 ~/.local/state/watermark-studio/<job-id>/ 记录 summary.json、results.jsonl 和 invalid.json。summary 包含模型 revision、参数、计数与时间。进程重启后不自动恢复队列；使用同一目录且不勾选覆盖重新提交，已成功输出会跳过。

## 模型部署与吞吐参数

编辑 config/deployment.toml 后重启服务。默认使用 GPU 1，BF16，单副本，CFG=1，官方 Turbo 8 步调度，启用 KV cache、VAE tiling/slicing 和 TF32。

| 参数 | 作用与限制 |
| --- | --- |
| gpus | 物理 GPU 编号列表；每张卡独立处理图像，实现数据并行 |
| replicas_per_gpu | 每卡完整模型副本数；默认 1，增加会成倍占显存 |
| load_parallelism | 同时加载的副本数；默认 1，限制 CPU 内存与磁盘峰值 |
| attention | sdpa 或 flex；flex 必须配合 compile |
| compile / compile_mode | 编译 transformer；第一张和新形状可能重新编译 |
| kv_cache | 重用当前图像各去噪步的前缀 KV；不是跨图片复用 |
| offload | none / model / sequential；省显存通常会减吞吐，不与 compile 组合 |
| vae_tiling / vae_slicing | 降低 VAE 峰值显存 |
| dtype / tf32 | BF16/FP16 及矩阵乘精度设置 |
| cpu_threads | 每个工作进程的 PyTorch CPU 线程数 |
| allowed_roots | 桌面可访问的目录根，默认用户家目录与 /mnt |

config/deployment-throughput.toml 提供 GPU 1..7 七副本 + flex/compile 的吞吐配置，需先在单卡测量显存。并发数受已部署副本数限制。官方 Diffusers 的 image 列表代表多参考输入，因此本项目不会把不同图片误传为一个图像批次。此实现提供独立 GPU 数据并行；没有声称实现 TP/Ulysses/Ring 等分布式单图并行。

服务默认沿用 checkpoint 中的 8 步 sample_sigmas。没有提供会被该调度忽略的 num_inference_steps 滑块，也不默认打开 CFG 或额外负向提示词。

```bash
bash scripts/stop-server.sh
bash scripts/start-server.sh config/deployment-throughput.toml
```

## 质量与提示词

见 [docs/prompts-and-quality.md](docs/prompts-and-quality.md)。实际提示词在 src/watermark_studio/prompts.py，专门约束蓝底白字、灰底黑字、户外地表、人物衣物与道路边线的局部修补。auto 是两种样例的模板分类器，不能保证检测任意水印。请先用代表样本确认区域和修复效果。

模型依照 [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) 发布；项目代码许可证不替代模型许可证。

## 开发与代码同步

全部安装、运行和测试在 WSL2 或开发机完成。Windows 只作为共享源码目录。建议将 WSL 环境放到 Linux 文件系统：
```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.cache/uv-envs/remove-watermark"
uv sync --frozen --extra desktop
uv run --no-sync pytest
uv run --no-sync ruff check src tests
```

代码更新统一使用本地 git commit、git push，随后开发机在 ~/project/inference/remove-watermark 执行 git pull --ff-only，再 uv sync --frozen --extra server --no-dev 并重启服务。不得通过 SSH 传输源码包或图像大数据。

测试使用明确标记的 mock 后端，它只复制图像来验证队列、目录、UI 和错误处理，不代表模型推理效果。
