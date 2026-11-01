# 新版批量 / 单张界面

见 [工作流与分辨率说明](docs/workflows.md)。

> 动态凑批版本：默认 GPU 0–7，每卡一个模型。concurrency 现为全局在途图片数；
> 每卡 batch_size 独立配置。所有 bool 未指定为 false。完整启动、参数和实测见
> [启动调优指南](docs/startup-and-tuning.md) 与 [整图吞吐调优](docs/deep-tuning.md)。
> 默认输出保持原文件存储宽高及 EXIF 方向，也可指定完整输出尺寸。
> 批量整图提示词包含左右两类水印，无需设置区域或 remove-logo。

# Watermark Studio

使用 Qwen-Image-2.1-Turbo 在 Linux 开发机批量修复工地/户外照片的数字水印。提供 Linux 原生 PySide6 桌面窗口、GPU 推理服务与命令行。环境统一由 uv 管理。

图像由用户提前放在开发机。服务在开发机深层递归镜像输入目录（含空目录和原始名称），对有效图像去水印；非图像文件、损坏/多帧和未支持格式原样复制，并按相同相对路径保留。图像默认保持原始宽高和文件名，复制统计与失败详情可在客户端查看。桌面窗口通过少量 JSON 控制请求操作；单张框选时按需请求不超过 64 KiB 的预览，不自动下载原图。

针对整图处理，默认使用中文中性材质保真neutral提示词；scene原图材质约束仅显式实验启用。历史关闭原图保护的32张同源对照未达到严格验收标准，未放宽检验；全文、取舍及复现见[提示词针对性调优](docs/whole-prompt-tuning-2026-11-01.md)。

当前推荐批量和单张自动模式采用**整张原图输入模型 + 自动逐图原图保护**：显式勾选原图保护，仅合成自动识别水印处的修复结果，其余保留原像素；无需手工框选，不裁切模型输入，生成返工同样保持整图输入。单张自定义区域保留局部修补。关闭保护时仍支持完整生成结果直接输出；历史直接整图和裁切策略的成绩分别记录。详见[整图保护验证](docs/whole-protected-scale-2026-11-01.md)。 最新冻结1024张首轮自动844通过，有限返工后905通过、119待复核；5张L40完整耗时4324.338秒。此为自动检查结果，不是人工准确率，仍未达到接近全部首轮通过。

最新修补与检验配方：`bash scripts/launch-throughput-quality.sh`，默认独立 Qwen3-VL 检验、调和明暗衔接及局部阴影/纹理检查；未通过项输出可见依据和位置，逐张回传下一轮返工。客户端须显式启用原图保护、检验、自动返工；见[流程](docs/quality-review.md)和[两张待复核图的修复实测](docs/review-repair-2026-11-01.md)。

历史裁切首轮精细策略：自动补查细小防伪码，按原图实际材质生成各覆盖层，保住其余可见内容；仅保留 CLI/API --first-pass-mode 用于历史对照，客户端不展示这些裁切策略。首轮可能包含多个局部模型调用。参数、完整对照与限制见[首轮质量报告](docs/first-pass-quality-2026-11-01.md)。

本轮去水印/检验对抗调优：逐行补查完整防伪码，小预算可独立凑批；继续采用4B主检验，8B/NF4作为单独加载的实验配置。8B与编辑侧基础权重相同，禁止双模型自证；仍有已知误拒和淡暗框漏检，不能保证千张无缺陷；开发对照、千张留出验证及耗时见[对抗调优报告](docs/duel-quality-2026-11-01.md)。

历史第二轮首轮优化新增可选CPU矩形明暗检查、受限残留框重检和客户端`--context`；当时质量配方采用大输入batch1/小输入batch2；当前整图保护配方采用1024输入batch2、大于1024输入batch1，默认8卡。已知对照仍有误拒和漏检，不能保证千张全部正确一次通过；完整过程与新留出验证见[第二轮报告](docs/first-pass-round2-2026-11-01.md)。

编辑侧可提出实物文字纠正说明，由独立验证器按原图证据复核；显式启用 `--quality-cross-review`，明暗/纹理疑点不能豁免。见[争议复核实测](docs/cross-review-2026-11-01.md)。

## 启动参数与调优

完整参数脚本：[scripts/launch-server.sh](scripts/launch-server.sh)。支持通过启动命令选择 single/data 并行、GPU 列表及全部部署选项；先用 --print-config 查看最终设置。详细参数表、完整脚本、任务配置和调优顺序见 [启动与调优指南](docs/startup-and-tuning.md)。

### 已实测的吞吐配方

追加部署组合实测：相同257张混合输入，FP8动态投影编译的热耗时从227.745秒降至210.115秒，
减少7.7%；采用batch1、关闭VAE tiling，峰值分配为33501 MiB，输出仍保持各自原图尺寸。
完整启动脚本：**bash scripts/launch-throughput-untiled.sh**（测于1024输入预算）。
参数、失败候选、26个端到端任务及数据见[第二轮部署调优报告](docs/tuning-second-pass.md)。
更高输入预算或显存紧张时可追加 --vae-tiling，或选择下面保留分块的配方。

以下保留第一轮结果：

本机8张L40、输入1024、输出原尺寸，128张相同输入的热任务：
原BF16配置154.2秒，FP8分块编译108.8秒（耗时减少29.5%）；
FP8动态投影编译116.7秒。混合形状1025张连续测试全部成功，约14分58秒，输出尺寸全部正确。

- 混合宽高比配方：`bash scripts/launch-throughput.sh`
- 固定形状、大量连续任务：`bash scripts/launch-throughput-fixed.sh`
- 保留BF16：`bash scripts/launch-server.sh --compile --compile-dynamic --prompt-encoder lean`

切换前先等任务结束并执行bash scripts/stop-server.sh。所有配方默认8卡，布尔开关仍须显式指定。
FP8会改变生成细节；首次编译和新形状可能额外耗时。测试使用两张实拍样例及其形状变体重复构造，
不等同于一万张不同实拍图的质量验收。完整参数、数据、限制和复现命令见[深度调优报告](docs/deep-tuning.md)。

## 环境与安装

需要 Linux、Python 3.12、NVIDIA CUDA 驱动；权重和依赖由开发机直接下载。项目的 uv.lock 固定依赖版本及 Diffusers 源码归档提交，Python 包使用已验证可达的阿里云 PyPI 镜像并校验下载哈希。模型是新发布的 Turbo 版本，需要包含 sample_sigmas 支持的 Diffusers 与 transformers>=5.17。

```bash
# 初次安装 uv（已安装则略过）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 开发机
mkdir -p ~/project/inference
cd ~/project/inference
git clone https://github.com/yykzjh/remove-watermark.git
cd remove-watermark
UV_HTTP_TIMEOUT=600 uv sync --frozen --extra server --extra quantization --no-dev
bash scripts/launch-server.sh
uv run --no-sync python scripts/check-health.py
```

当前配置使用 Qwen 官方 ModelScope 仓库（开发机访问 Hugging Face 超时）。可设置 model_source="huggingface"、revision="main" 切换来源；生产复现建议将 revision 固定为仓库提交。

首次启动先下载权重，再加载 GPU 工作进程；/health 的 stage 依次为 downloading、loading、ready 或 failed。服务日志在 .run/server.log。主批处理窗口仅在 ready 后展示。编译采用 PyTorch 的延迟编译，第一张图片仍可能有明显编译开销。

## 桌面启动

**在有 Linux 图形桌面的开发机上：**

```bash
uv sync --frozen --extra server --extra quantization --extra desktop
uv run --no-sync watermark-studio
```

这会先启动本机服务，模型就绪后打开桌面批处理窗口。若服务已由脚本启动，使用 `watermark-studio --url http://127.0.0.1:8765`，避免重复启动。

**在 WSL2 / WSLg 中使用开发机服务：**

```bash
uv sync --frozen --extra desktop
uv run --no-sync watermark-studio --ssh-host cdu_dev
```

需要 WSL 内的 ssh cdu_dev 免密连接。启动器调用开发机 scripts/start-server.sh，并建立主要传控制信息、按需传小型预览的 SSH 隧道。默认项目目录 ~/project/inference/remove-watermark，可用 --remote-dir 调整。目录浏览与输入/输出路径都指向开发机，不是 WSL 本地路径。默认每 5 秒查询一次简短状态；可用 --poll-seconds 增大间隔。

GUI 需要 Linux Qt 系统库；Ubuntu 常用依赖：
```bash
sudo apt-get install libegl1 libgl1 libxkbcommon-x11-0 libxcb-cursor0 \
  libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 \
  libxcb-shape0 libxcb-xkb1 fonts-noto-cjk
```

## 批处理

导航栏提供“批量去除水印”和“单张去除水印”。批量默认整图处理，无需选择水印位置。
单张提供“自动识别水印位置”（整图模型编辑）和“自定义区域”（鼠标多框/数值坐标）两种模式。
模型输入分辨率与完整输出尺寸分开设置，输出默认原尺寸；调整输出使用 Lanczos 重采样。
支持暂停、继续与取消；已提交 GPU 推理完成后停止派发。关闭客户端不会取消开发机任务，重新连接可查看当前任务。

输入/输出必须互不包含；不跟随目录/文件符号链接。默认跳过已有输出，实现重启后继续处理；需要重新修复已有结果时勾选覆盖。逐图错误互不影响，输出采用原子写入。保存格式、名字和目录结构与输入对应，非图像原样复制，空目录保留。

命令行：
```bash
uv run --no-sync watermark-batch /home/zjh/photos /home/zjh/photos-clean \
  --concurrency 32 --input-resolution 1024
```

每次任务在 ~/.local/state/watermark-studio/<job-id>/ 记录 summary.json、results.jsonl 和 invalid.json。summary 包含模型 revision、参数、计数与时间。进程重启后不自动恢复队列；使用同一目录且不勾选覆盖重新提交，已成功输出会跳过。

## 模型部署与吞吐参数

非布尔值可修改 TOML，bool 必须在启动命令显式传入。完整 launch-server.sh 使用 GPU 0–7，BF16，每卡一个完整副本，CFG=1，官方 Turbo 8 步调度，启用 KV cache、VAE tiling/slicing 和 TF32。

| 参数 | 作用与限制 |
| --- | --- |
| gpus | 物理 GPU 编号列表；每张卡独立处理图像，实现数据并行 |
| replicas_per_gpu | 每卡完整模型副本数；默认 1，增加会成倍占显存 |
| load_parallelism | 同时加载的副本数；本机实测采用 8，限制 CPU 内存与磁盘峰值 |
| attention | sdpa 或 flex；flex 必须配合 compile |
| compile / compile_mode | 编译 transformer；第一张和新形状可能重新编译 |
| kv_cache | 重用当前图像各去噪步的前缀 KV；不是跨图片复用 |
| offload | none / model / sequential；省显存通常会减吞吐，不与 compile 组合 |
| vae_tiling / vae_slicing | 降低 VAE 峰值显存 |
| dtype / tf32 | BF16/FP16 及矩阵乘精度设置 |
| cpu_threads | 每个工作进程的 PyTorch CPU 线程数 |
| allowed_roots | 桌面可访问的目录根，默认用户家目录与 /mnt |

默认启用全部 8 张 GPU，每卡一个模型。独立图片由适配层分别编码，再按相同推理尺寸动态凑批；
不会把不同照片误当作共享参考图。concurrency 表示全局在途原图数，本机推荐 32，
单卡 batch_size 默认 2。OOM 拆批、候选窗口和尺寸分桶详见完整指南。
当前未接入 TP/Ulysses/Ring 等单图计算并行。Flex 在当前固定版本实测编译失败，
不能作为生产启动方案；SDPA 编译的速度、显存与动态形状代价见实测记录。

服务默认沿用 checkpoint 中的 8 步 sample_sigmas。没有提供会被该调度忽略的 num_inference_steps 滑块，也不默认打开 CFG 或额外负向提示词。

```bash
bash scripts/stop-server.sh
bash scripts/launch-server.sh
```

## 质量与提示词

见 [docs/prompts-and-quality.md](docs/prompts-and-quality.md)。提示词覆盖不同位置和样式的数字水印，以及人体、地表、植物和工程结构的自然重建。批量默认整图编辑；勾选原图保护后自动定位覆盖层，仅合成这些位置。局部自动模式也逐图定位；无检测时复制原图，异常检测明确记录失败。真实测量读数、证书和现场文字应保留，检测仍需结合检验与抽样验收。

模型依照 [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) 发布；项目代码许可证不替代模型许可证。

## 开发与代码同步

全部安装、运行和测试在 WSL2 或开发机完成。Windows 只作为共享源码目录。建议将 WSL 环境放到 Linux 文件系统：
```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.cache/uv-envs/remove-watermark"
uv sync --frozen --extra desktop
uv run --no-sync pytest
uv run --no-sync ruff check src tests
```

代码更新统一使用本地 git commit、git push，随后开发机在 ~/project/inference/remove-watermark 执行 git pull --ff-only，再 UV_HTTP_TIMEOUT=600 uv sync --frozen --extra server --extra quantization --no-dev 并重启服务。不得通过 SSH 传输源码包或图像大数据。

测试使用明确标记的 mock 后端，它只复制图像来验证队列、目录、UI 和错误处理，不代表模型推理效果。

部署实测、八卡并发验收及当前限制见 [docs/deployment-validation.md](docs/deployment-validation.md)。

去水印后可启用原图/输出质量检验，检查水印残留、修补自然度与内容一致性。参数、记录和局限见 [质量检验说明](docs/quality-review.md)。

自动返工、扩大抽样与性能实测见 [检验说明](docs/quality-review.md)、[框架迁移](docs/framework-migration.md)。


实验开关 `--prefill-overlays` 仍默认关闭。用于整图自动保护时，它只根据原图安全边界外
的颜色，在已验证的数字信息卡核心内生成平滑的条件初始化；输入模型仍是一张完整照片，
不裁切推理，不增加生成次数。原图保留作合成和检验参照。核心与检测范围不匹配时跳过，
不会对未确认的实物文字做初始化。该变体须先做同图质量和吞吐对照，不能把平滑补色当成最终修复。
整图条件初始化同时需要 `--protection-core-feather` 提供原检测核心；相机小字等未初始化的覆盖层仍由一次整图生成清除。


源检测安全：启用现有 --protection-physical-vet 时，对于室外单个检测框覆盖整图至少35%的异常情况，
先用两份不同权重各重新定位一次；只接受位置高度一致、更精确且经双方确认属于数字覆盖层的范围。
若几何或载体判断不一致，原大范围保留原像素并进入复核，不能盲目重绘设备标牌、手持验收牌或人体。
该检查不假设水印位于固定角落，不裁切模型推理输入，也不以大框截断水印。

稀有无字争议实验增加 uv extra quality-ocr（Python3.11/3.12），CPU执行文字存在检测，
发现任意字形或检测错误即保留复核；空检测结果从不单独构成通过。实际转录文字不存储、不进入提示词。
此功能目前只用于显式实验脚本。3.5第三检查点的实验必须同时启用 --text-free-ocr-guard，
因为原生抽查已否定单靠双模型语义“无字”证明的可靠性。


实验开关--prefill-original-semantics需与--prefill-overlays一起显式指定，用于整图自动保护。
它将完整原图与正常整图去水印提示词提供给语义编码器，只初始化生成侧VAE图像参照中的可靠大卡片。
不增加图像数量、语义编码次数或Qwen生成次数；初检仍对比未修改的完整原图。
它用于测试初始化是否丢失半透明卡片下已有的人体/物体线索，默认False，未实测成功前不推荐生产使用。
