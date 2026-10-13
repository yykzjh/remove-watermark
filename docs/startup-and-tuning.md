# 服务启动参数与调优指南

所有示例均在 Linux 开发机的 `~/project/inference/remove-watermark` 执行，使用 uv 环境。

## 1. 完整启动脚本

可执行脚本：[scripts/launch-server.sh](../scripts/launch-server.sh)。它显式列出全部部署字段，并调用负责后台进程管理的 [scripts/start-server.sh](../scripts/start-server.sh)。

```bash
#!/usr/bin/env bash
set -euo pipefail
cd ~/project/inference/remove-watermark

# 第一次安装或更新依赖后执行；平时仅重启无需重复安装。
UV_HTTP_TIMEOUT=600 uv sync --frozen --extra server --no-dev

# 确认当前任务已完成后停止旧服务；新参数仅对新进程生效。
bash scripts/stop-server.sh

bash scripts/start-server.sh \
  --config config/deployment.toml \
  --host 127.0.0.1 \
  --port 8765 \
  --backend qwen \
  --model Qwen/Qwen-Image-2.1-Turbo \
  --model-source modelscope \
  --revision master \
  --cache-dir "$HOME/.cache/watermark-models" \
  --parallelism data \
  --gpus 1,2,3,4,5,6,7 \
  --replicas-per-gpu 1 \
  --load-parallelism 1 \
  --dtype bfloat16 \
  --attention sdpa \
  --no-compile \
  --compile-mode default \
  --offload none \
  --vae-tiling \
  --vae-slicing \
  --kv-cache \
  --tf32 \
  --cpu-threads 4 \
  --allowed-roots "$HOME" /mnt \
  --state-dir "$HOME/.local/state/watermark-studio"

uv run --no-sync python scripts/check-health.py --url http://127.0.0.1:8765
```

以上完整参数已经保存在 launch-server.sh；日常可以简写：

```bash
bash scripts/launch-server.sh
# 尾部参数覆盖脚本中的默认值：
bash scripts/launch-server.sh --gpus 1,2,3 --load-parallelism 2
```

启动脚本输出 starting 只表示后台进程已创建。/health 的 stage=ready 才表示所有模型副本就绪。日志为 .run/server.log；PID 为 .run/server.pid；本次最终参数快照为 .run/deployment.json。任务报告在 --state-dir。

不自动停止已有服务：有显式新参数时返回错误，明确提示参数未应用；无参数调用 start-server.sh 时返回已有进程，兼容桌面启动器。launch-server.sh 总是带显式参数，修改配置后应先完成任务并停止旧服务。停止服务会中断未完成任务。

## 2. 参数优先级与预检查

优先级为：**命令行 > TOML 文件 > 程序内置默认值**。下面表格的值对应完整启动脚本，不代表每个字段的内置默认值。

```bash
# 查看全部可用选项，不启动模型
bash scripts/start-server.sh --help

# 校验并打印最终参数，不创建服务、不加载 GPU
bash scripts/launch-server.sh --gpus 1,2 --print-config

# 原始服务命令，适用于前台调试/系统服务管理
uv run --no-sync watermark-server --config config/deployment.toml \
  --parallelism single --gpus 1 --replicas-per-gpu 1
```

布尔参数均同时提供正反开关，例如 --compile / --no-compile、--kv-cache / --no-kv-cache。不能写 --compile false，应写 --no-compile。重复的标量/布尔参数以最后一次为准；--gpus 和 --allowed-roots 替换整组列表，不是追加。

### 基础与模型参数

| 参数 | 完整脚本值 | 含义 |
| --- | --- | --- |
| --config | config/deployment.toml | 基础 TOML 文件。保留旧的 start-server.sh config/xxx.toml 写法 |
| --host | 127.0.0.1 | 只接受 127.0.0.1 或 ::1；远程桌面通过 SSH 控制隧道访问 |
| --port | 8765 | API 端口，1024–65535。改端口后客户端 URL/SSH 转发端口也要对应修改 |
| --backend | qwen | qwen 执行实际模型；mock 仅测试、原样复制，不去水印。--mock 是后者的别名 |
| --model | Qwen/Qwen-Image-2.1-Turbo | 仓库 ID 或完整本地模型目录；本地目录应包含 model_index.json 等组件 |
| --model-source | modelscope | modelscope 或 huggingface。已有本地模型目录时不使用下载源 |
| --revision | master | ModelScope 分支/提交；HF 通常 main。可固定不可变提交以便复现，master 会变化 |
| --cache-dir | ~/.cache/watermark-models | 由开发机直接下载的模型缓存；本地模型目录不需要下载 |
| --allowed-roots | 用户家目录、/mnt | 允许目录浏览和批处理访问的根目录，空格分隔；带空格的路径须加引号 |
| --state-dir | ~/.local/state/watermark-studio | 每个任务的参数、检测范围、耗时、逐图结果和错误报告目录 |
| --print-config | 未启用 | 打印合并后的 JSON 配置并退出，不启动模型 |

### 并行与资源参数

| 参数 | 完整脚本值 | 含义及调优 |
| --- | --- | --- |
| --parallelism | data | data 为独立图片数据并行；single 为单 GPU 单副本。single 下多个 GPU 或 replicas>1 会直接报错 |
| --gpus | 1,2,3,4,5,6,7 | 逗号分隔 GPU 编号，每个 worker 只看见分配给它的 GPU。当前机器 GPU 0 有其他进程 |
| --replicas-per-gpu | 1 | 每卡完整模型副本数，1–4。增加副本会复制模型显存；当前 L40 实测每副本约 34 GiB，不能据此在 46 GiB 卡上放两个副本 |
| --load-parallelism | 1 | 同时加载模型的 worker 数，1–16。只影响启动阶段，不是批处理并发；可试 2，观察 CPU 内存、磁盘和加载时间 |
| --cpu-threads | 4 | 每个 GPU worker 的 PyTorch CPU 线程数，1–64；不是整个服务总线程上限，也不限制所有第三方库线程 |

工作进程容量 = GPU 数 × 每卡副本数。每批图片的 concurrency 再决定同时占用多少 worker，不能大于已部署容量。一个 worker 依次完成一张图的检测及各个水印区域修复，然后领取下一张，按任务动态分配。

**本项目当前接入 single/data 两种运行方式。** Diffusers 框架还支持模型分片、上下文并行等机制，但没有给本项目尚未实现的 TP/Ulysses/Ring 增加空参数。--gpus 1,2 表示两个完整副本处理不同图片，不表示将单张图片拆到两张卡上。此服务自行创建 GPU worker，不要再套 torchrun 或给 uvicorn 增加 workers，否则会重复创建模型。

### 推理、显存与编译参数

| 参数 | 完整脚本值 | 含义及限制 |
| --- | --- | --- |
| --dtype | bfloat16 | bfloat16 / float16。L40 建议沿用已验证的 BF16；更换精度需重新检查画面和数值稳定性 |
| --attention | sdpa | sdpa / flex。SDPA 为当前实测基线；flex 必须同时 --compile |
| --compile / --no-compile | --no-compile | 编译 transformer；有首次编译开销，不会加速所有检测/文本编码代码 |
| --compile-mode | default | default、reduce-overhead、max-autotune。仅 compile 开启时生效；更激进模式可能增加编译时间/显存 |
| --offload | none | none 全部留在 GPU；model 按组件卸载到 CPU；sequential 更细粒度卸载。通常以传输开销换显存，不能与 compile 同时开启 |
| --vae-tiling / --no-vae-tiling | 开 | VAE 分块，降低高分辨率编解码的峰值显存，可能增加少量计算 |
| --vae-slicing / --no-vae-slicing | 开 | VAE 分片。当前每次解码一张，收益可能有限，不能视为图片批次并行 |
| --kv-cache / --no-kv-cache | 开 | 同一张图去噪过程复用前缀 KV；不是不同图片之间共用内容 |
| --tf32 / --no-tf32 | 开 | 允许适用的 FP32 CUDA 矩阵乘使用 TF32。不会把 BF16 权重改为 TF32，不保证所有算子都受益 |

Turbo 沿用 checkpoint 内的 sample_sigmas（当前 8 步）与 CFG=1。当前没有暴露会与 checkpoint 调度冲突的步数参数，也未接入量化、跨图 prefix-cache 或单次多图片 microbatch。官方 pipeline 的 image 列表是多参考图输入，不能当成独立照片批次。

## 3. 常用启动方案

以下使用 start-server.sh，从 TOML 继承未指定的值。若旧服务在运行，先等待任务结束并停止它。

```bash
# 单卡验证效果
bash scripts/start-server.sh --parallelism single --gpus 1 --replicas-per-gpu 1 \
  --attention sdpa --no-compile --offload none

# 当前七卡吞吐基线（建议）
bash scripts/launch-server.sh

# 只用三张卡；可尝试同时加载两个副本，减少启动等待
bash scripts/launch-server.sh --gpus 1,2,3 --load-parallelism 2

# 显存不足时的单卡方案；该组合尚未在本任务完成性能实测
bash scripts/launch-server.sh --parallelism single --gpus 1 \
  --offload model --no-compile --attention sdpa

# 编译实验：先单卡测量，勿直接将首次编译等待当作稳态吞吐
bash scripts/launch-server.sh --parallelism single --gpus 1 \
  --attention flex --compile --compile-mode default --offload none

# 使用已经下载的模型目录，避免启动时查询远端仓库
bash scripts/launch-server.sh \
  --model "$HOME/.cache/watermark-models/models/Qwen--Qwen-Image-2.1-Turbo/snapshots/master"

# 自定义端口
bash scripts/launch-server.sh --port 8877
uv run --no-sync python scripts/check-health.py --url http://127.0.0.1:8877
```

默认桌面 SSH 启动器转发到远端 127.0.0.1。若使用 --host ::1，需要自行建立 IPv6 对应的隧道并用客户端 --url 连接；常规桌面使用保持 127.0.0.1。

自定义远端端口的 WSL 客户端：
```bash
uv run --no-sync watermark-studio --ssh-host cdu_dev --remote-port 8877 --port 8877
```
先在开发机用指定参数启动服务；桌面客户端只连接该服务，不负责修改已启动实例的部署参数。修改 GPU 数或每卡副本数后，重新打开桌面客户端，以刷新并发数上限。

## 4. 每批图像的全部处理参数

这些参数在提交任务时生效，不要求重新加载模型；它们属于任务配置，不能混入启动服务的部署参数。GUI 提供常用项，API 可以设置全部字段。

| options 字段 | 默认值 | 含义 |
| --- | --- | --- |
| concurrency | API 为 1；GUI 为服务容量 | 同时处理的图片数，1–64 且不超过 worker 容量；七卡建议先 7 |
| preset | auto | auto 每图自动检测；gray/blue 固定样例位置模板；custom 使用 regions。批量不同布局用 auto |
| regions | [] | custom 模式必填，最多 8 个对象；kind 为 panel/logo/custom，box 为归一化 x0,y0,x1,y1 |
| detection_resolution | 1280 | 自动检测图的长边上限，512–2048、32 的倍数；更大更利于小字定位，但增加检测开销 |
| remove_logo | true | 同时清理相机标识及附属文字，无论位于哪个角落 |
| resolution | 1024 | 每个修复裁剪的生成分辨率参数，256–2048、32 的倍数；不是最终整图输出尺寸 |
| context | 0.6 | 水印区域周围的上下文扩展比例，0.1–2；给自然补全提供场景信息 |
| feather | 12 | 原图像素单位的外侧过渡宽度，0–64；区域内部不做半透明回贴 |
| seed | 42 | 0–4294967295；与相对路径及区域序号组合，固定测试条件，但不保证跨硬件/版本位级复现；目前通过 API 指定 |
| overwrite | false | 跳过已有输出；开启则重新生成并原子替换。调参对比应换输出目录或开启覆盖 |
| extra_prompt | 空字符串 | 最多 2000 字符的场景补充指令，与项目自然补全提示词组合 |

请求本身还需 input_dir 和 output_dir，均为开发机上的绝对路径或可展开的家目录路径，二者不能相同或相互包含。路径及 task options 完整示例见 [config/job.example.json](../config/job.example.json)：

```bash
# 在开发机编辑示例中的输入/输出路径，再提交小型 JSON 请求
curl --fail -H 'Content-Type: application/json' \
  --data-binary @config/job.example.json http://127.0.0.1:8765/jobs

# 常用参数也可由批处理命令指定
uv run --no-sync watermark-batch /home/zjh/images /home/zjh/images-clean \
  --concurrency 7 --preset auto --detection-resolution 1280 --resolution 1024
```

## 5. 调优顺序与观测

1. **先质量后速度**：在开发机准备少量有代表性的真实照片，涵盖人体遮挡、小字、多水印和无水印图。先固定 seed、auto、1280 检测、1024 修复、BF16、SDPA、单副本。
2. **把并发填满**：七个独立 GPU 对应 concurrency=7。增大 HTTP 线程数或将 concurrency 写成 64 不会多出 GPU 工作进程，也不会自动增加吞吐。
3. **分辨率对比**：修复 768/1024/1536 分别抽样。小字漏检时先提高 detection_resolution；提高生成分辨率不会弥补检测漏框。
4. **显存不足**：先保持每卡一个副本、降低修复分辨率并保持 VAE tiling；必要时试 model offload。只降低任务 concurrency 不会卸载已经驻留的模型副本。
5. **编译单独实验**：先单卡 flex+compile，记录首次编译和热身后的吞吐。自动检测产生不同裁剪形状，可能重复编译；确认有收益再扩到多卡。此方案尚未完成实测。
6. **加载速度**：load_parallelism 从 1 试到 2，观察 CPU 内存和存储吞吐。它不影响服务 ready 后的任务并发上限。
7. **测真实数据**：使用足够多图片，分别记录图/分钟、逐图耗时、水印区域数、错误率和显存峰值；区分磁盘 I/O、自动检测与生成耗时。results.jsonl 含 detection_seconds、seconds、regions 和 action。

```bash
nvidia-smi
uv run --no-sync python scripts/check-health.py
tail -n 40 .run/server.log
```

前次实测：七卡处理 8 张合成测试图约 14.31 秒；单卡 1024 单区域生成记录 8.15 秒。这些用于链路验收，不能直接作为真实图片吞吐承诺；当时显存快照约 34 GiB/卡，也不是峰值。详见 [部署验证记录](deployment-validation.md)。

## 6. 下载和代码同步

模型与依赖均由开发机直接下载。网络较慢时可用 UV_HTTP_TIMEOUT=600 延长 uv 请求超时。ModelScope SDK 支持 MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS（单文件范围下载线程，默认 1；此前下载用 8），它是下载环境变量，不是 GPU 并行参数；可在启动脚本前设置，过高可能增加下载重试。

代码继续使用本地 git commit/push，开发机 git pull --ff-only。通过 SSH 只运行普通长度命令、查看有限文本输出，不传图片、模型或源码压缩包。

## 参考

- [Diffusers 分布式推理](https://huggingface.co/docs/diffusers/main/training/distributed_inference)：数据并行、模型放置与其他并行机制。
- [QwenImage21Pipeline](https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage21)：多参考输入、KV cache、Flex attention 与采样参数。
