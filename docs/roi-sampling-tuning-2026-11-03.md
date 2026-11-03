# ROI 首轮去水印调优，2026-11-03

当前状态：首个冻结千张候选已证明不能达标并提前停止，继续优化中。没有提交全量 38,319 张生产任务。

## 验收口径

目标为冻结 1,000 张图像的首轮自动验收通过率至少 90%，不包含返工收益。
每个被定位的水印 ROI 在首轮只生成一个候选；一张照片可能有多个 ROI。
保留独立 4B 检验、残留扫描、自然度/内容一致性、边界、纹理、矩形相位和原图实物保护。
不得用处理成功数替代质量通过数，不降低阈值，不排除待复核或失败项目。
原图字节、输出文件名、完整多层目录和原始像素尺寸均需独立核对。

## 已完成的开发对照

以下 64 张来自同一固定开发选集，随机种子 42；同组输入和检测区域完全一致。
输入预算为大 ROI 1024、小标记 512，输出为原图尺寸。无自动返工或检验申诉。

| 方案 | 首轮通过 | 待复核 | 说明 |
| --- | ---: | ---: | --- |
| 原始 ROI、模型存储调度 | 53/64 | 11 | 本轮匹配基线 |
| 覆盖层预清除，两个编码分支均看初始化 | 50/64 | 14 | 不采用 |
| 原图语义 + 预清除 VAE 条件 | 54/64 | 10 | 恢复 4、退化 3，收益不稳健 |
| 精简中文提示词 | 43/64 | 21 | 不替换原有详细提示词 |
| 原始 ROI、全局 12 步均匀调度 | 56/64 | 8 | 恢复 4、退化 1；不同区域仍需区分 |

12 步分片的单 GPU 处理耗时合计 1136.181 秒，基线为 993.314 秒。
这包含不同分片的暖机差异，不能直接当作四卡服务端到端吞吐比较。
两张原生像素诊断拼图确认了具体材质和边界改善，也保留了真实退化的拒绝。

另一个 128 张选集上，全局 12 步为 114/128、576.515 秒；
原存储调度的历史同组服务基线为 115/128、581.981 秒。
因此全局增加步数不作为默认。分析这些开发结果后，128 张不再是后续路由的独立盲测。

16 张局部消融：原始 14/16，取消核心边缘羽化 13/16，
额外原图材质描述 12/16，预清除语义分离且上下文 0.85 为 13/16。
失败实验保留为显式研究工具，没有加入推荐服务。

## 源图面积路由

部署参数：

- `--uniform-steps 0`：全局保留模型的原存储 Turbo 调度。
- `--roi-steps 12`：仅对满足源图面积条件的 ROI 使用 12 步均匀调度。
- `--roi-step-min-area 0.14`：已定位编辑掩码在完整原图中的面积占比阈值。
- 不足阈值的 ROI 使用原存储 8 步；整图生成也保持原存储调度。
- 分组 ROI 按掩码并集计算面积，重叠和框间空白不会重复计费。
- 路由在生成之前确定，不读取质量结果，不增加候选或挑选种子。
- 不同采样调度进入不同的微批；关闭凑批分桶时仍保持相邻兼容关系。
- 每次执行后恢复原采样配置，包括解码异常和 OOM 恢复路径。

该阈值来自已消耗的开发样本。新的千张选集在调参前冻结，不用于挑选阈值。
模型权重文件没有修改；这里的 12 步是显式实验的均匀调度，不表示官方推荐默认值。
`model_sampling` 和每次尝试记录会保存实际执行步数及源面积依据，
`/health` 的 optimization 展示部署值。

四张真实 GPU 冒烟任务 c64035d8afbf4e22a77874d4683e8e26：
4/4 首轮通过，8 个 ROI 中实际 6 个执行 8 步、2 个执行 12 步。
无 OOM、无返工、无尺寸错误。该四张结果不属于千张验收。

## 保持的部署配置

4 个独立模型实例使用 GPU 0–3（NVIDIA L40），其他任务使用的 GPU 未动。
API concurrency 仍是允许在途的图像数，取 16；worker lookahead 4，
兼容编辑动态凑批最大 2，小预算 2，大于 1024 的预算最大 1。
保留 FP8 tensor DiT、BF16 视觉/语义编码、投影编译、KV cache、TF32、
独立 VAE 解码、VAE slicing、独立 BF16 4B 检验和大预算检验权重阶段管理。
source vet B1、detection/encoder B4。没有启用新 LoRA、源图覆写或实验检验改判。

所有布尔 CLI 参数仍为不指定 false、指定 true。
通用默认 GPU 列表仍为 0–7，本次明确通过参数只使用 0–3。
代码通过 WSL2 uv 环境开发、Git commit/push 后由开发机 pull。

## 千张冻结与复现

新选集：`first-pass-goal90-blind1000-nov03-v1`。

选集摘要：
`951a8ad5f23d9928be00edd06b6deb56f7401f3df315993f28e426cb3e741359`。

1,000 个唯一源文件，覆盖 11 个镇乡，排除 420 个历史或本批重复/近重复候选。
只按固定种子打乱的源文件和既有像素指纹筛选，没有使用生成质量标签、模型输出或预览。
所有大样本与模型留在开发机；本地只接收汇总和极少量压缩诊断预览。

验收工具 `scripts/benchmark-frozen-first-pass.py` 会：

1. 确认预期 GPU、独立权重、采样参数及冻结源文件 SHA。
2. 提交不覆盖、不返工的一轮任务。
3. 检查 1,000 个源 SHA、全部输出名称、递归目录及存储像素尺寸。
4. 检查实际采样步数、源面积路由、零生成 OOM 重试、零已有输出复用。
5. 检查推理源码树、uv.lock 和部署参数在测试期间保持一致。
6. 仅在至少 900 张质量通过且所有完整性检查通过时设置 qualified90。

## 首次千张候选：提前判定失败

任务 573616dbef53439d950e4a2c302508d1。在待复核数超过 100 后停止新的派发，
在途完成后的记录为：836 张已评估、732 通过、104 待复核；164 张未处理。
即使余下全部通过，1,000 张的通过上限也只有 896（89.6%），因此不再浪费 GPU 补齐该候选。
这不是完整千张成绩，不能把 732/836 报成千张通过率，也没有把未通过项目移出分母。

所有 1,000 个原图 SHA 再核对无变化，已完成输出尺寸错误 0、处理失败 0、
生成 OOM 重试 0，推理源码树及部署配置保持不变。过程、原始停止证明和最终
stopped-report.json 均留在开发机。验收工具因未满 1,000 条而拒绝生成合格报告。

后续试验保持全部检验标准，先比较逐行 FP8 与自动 ROI 坐标映射的源潜变量上下文约束。
已有“无文字改判”研究曾在旧 1,024 张上 0 改判，不会为了凑通过率重新默认启用。
当前尚未完成通过率至少 90% 的千张验收。

## Continued experiments: numerical safety and grounded review

The row and row+ROI-context fast-accumulation probes emitted a non-finite cast warning and were stopped. Their incomplete results are unsafe research output, not qualified candidates. Commit f98e686 adds pre-conversion finite validation and opt-in accurate FP8 accumulation. The deployed/default quality grading remains unchanged.

A new offline-only `grounded` inspection profile records visual observations before the same semantic booleans. It must pass known-defect controls and native-image audits before any online adoption; it cannot waive deterministic vetoes. Merely improving the numerical pass rate is insufficient. The default profile remains `legacy`.
