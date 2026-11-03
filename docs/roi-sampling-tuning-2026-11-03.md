# ROI 首轮去水印调优，2026-11-03

当前状态：新增完整冻结千张测试为 872/1000 首轮自动检验通过（87.2%），尚未达到 90%，继续优化中。没有提交全量 38,319 张生产任务。

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

## Completed precision controls and rejected QA experiments

With the same 64 development sources, source-area 8/12 sampler, original ROI inputs and
strict legacy verifier, tensor FP8 scored 57/64 in 1048.026 seconds. BF16 scored 56/64
in 1273.037 seconds. Row FP8 plus a 1e-6 activation-amax floor scored 56/64 in 1185.154
seconds, with no non-finite errors or changed output sizes. These are single-GPU
development diagnostics, not four-GPU serving throughput. Tensor FP8 stays preferred.
Increasing large-ROI steps to 16 or 20 was stopped at inferior partial comparisons;
these partial runs have no qualified 64-image score.

Grounded narrative QA had 2 clean vetoes and 13 defect misses on the 108 labeled controls,
but incorrectly accepted a real rewritten physical English sign. Explicit untrusted-mask
instructions still missed that real defect. Both full protocols are rejected.
A narrower physical-print inquiry correctly blocked that rewritten sign by OCR/native
pixel mismatch. Its native-scan prototype, however, missed faint retained synthetic text.
The physical-print inquiry is therefore NOT a service deployment option; no known
unsafe change may be used to attain the target. Current production grading remains
legacy with all existing deterministic vetoes. Native and synthetic inquiry reports
are research artifacts, never a new thousand-image qualification.

Existing dataset-trained adapters are now being tested on the ROI route rather than
assuming their earlier whole-frame results transfer. Three pinned fused adapters at
scale 0.5 and a harmonic-fine blending control use the same 64 source files, masks,
sampling and legacy QA. No automatic generation retries, threshold changes or production
38319-image submission are enabled.

## 追加的首轮实验

对相同开发64张保持独立检验不变。新的区域对齐 LoRA 使用48个训练原图、
12个验证原图，派生232对水印区域；训练源与新封存千张集合无重叠。
rank8、alpha8、200步、LR5e-5的训练已完成，但训练损失下降不能证明图像质量提高。

| 方案 | 首轮通过 | 单卡处理秒 | 结论 |
| --- | --- | --- | --- |
| 无 LoRA，1024，按原图水印面积选择8/12步 | 57/64 | 1048.026 | 当前开发对照 |
| 区域对齐 LoRA，768 | 52/64 | 832.644 | 不采用 |
| 区域对齐 LoRA，1024 | 53/64 | 1054.741 | 不采用 |

新版详细中文提示词在48张配对开发图上27张通过，同一子集英文对照42张通过，
已提前停止；这是未完成实验，不能当作64张或千张成绩。

新增 `--tone-step-repair` 为显式启用的部署参数，默认False。仅在原图已定位的
户外修补范围内，拟合多条边共同支持的有界RGB明暗台阶，保持区域外像素、
原图实物边框和补足检测所用上下文不变；不生成新纹理，不额外调用扩散模型。
必须同时启用独立检验和原有矩形相位保护。修正后仍按原有标准完整检验，
处理器不能授予通过权限。59份原生开发候选中只有2张实际改变、32张通过对照均未改变；
这两张新像素经原有独立检验均通过，暂不能外推为千张增益。

LaMa掩膜修补仅加入离线实验，未进入服务默认路径。
模型文件从开发机直接下载，MD5严格匹配
[IOPaint官方实现](https://github.com/Sanster/IOPaint/blob/main/iopaint/model/lama.py)
所列e3aa4aaa15225a33ec84f9f4bc47e500，另固定SHA256
344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9。
逐ROI只依据原图几何选择一个后端，不做多候选择优或失败回退；
小区域混合与全部户外区域修补分别测试，结果不替代最终独立千张验收。

## 完整千张复测与后续检验对照

第二个预先封存的千张集合 first-pass-goal90-blind1000-nov03-v2：
选择清单 SHA256 为
2205c90930bf258e72b9b9c0a82dd03a98806a11a482c936b47c84bd50f5ae80，
1000 个唯一原图，覆盖 11 个镇乡，与 9536 个历史来源及训练来源隔离。
集合在第一次运行后已消耗，不能再称为新盲测。

完整任务 b9f408b798514beea4999644f6b1c88a：
872 首轮通过、128 待复核、失败/无效均为零；
4156.265 秒（69 分 16 秒），866.162 张/小时。
这是真实四卡端到端数据；不同历史集合上的 866/1000 不构成配对增益证明。
自动检验通过率不是人工标注准确率，也不能证明所有自动通过项都无缺陷。

加载版本 06f9e01，GPU 0–3、原始 ROI、FP8 tensor、
原图面积路由 8/12 步、bounded tone-step repair、原有独立 BF16 4B 检验和矩形保护。
1815 次 ROI 生成中 1343 次为 8 步、472 次为 12 步；
954 张走区域修补，46 张走原图保留。
没有生成返工、已有结果跳过或生成 OOM 重试。原图字节不变，
1000 张输出的像素尺寸、文件名和目录均正确，并核对完整的 1079 个目录。
tone 修正仅在一张上实际生效，不能由此推断是千张提升的原因。

最初监视器设置了不可达 90% 时提前停止。为取得完整千张记录，
用只读 --attach-job 绑定同一任务、封存摘要、参数、部署版本及权重身份，
替换监视器后继续监测，没有重新提交任务，也没有再次生成候选。
详细记录仍在数据盘 roi-adaptive-tone-blind1000-20261103-v2/report.json。

配对 64 张的 source-edge-small-panels 开关：关/开均 57/64，
分别为 1063.641/1062.000 秒，没有测得收益，不默认启用。

固定 191 张 QA 对照包含 27 个明确合格、164 个已知缺陷。
以下比较使用同一套语义、数值和源图保护实现，所有错误保留在分母：

| 独立 QA 候选 | 错放已知缺陷 | 拒绝合格 | 错误 | 结论 |
| --- | ---: | ---: | ---: | --- |
| 原有 4B BF16 | 30 | 4 | 0 | 当前对照，不代表检验已完美 |
| 32B NF4 | 53 | 3 | 0 | 新增 23 个错放，不采用 |
| Qwen3.5 9B BF16 | 37 | 3 | 17 | 另有残留定位 JSON 错误，不采用 |

不能把较少的待复核直接当作质量进步。较大模型未自动带来更可靠的拒绝。
旧版本模型对照不能与本表混合，避免把检验实现差异当成模型差异。

OCR 实验也独立记账：服务器版 PP-OCRv5 在原生 4096 分辨率、
pixel/box 阈值 .05/.1 的 24 张覆盖测试中只覆盖 7 张（均为不透明样式），
不能替代现有定位。OCR 字形或读字一致均不是数字水印不存在的证明。
所有新权重由开发机直接下载并核对发布的 SHA256，未通过 SSH 传输。

专用检验 LoRA 的继续训练、视觉与语言联合训练、未遮挡完整结果图上的实物文字复核
仍为离线研究。必须先通过淡字、底板、重叠实物印字、阴影和真实内容改写对照，
再检查原生样本；不能绕过原有自然度、内容、数值或不确定性拒绝。
它们尚未进入服务默认路径，也没有取得新的千张验收成绩。
