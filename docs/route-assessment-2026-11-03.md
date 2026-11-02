# ROI、整图与返工方案评估（2026-11-03）

## 决策与实际边界

采用 **自动定位每张图的水印 → 各ROI独立处理 → 原图保护合成 → 严格检验 → 有明确位置依据时有限返工**。推荐 complete、context0.45、主体输入预算1024、小覆盖层512、harmonic、独立4B检验768/batch1、最多2次尝试（含首次）。原图尺寸、文件名、深层目录和非图像保留机制继续使用。

本轮没有找到显著提高自动通过率的新配置。整图、合并ROI输入没有显示综合优势；增加候选数量或分辨率在两例上没有恢复通过。有依据的有限返工用于保留处理真实缺陷的后手、减少无效重试，并维持吞吐，不能描述为已解决千张通过率问题。

服务当前使用GPU0–3，GPU4–7未纳入本次部署；全局默认8卡不变。本轮没有提交38,319张生产任务。检验标准及明暗、纹理、矩形阴影和可见内容否决未降低。

## 数据与计时

192张不同原图：64张用于路线筛选，另128张用于配对回归，交集0。它们都来自已推理过的千张数据，明确标记 development_reuse，不是新的未见测试集。加上两例返工对照，共10个任务、646张次首份候选流程，另有实际返工。

耗时为任务 finished_at-started_at，包含扫描、定位、生成、合成、检验、读写与补充尝试，不含模型加载和准备。单次顺序对照受缓存和共享资源影响，细小差异不能视为稳定因果收益。自动通过率不是人工确认准确率，也不能将128张与旧千张86.6%直接比较为提升。

## 64张同源路线筛选

4张L40、原尺寸输出、相同独立检验，全部关闭返工和争议改判。

| 路线 | 首轮通过 | 总秒数 | 逻辑编辑输入数 |
| --- | --- | --- | --- |
| roi | 51/64 | 297.8 | 113 |
| roi-context | 53/64 | 302.5 | 113 |
| whole-protected | 43/64 | 302.5 | 60 |
| roi-grouped | 49/64 | 305.1 | 98 |

roi：独立ROI、context0.45；roi-context：context0.85；whole-protected：整图生成后仅合成水印位置；roi-grouped：保守合并相邻可读ROI输入，仍分别回填掩膜。

整图未获得速度优势，通过数更少。相同1024像素预算下，裁切不代表每次主模型计算量自动下降；512小覆盖层、细节可读性和后续质检成本都会影响总耗时，不能只按调用次数判断效率。

合并输入将113个编辑输入降至98个，但通过数和耗时均未胜出，因此不作推荐。研究CLI/API的 roi_grouping=adaptive 保留复现，默认separate，普通桌面不增加实验选择器。

## 128张配对回归

| 配置 | 首轮通过 | 最终通过 | 最终复核 / 检验错误 | 总秒数 | 张/小时 |
| --- | --- | --- | --- | --- | --- |
| roi | 115/128 | 115/128 | 13 / 0 | 582.0 | 791.8 |
| roi-balanced | 115/128 | 116/128 | 11 / 1 | 628.8 | 732.8 |
| roi-evidence | 115/128 | 115/128 | 13 / 0 | 582.4 | 791.3 |

- roi：原配置，context0.45，无返工。
- roi-balanced：context0.85，普遍补充尝试，最多2次、返工1024。10张尝试第二次，其中7张只复检、3张实际重生成，仅1张恢复。13个首轮未通过中恢复1个，不能计入首轮。增加约8.0%耗时，且有1张残留扫描区域过多错误，不推荐普遍追加处理。
- roi-evidence：context0.45，仅有明确位置依据的测量异常或可见缺陷可返工。13个未通过中实际重生成2张，0张恢复。最终通过数未变，约791张/小时与原配置接近。首轮成功候选直接原子写入输出，省去candidate/best/output复制；拒绝和审计快照保留。

相对本次128张原配置：普遍补充模式115张共同通过、1张拒绝转通过、0张通过转拒绝；有依据返工模式115张共同通过、0张转通过、0张通过转拒绝。

原始benchmark JSON的 baseline_*_to_* 字段比较旧千张子集，不能误当作新128张配对；机器报告另有 paired_validation。本组没有提高首轮或最终通过率，尚未在新配方下重测千张，不能声称达到千张首轮95%。

## 检验描述如何指导返工

此前已有问题类别、原图坐标、模板动作和不同seed；本次补充：

1. 有位置依据的数值观察、可见描述以结构化数据传入，最多160字符，不把无依据summary当事实。
2. 完整图坐标转换到实际模型裁切坐标；不相交的观察不传给其他ROI，避免衣物问题被错误用于右下防伪码。
3. 观察以JSON引用并标明是数据、不是新指令；只采纳与原图可见内容一致的事实。
4. 每轮从原图重做；下一轮使用最佳候选的依据和位置，不采用更差候选的提示，更差图像不覆盖最佳结果。
5. 纹理异常增加内外纹理比，并提醒核实是否同一材质。自然平滑的衣物、车漆、玻璃或纸张不应被强行补成粗糙草地/泥土纹理。

实物文字分歧、位置不可靠或缺少可见依据的疑似残留继续保留review/error。说明“看不出问题”不能豁免数值检查。两例原生预览提示存在跨材质比较疑点，尚未据此修改标准；需要用明确真值及缺陷反例进一步校准。

## 随机性与预算

当前固定版本接口支持 generator/latents/sigmas/true_cfg_scale，没有图像生成的temperature或top_p。基础seed默认42；每次返工用不同seed，当前步长104729。更大的seed不代表更强随机性。定位和检验保持确定性采样。

桌面和CLI支持 --seed；quality_max_attempts=3表示首次加最多2次补充尝试，合格即停止。model_seeds与attempt_history记录实际输入种子。一份候选可能包含多个ROI调用，不等于每图只调用扩散模型一次。

两例来自上组有位置依据但未恢复的图，路径及种子计划固定；以下是失败集开发探针，不是全批通过率：

| 配置 | 首轮失败后恢复 | 总秒数 | 逻辑编辑输入数（含探针首轮） |
| --- | --- | --- | --- |
| one-retry-1024 | 0/2 | 63.1 | 8 |
| two-retries-1024 | 0/2 | 87.3 | 12 |
| two-retries-1536 | 0/2 | 118.2 | 12 |

三组无推理错误、无OOM，实际传入4/8/8条观察反馈，并记录不同seed。增加候选数量和返工分辨率在这两例上没有收益，默认仍为一次返工和1024预算。不调高全局噪声强度、不随意改变Turbo调度；增加步数也不等同于增加随机性。

## 其他方案

传统Telea在9个精确掩膜、已知原图真值对照中仅需1.315秒，但遮挡区平均MAE33.46、PSNR15.39dB、纹理能量比0.372，7例新增纹理或边界否决。速度不含定位/质检，不能作为工地大信息卡、人体和复杂背景的默认修补。

宽视野双模型源载体实验两轮均0/15分歧获完整数字证明、0/3已知实物反例误批，无可测收益，source_carrier_recheck保持关闭。保留“已有实物证据而后续核实错误时保护原图”的安全修正。没有启用未通过对照的LoRA、超分或其他权重。

## 启动与使用

现有服务保持运行。以后重启时显式选空闲卡，全局脚本默认仍8卡：

~~~bash
bash scripts/launch-throughput-quality.sh --gpus 0,1,2,3 --load-parallelism 4 --uniform-steps 0
~~~

WSL/Linux桌面连接现有通道，在窗口将全局在途图像数设为本轮4卡实测的16：

~~~bash
bash scripts/launch-roi-desktop.sh --url http://127.0.0.1:8765 --seed 42
~~~

开发机CLI，输入图像已在开发机，输出深层镜像：

~~~bash
bash scripts/launch-roi-batch.sh /mnt/data/input /mnt/data/output --concurrency 16 --seed 42
# 扩大返工搜索预算必须显式覆盖；本轮两例未证明更好：
bash scripts/launch-roi-batch.sh /mnt/data/input /mnt/data/output --concurrency 16 --quality-max-attempts 3 --quality-retry-resolution 1536
~~~

连接配方显式传 quality-check、quality-auto-retry、quality-retry-localized、quality-max-attempts2、quality-retry-resolution1024。数据模型和普通CLI的bool默认仍false，出现开关才true。桌面可取消自动返工。CLI无返工对照可直接 uv run --no-sync watermark-batch 并省略返工开关。退出2表示仍有复核/检验错误，不应据此重复提交。

concurrency是全局在途图像数；每卡batch_size2、大小输入上限、窗口、编码/定位凑批由服务另配。保留FP8、动态投影编译、KV缓存、TF32、独立VAE解码、VAE slicing、形状分组及按预算停放检验权重。大于1024预算的推理batch上限仍为1。

## 验证与版本

646条结果无重复、输出缺失或尺寸不符；192张原图和输入SHA均未改变，目录层级无缺失。194个输入路径包括两例另建硬链接，并非194张不同原图。本轮子集只有有效图像；非图像保留由既有镜像模块与回归覆盖。输出保持原分辨率，JPEG编码有正常量化差异。

最高allocated显存39,856MiB，无已记录推理错误或OOM；普遍补充模式的1张质检错误仍保留，没有排除分母。

WSL完整测试728通过、34跳过（缺Torch等条件），Ruff通过；开发机83项针对性CPU回归通过，两份连接配方bash -n与--help通过。编辑器/验证器canonical摘要保持不同，quality.py、rectangle_phase.py、repair_checks.py否决阈值未改。代码只经WSL Git提交/push，再由开发机pull。

初筛前三组加载6c13af7，合并ROI与128张前两组加载3b4a55b；有依据返工128加载0db446f；详细观察和两例三组实验加载2621583。后续提示词更新不能当作已重跑128张或千张的实测。

机器摘要：docs/route-assessment-2026-11-03.json。完整开发机记录：

~~~text
/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031/route-assessment-20261103-v1/summary.json
/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031/route-evaluation-20261103-v1/
/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031/route-validation128-20261103-v1/
/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031/route-retry-focus2-20261103-v1/
~~~

逐图依据、种子、最佳候选和首份拒绝快照留在 ~/.local/state/watermark-studio/<job>/。SSH只传命令、小统计和两份少量压缩诊断预览，未传整批数据或代码包。

## 原始资料

- [Qwen官方模型卡](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)：原生8步、推荐调度、CFG1/KV缓存。
- [Diffusers批推理](https://huggingface.co/docs/diffusers/main/en/using-diffusers/batched_inference)：吞吐、显存与批延迟权衡；ROI调度和张量组装为项目实现。
- [Diffusers局部修补](https://huggingface.co/docs/diffusers/main/en/using-diffusers/inpaint)：掩膜与羽化原理；本项目使用Qwen编辑与后合成，不声称通用inpainting参数均可用。
- [OpenCV Telea说明](https://docs.opencv.org/4.13.0/df/d3d/tutorial_py_inpainting.html)：邻域传播修补。
