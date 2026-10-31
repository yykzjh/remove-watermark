# 去水印后质量检验

启用后，每张成功输出的最终文件都会与对应原图配对，使用所选视觉检验后端检查：
数字水印和底纹是否残留、修补区域是否自然、原来可见的场景内容是否被改变。
两种后端均不能替代人工验收或证明隐藏内容的真实还原。

## 部署后端

服务启动需指定 --quality-verifier（不指定=false），并选择 --quality-backend：
- resident：复用去水印模型内的视觉语言编码器；使用短提示词对照检查、快速异常检查，
  对尚可通过的输出再单独扫描残留水印，不新增模型权重。
- independent：在每卡额外加载 Qwen3-VL-4B-Instruct，并配对比较原图与结果；
  通过 --quality-model 和 --quality-revision 指定模型/版本，会增加显存。
  Hugging Face 默认版本 main，ModelScope 通常需显式 master。支持已校验的本地模型目录。

完整脚本 scripts/launch-throughput-quality.sh 以 resident 为预设，末尾可覆盖 scalar 参数。
例如 --gpus 1,2,3,4,5 可避开正在被其他用户使用的卡。
独立模型的首次下载及加载不属于每批处理耗时，仍需另行考虑启动成本。

## 使用

桌面客户端勾选“去水印后检验残留、自然度及内容一致性”，选择检验预览长边和每卡检验凑批上限。
bool 选项沿用不指定=false、指定=true 的约定。批量与单张任务均支持。

~~~bash
uv run --no-sync watermark-batch /mnt/sdb/zjh/input /mnt/sdb/zjh/output \
  --concurrency 20 --input-resolution 1024 \
  --quality-check --quality-resolution 768 --quality-batch-size 2 --quality-max-tokens 128
~~~

- quality_resolution：对照图预览的最长边，默认768；较低值更快，但小字、远处人物等细节更难可靠判断。
- quality_batch_size：每卡一次视觉检验的图片对数，默认1，上限4；显存不足时拆分重试。
- quality_max_tokens：每张报告最多生成的 token 数，默认128；截断或无效 JSON 会记为检验错误。
- resident 后端没有额外检验权重；两个后端均没有将整批原图长期放在内存中；每个 worker 的 lookahead 窗口内完成输出保存，再检验最终编码文件。
- 不同 GPU worker 继续独立处理窗口。resident 共用本卡编码器，independent 使用额外编码器；单卡按阶段顺序执行，不在同卡同时运行两个推理。
- 暂不自动反复重画待复核图，以控制耗时并保留第一次输出供检查。

## 结果

每张图片的 results.jsonl 保存 quality 字段：

- passed：本次预览中模型判断三项满足且未报告不确定。
- review：残留、自然度、内容一致性任一不满足，或模型表示不确定。
- error：模型调用、解码或结构化结果解析失败。不能算通过。
- skipped：图像处理本身失败或使用 mock 后端。mock 不产生真实质量通过结果。

watermark_removed / natural / content_preserved 是三项判断；uncertain 表示不确定；
reason 描述缺陷及位置。pixel_mae 仅是缩略图差异诊断，不是质量得分或硬阈值。
seconds 包含配对检验按批次均摊的耗时，以及该图的残留扫描耗时（如执行）；
batch_seconds 是配对检验批次耗时，residual_scan_seconds 是额外残留扫描耗时。
window_seconds 包含该窗口的去水印、保存与检验，完整任务耗时应从任务提交到全部完成衡量。

summary.json 保存 quality_counts，客户端显示通过、待复核、错误数量及最近20条复核原因。
全部逐图记录仍保存在 results.jsonl。已输出文件不会因待复核被删掉或覆盖回原图。
CLI 退出码0表示处理完成且无待复核/检验错误（或未启用检验）；1表示任务/处理失败；
2表示处理完成，但有待复核或检验错误。

## 提示词与边界

去水印提示词位于 src/watermark_studio/prompts.py，检验提示词位于 quality.py。
抽样已观察到蓝白/灰黑信息卡、黄黑横条、绿色蓝色应用图标、小型白色品牌文字等不同样式，
目标不限定固定角落。真实安全标牌、机械品牌、衣服文字和施工标记应保留。
施工场景的泥水、轮迹、履带印、土层、钢筋、模板、管口及工程边线属于原图内容；
不应被“美化”成平滑干净地面。

水印完全遮挡的区域没有可核对的原始像素，只能检查重建的合理性。
低清预览可能漏检小字，视觉模型可能误判；不承诺所有输入达到完美或零内容变化。原图与结果几乎相同、或引入大面积新纯色区域时，快速异常检查会保守标记待复核，包含原图本来无水印的情况。
需要更细检查时提高检验预览档位，或对待复核图进行人工检查。

## 复测

scripts/benchmark-quality.py 对同一24张真实图片比较关闭检验、512/768预览、batch1/2的完整任务耗时。
scripts/verify-quality-controls.py 使用未去除水印和明显破坏内容的反例，检查评估器是否拒绝问题输入。
完整数据集下载完成后还需补充分散目录的抽样，再复核提示词覆盖和检验结果。
性能成绩与反例结果在完成本轮实验后补充；不同分辨率、图片内容与硬件需要重新测量。
