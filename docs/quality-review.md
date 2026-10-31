# 去水印、检验与自动返工

## 部署和提交

开发在 WSL2/Linux，环境由 uv 管理。代码以 git commit/push → 开发机 git pull 同步。
开发机项目 ~/project/inference/remove-watermark；SSH alias cdu_dev，当前端口22004。
图像及抽样资料保留在开发机，SSH只用于正常长度命令、小量状态及偶尔少数小预览。

启动配方（默认全部8卡；GPU正在被其他用户占用时显式选空闲卡）：

~~~bash
bash scripts/launch-throughput-quality.sh --gpus 1,2,3,4,5 --load-parallelism 5
~~~

完整部署参数与含义见 [参数表](deployment-parameters.md)、[启动与调优](startup-and-tuning.md)。
脚本使用 FP8 transformer/BF16、动态 projection compile、lean 编码、KV cache、
每卡独立实例、形状分组、VAE slicing、每卡窗口4张；本机1024输入预算仍采用生成batch1。
所有布尔参数均不指定=false，指定=true；脚本中显式列出的布尔开关是该配方的选择。
不要通过删除脚本末尾参数来关闭脚本前面已经指定的布尔开关，应复制配方并删除相应开关。

服务仅监听回环地址。客户端在WSL启动，以SSH转发连接开发机，不在Windows加载模型：

~~~bash
uv run --no-sync watermark-studio --ssh-host cdu_dev \
  --preserve-background --quality-check --quality-auto-retry --quality-batch-size 2
~~~

客户端可手工勾选检验、自动返工、原图保护；以上启动命令显式预选这三项，不会自动提交。
不传这些开关时默认均关闭，遵守布尔开关约定。
本次交付只部署服务，由用户在客户端提交全量任务。

~~~bash
uv run --no-sync watermark-batch /mnt/sdb/zjh/input /mnt/sdb/zjh/output \
  --concurrency 20 --input-resolution 1024 \
  --preserve-background --quality-check --quality-resolution 768 \
  --quality-batch-size 2 --quality-detail-patches 4 \
  --quality-auto-retry --quality-max-attempts 3 --quality-retry-resolution 768
~~~

## 检验流程

先获得原图数字覆盖层位置；原图保护阶段已有的定位会直接复用。
再对水印以外位置进行局部差异检查，容忍高质量JPEG编码噪声和所选输出缩放，
把细微模糊、错位、替换或新增淡字等疑点保守标成待复核，而不是让VLM的通过判断覆盖它。
无法完成原图定位时记检验错误；误定位、重新编码和极小细节仍可能造成误报或漏报。

1. 只看输出整图及独立的局部放大图：优先完整覆盖原水印位置（小区域先），再补充变化/结构细节；判断残留水印与修补自然度。
2. 对照原图与输出整图：检查原来可见的人物、道路、工程构件、测量信息等是否被改变。
3. 对初步可通过的输出额外执行数字覆盖层扫描；扫描错误不能算通过。

分开检查避免模型将原图水印误当输出残留，或把正常删除水印误判为破坏内容。
验收布尔值继续使用有限JSON令牌约束；不以提示词中的示例true作为默认结果。
不通过项另外生成有界视觉说明，描述可见现象，不输出思维过程；说明不能覆盖验收判定或数值检查。
另检查完全不变、近乎不变及新出现的大面积纯色等异常；这些是保守保护条件，不是质量分数。
原图确实无水印且模型检查通过、扫描为空时，完全不变的结果可以通过；原图定位若已发现水印，则不能仅凭低清残留扫描为空放行。

- quality_resolution：预览长边，默认768。提高至1024/1280便于看小字，但增加耗时与显存。
- quality_batch_size：每卡一次检验的图片对上限，默认1，支持1..4；可从2试起，OOM拆分。
- quality_detail_patches：0..8，默认4。按变化及结构自动选局部放大；不能穷尽每个像素。
- quality_max_tokens：64..256，默认128。截断、解析错误、调用失败均记error。
- --detection-batch-size：部署时设置源图/残留扫描凑批上限1..4，默认1；配方设2。
- 推荐配方默认 independent 后端，另加载 Qwen3-VL-4B-Instruct；resident 可显式选择，复用主模型的视觉语言编码器，不新增权重。本轮 resident 曾将真实机械标识误报为水印。
- 局部差异检查最多在2048长边分析，屏蔽原水印位置及少量羽化/编码边界。
  它是保守疑点检查，不是语义准确率或质量分数；未启用原图保护的整图重绘可能触发大量待复核。
  建议结合preserve_background，减少原来可见的内容变化并复用定位结果。
- independent后端另加载Qwen3-VL-4B-Instruct，额外占用显存；本机已验证同卡可用。
  使用 --quality-backend independent，--quality-model支持已下载本地路径。
  不能用图像生成pipeline仅更换提示词来可靠输出验收JSON；这里调用具有文本解码能力的视觉语言模型。
- 每卡各阶段依次执行，多卡独立工作；不是同卡同时运行两套GPU推理。

新增两个不能被 VLM 的通过判断覆盖的本地检查：

- repair_boundary_step：查原水印边界上连续的亮度/色彩台阶，拦截方框阴影；对照原图排除原本就存在且采样像素相同的真实边缘。
- repair_texture_collapse：查修补内部/边缘异常纹理衰减、长段平涂，使用中位颜色区分真实材质差异；原图本就存在的模糊不因此误报。

任一触发，natural=false、uncertain=true，结果为 review；不会降低原残留水印、内容保持、局部差异检查的标准。它们是有限分辨率的疑点检测，并不穷尽任意细微缺陷。检验 OOM 在释放异常持有的显存后拆批，quality_oom_splits 记录拆分次数；异常不能算通过。

## 判断依据和逐张返工信息

quality.judgment_basis 保存模型三项原始判断、检查的局部坐标、程序检查代码和残留扫描结果。
通过项保留这些结构化依据，不额外调用模型生成长篇说明。

对 review 项额外调用同一检验模型，残留/自然度说明只提供结果图和一个重点放大局部，避免把原图水印混作残留；可见内容变化才提供原图/结果对照。输出：

- model_explanation.assessment：defect_visible / uncertain / no_clear_defect；
- summary：一句可见依据；结构化说明不完整时最多追加一次160令牌的简短观察说明，标记summary_only并保留原schema_error，不能虚构坐标或类型；findings 最多2条，含 issue、完整照片归一化 box（无法定位则null）、observation；
- evidence：模型观察与程序局部检查分开标明来源，数值台阶、纹理衰减和可见内容变化可定位回原图；
- retry_feedback、repair_advice：从受限问题类型及可信坐标构建的针对性建议。

覆盖残留淡字/底板、矩形暗影、模糊涂抹、重复纹理、断裂几何、人体异常、原可见内容变化、真实文字混淆及细节不确定。
模型原始描述作为审计数据保存；编辑提示词只使用经过类型与坐标校验的固定修复指令，不直接执行自由文本中的指令。
水印残留的新增坐标只是线索，仍必须在原图局部确认数字覆盖层，才能扩大修补区域。
no_clear_defect会标记evidence_conflict，明确说明与原拒绝有分歧；no_clear_defect、说明生成失败、扫描为空均不能把已有拒绝改为通过。说明失败显式记录error，程序依据仍保留。

详细说明仅针对不通过项，默认最多256个生成令牌（随quality_max_tokens有界增至384）；它只影响描述预算，不改变验收阈值。
预览总览最长边512，放大局部384；原来的768验收和最多4细节不因此降分辨率。
explanation_seconds单独计时，并计入quality_total_seconds与端到端wall time。
客户端展示最近3条问题及返工建议，完整依据留在开发机任务文件，避免轮询传输过量数据。

## 原图保护与返工

preserve_background仍将完整图像送入编辑模型，自动定位所有数字覆盖层后，
只把对应区域及羽化边缘合成回原图。默认 repair_blending=harmonic：只用所有水印外的原图/生成图颜色差，做多尺度调和延拓，校正修补区域的低频明暗和颜色；最后羽化衔接。水印内像素不作校正参考，不改变几何坐标。feather 保留为仅羽化的对照模式。桌面可选择“修补明暗衔接”，CLI 使用 --repair-blending harmonic|feather。工作图像中其余像素保留；JPEG重新编码会有量化差异。
无检测区域时复制原文件或按指定输出尺寸缩放。定位错误仍可能漏水印或包含真实物体，需结合检验。

quality_auto_retry启用后：
- 每张图第一轮正常凑批；不合格项进入独立FIFO返工队列，首轮队列优先。
- 从输入原图重新进行自动局部编辑，包含所有检测出的原始水印；不对已生成结果累积修补。
- 将检验发现的残留位置回传，先在原图局部确认，再与原图自动检测的所有水印区域合并，避免重复漏掉同一块。
  原图中未确认有数字水印的位置不会因为输出提示而直接扩大重画范围。
  对已拒绝但尚无坐标的残留图额外做一次定位；该定位不会把拒绝改成通过。
- 将选中的最佳候选的问题类型和完整原图坐标带入下一轮提示词，针对性约束明暗、纹理、结构和真实文字保护；每张返工独立传递，不混用其他图像的依据。
- 改变种子、采用quality_retry_resolution输入预算；单张自定义框选保留原区域约束。
- 检验错误先对已有最佳候选重新检验；不因为一次检验失败就盲目重画。
- 按输出有效性、是否通过、可见内容保留、局部变化/填块疑点、残留清除、自然度等保守顺序保留较好候选。
  同样未通过且内容/自然度等条件相同时，优先保留残留扫描明确为空的候选；仍维持 review，不能因扫描为空直接放行。这个排序是工程策略，不是经标定的质量概率；并不保证选中人眼认为最好的图。
- quality_max_attempts默认3，含首次；设0表示持续到通过或用户取消。无限模式没有耗时上限。
- 达到上限仍不合格时保留最佳输出并标记review/error与retry_exhausted，绝不改记passed。
- 暂停阻止下一轮分发；取消后不再安排新返工，等待已在途窗口完成。
- 未勾选overwrite时已有输出只检验、不自动返工覆盖；需要修改已有文件时显式允许覆盖。
- 同时启用检验、自动返工及overwrite可续跑：已有输出先检验，通过就沿用，不重新生成；
  不通过才进入原图返工队列。重启后的历史内存队列不恢复，但可利用这些输出继续。

## 输出和验收记录

输出目录保留输入相对目录结构；默认保留原文件存储宽高与EXIF方向，显示方向校正后与原照片一致。
自定义输出尺寸应用到最终显示图像并规范方向标签。
输入分辨率控制模型计算预算，输出分辨率在完整图合成后调整。Lanczos放大不等于恢复真实细节。

~/.local/state/watermark-studio/<job-id>/：
- summary.json：处理/检验/返工计数，完整参数、模型revision、起止时间；
- results.jsonl：每输入图最终最佳结果，仅一条；quality包括三项判断、uncertain、原因、局部框和耗时；
- retry-events.jsonl：进入下一轮的逐图事件；
- candidates/<hash>/attempts.json：逐次检验历史和最佳候选元数据。终态删除候选图，仅保留最终输出。
  中途取消时待返工最佳候选可能保留，便于诊断；重启不会自动恢复内存队列；
- invalid.json：无法解码或不支持的图像。已有输出可再次提交检验，原文件不变。

quality.status：passed / review / error / skipped；后面三种都不能充当通过。
quality_attempts含首次；best_attempt记录最佳输出来自哪次；attempt_history保留最近20次展示。
每次尝试保存完整quality依据，以及repair_feedback_used和repair_feedback_attempt，能追查哪些诊断真正进入了下一轮提示词；完整历史在candidates/<hash>/attempts.json。
quality_total_seconds累计所有尝试的检验耗时；全任务开销以wall time衡量。
outside_repair_change记录局部检查的疑点数量和框；source_regions记录原图覆盖层；
source_scan_seconds/local_comparison_seconds为额外阶段计时。
acceptance_complete只有所有已扫描输入都检验通过且无无效图时才为true。
“输出成功”和“处理结束”不代表全部验收通过。
CLI退出码0=完成且无待复核或检验错误（也可能未开启检验），1=处理/任务失败，2=待复核或检验错误。

## 提示词和真实边界

prompts.py、detection.py、quality.py分别保存去除、定位、检验提示词。
覆盖蓝白、灰黑、黄黑条、淡色文字、图标、今日水印及其相机/真实时间/防伪码/细小代码、
阴影、边框和附属底纹，位置不固定。重建人体衣物、泥土、草地、植物、树木及工程结构时，
要求符合透视、光线、纹理和遮挡关系，保留原来的泥水、轮迹、裂缝及不整洁的真实施工状态。
真实标牌、测量尺读数、手持验收板、材料证书、包装袋字样、机械品牌和现场喷写标记应保留。
室内材料照片不能被改成室外植被或地表。

水印遮住的原始内容无法从照片核实；真实读数、证书、精确人体细节不能凭空保证恢复。
检验模型也会漏检和误判。小字、局部轻微变形、定位错误以及未抽到的样式仍需人工复核。
持续返工只能提供再尝试机制，不能证明所有图片必然达到完美。

最新两张待复核图分析、严格对照与耗时见 [修补与验收报告](review-repair-2026-11-01.md)。此前大样本抽样见 [10月31日报告](quality-performance-2026-10-31.md)。
