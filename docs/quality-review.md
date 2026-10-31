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
uv run --no-sync watermark-studio --ssh-host cdu_dev
~~~

勾选检验、自动返工、原图保护后再提交。它们默认均关闭，遵守布尔开关约定。
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

1. 只看输出整图及自动选择的放大细节：判断残留水印与修补自然度。
2. 对照原图与输出整图：检查原来可见的人物、道路、工程构件、测量信息等是否被改变。
3. 对初步可通过的输出额外执行数字覆盖层扫描；扫描错误不能算通过。

分开检查避免模型将原图水印误当输出残留，或把正常删除水印误判为破坏内容。
结构化输出使用有限JSON令牌约束；不以提示词中的示例true作为默认结果。
另检查完全不变、近乎不变及新出现的大面积纯色等异常；这些是保守保护条件，不是质量分数。
原图确实无水印且模型检查通过、扫描为空时，完全不变的结果可以通过；原图定位若已发现水印，则不能仅凭低清残留扫描为空放行。

- quality_resolution：预览长边，默认768。提高至1024/1280便于看小字，但增加耗时与显存。
- quality_batch_size：每卡一次检验的图片对上限，默认1，支持1..4；可从2试起，OOM拆分。
- quality_detail_patches：0..8，默认4。按变化及结构自动选局部放大；不能穷尽每个像素。
- quality_max_tokens：64..256，默认128。截断、解析错误、调用失败均记error。
- --detection-batch-size：部署时设置源图/残留扫描凑批上限1..4，默认1；配方设2。
- resident后端复用主模型的Qwen3-VL视觉语言编码器，不新增权重。
- 局部差异检查最多在2048长边分析，屏蔽原水印位置及少量羽化/编码边界。
  它是保守疑点检查，不是语义准确率或质量分数；未启用原图保护的整图重绘可能触发大量待复核。
  建议结合preserve_background，减少原来可见的内容变化并复用定位结果。
- independent后端另加载Qwen3-VL-4B-Instruct，额外占用显存；本机已验证同卡可用。
  使用 --quality-backend independent，--quality-model支持已下载本地路径。
  不能用图像生成pipeline仅更换提示词来可靠输出验收JSON；这里调用具有文本解码能力的视觉语言模型。
- 每卡各阶段依次执行，多卡独立工作；不是同卡同时运行两套GPU推理。

## 原图保护与返工

preserve_background仍将完整图像送入编辑模型，自动定位所有数字覆盖层后，
只把对应区域及羽化边缘合成回原图。工作图像中其余像素保留；JPEG重新编码会有量化差异。
无检测区域时复制原文件或按指定输出尺寸缩放。定位错误仍可能漏水印或包含真实物体，需结合检验。

quality_auto_retry启用后：
- 每张图第一轮正常凑批；不合格项进入独立FIFO返工队列，首轮队列优先。
- 从输入原图重新进行自动局部编辑，包含所有检测出的原始水印；不对已生成结果累积修补。
- 将检验发现的残留位置回传，与原图自动检测的所有水印区域合并，避免重复漏掉同一块。
  对已拒绝但尚无坐标的残留图额外做一次定位；该定位不会把拒绝改成通过。
- 改变种子、采用quality_retry_resolution输入预算；单张自定义框选保留原区域约束。
- 检验错误先对已有最佳候选重新检验；不因为一次检验失败就盲目重画。
- 按输出有效性、是否通过、可见内容保留、残留清除、自然度等保守顺序保留较好候选。
  这个排序是工程策略，不是经标定的质量概率；并不保证选中人眼认为最好的图。
- quality_max_attempts默认3，含首次；设0表示持续到通过或用户取消。无限模式没有耗时上限。
- 达到上限仍不合格时保留最佳输出并标记review/error与retry_exhausted，绝不改记passed。
- 暂停阻止下一轮分发；取消后不再安排新返工，等待已在途窗口完成。
- 未勾选overwrite时已有输出只检验、不自动返工覆盖；需要修改已有文件时显式允许覆盖。
- 同时启用检验、自动返工及overwrite可续跑：已有输出先检验，通过就沿用，不重新生成；
  不通过才进入原图返工队列。重启后的历史内存队列不恢复，但可利用这些输出继续。

## 输出和验收记录

输出目录保留输入相对目录结构；默认最终完整图像尺寸等于经EXIF方向校正后的原始尺寸。
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

实测、抽样与对照实验见 [本轮报告](quality-performance-2026-10-31.md)。
