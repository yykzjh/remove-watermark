# 流式千张抽检

服务器的去水印任务继续运行时，可通过百度网盘提前传回已合格的抽样结果。抽检与服务器多次返工失败图像的本地兜底共用一个 Codex 队列，codex_concurrency=4 表示两类任务合计最多四个独立会话。抽检只判断结果，不对服务器已合格图像调用本地生成模型；发现的缺陷保留首次判定和依据。

## 抽样与最终名单

先对完整的有效输入路径清单进行固定种子的随机排序，预选 1000 个候选。每个候选在服务器完成并提交、且检验通过后即可导出。当前已合格图像越多，可检查的候选数量会按其占总输入的比例增长，实际数量有随机波动。

全量结束后，保留所有已抽检的合格候选，再从其余服务器合格结果中独立随机补足 1000 张（合格总数不足时全部检查）。两阶段抽样在合格集合中保持无放回均匀抽样。名单不依据 Codex 判定进行替换；提前发现不合格的样本也保留在最终名单中。候选清单、种子、完整输入清单哈希和每张首次判断都固定保存。

空文件及非图像不进入抽样，但仍按原目录完整保留。当前数据集的有效图像总体为 38,821 张。

## 服务器导出

两台服务器使用同一份完整原始 manifest，不能使用各自分片清单。各自只导出本机已提交目录内的候选。该命令不部署模型：

~~~bash
UV_PROJECT_ENVIRONMENT="$HOME/.cache/uv-envs/remove-watermark" uv run --no-sync python -m watermark_studio.early_review export --config /data/task/state/passed-export/config.json --manifest /data/task/original-manifest.json --group sample-1000 --sample-size 1000 --seed 20261106 --population 38821
~~~

--population 用于核对有效输入数量；--sample-size 为最终抽检目标；--seed 固定候选与补足名单。导出状态放在 state/passed-export/，网盘目录为任务目录下的 passed-review/。完整候选清单放在 ZIP 元数据中，经百度网盘传输；SSH 每次最多读取两条回执或两张图的提交状态，不传图像数据。

## 本地配置

~~~json
{
  "stream_passed_sample": true,
  "codex_sample_size": 1000,
  "codex_sample_seed": 20261106,
  "codex_sample_population": 38821,
  "codex_concurrency": 4,
  "codex_receipt_prefetch": 8,
  "codex_min_server_attempts": 3,
  "codex_server_failures_only": true
}
~~~

未启用 stream_passed_sample 时保持原流程。开启后，来自两个板块的任务交替补入同一个有界队列。原图优先复用本地校验一致的文件；缺失时从百度网盘下载。候选输出必须与服务器最终提交的 SHA-256 一致。

early-codex/passed-results.json 保存抽检结果，passed-needs-review-relative-paths.txt 和 passed-needs-review-details.json 保存发现的问题。完整首次判断缓存位于 codex/sample-1000/<路径哈希>/review.json。最终组装后直接复用相同原图和输出字节的判断，只补做未完成项；若字节、抽样清单或已保存判断发生变化，流程报错，不重新打分覆盖原判定。

状态中的 passed_sample_reviewed、passed_sample_accepted、passed_sample_needs_review 与失败图像兜底计数分开。最终报告同时给出最终千张抽检结果和提前完成的校验数量。

配置 codex_receipt_prefetch=8 可为每个来源预取至多八条回执，减少目录尾部空等；每条 SSH 命令仍最多返回两条回执。该值不是 Codex 并发数。

图像编辑服务若明确返回 moderation_blocked，仅将该图像标记为待复核，记录 native_edit_provider_rejected 和原始服务日志，保留此前最佳输出并继续其他图像。重启后复用这条拒绝记录，不重发该图像、不改写提示词规避拒绝；其他网络、配额或工具不可用错误仍作为基础设施故障显式报告。已完成的独立校验结论保持不变。

两台服务器的结果不必同时发布：某一节点的固定 SHA256 交付回执发布后，协调器立即经百度网盘下载并校验该节点的完整分包，另一节点可继续打包上传。只有两节点的全部分包都验证完成后，才组装最终目录和补齐千张抽样。每个节点只有一个下载所有者；重启会复用校验匹配的缓存，已发布回执发生变化则停止并报告，避免混入不同版本。
