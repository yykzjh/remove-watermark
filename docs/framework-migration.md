# vLLM-Omni调研及迁移范围

已读取“调研 FLUX 模型及本地部署”会话，框架为vLLM-Omni，并对当前官方实现重新核对。
当前官方配方明确列出Qwen-Image-2.1-Turbo、QwenImage21Pipeline及图像编辑支持：
[Qwen2.1官方配方](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Qwen/Qwen-Image-2.1.md)。
先前会话中的支持状态不能视为当前结论。

## 请求级与步级执行

[官方架构](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/diffusion_continuous_batching/)区分：
- REQUEST_BATCH：把兼容请求融合到一次完整forward；
- STEP_BATCH：分别维护每请求编码、去噪、调度、解码状态，按去噪步接纳和完成请求。
真正的连续凑批需要后者的原生pipeline阶段接口；不是调大concurrency。
[执行模式](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/execution_modes/)与
[请求调度源码](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/sched/request_scheduler.py)、
[步级调度源码](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/sched/step_scheduler.py)
用于核对边界和兼容性，而不是照搬H200成绩到L40。

Qwen2.1条件图像的布局及KV阶段也会限制兼容性；不同图片不能因为提示词相同就共享视觉条件。
官方文本到图像条件缓存不会自动覆盖图片编辑；CacheDiT/TeaCache不能未经支持验证直接打开。
Turbo继续使用检查点原生8步schedule，不能靠随意少步或提高CFG来冒充同质量提速。

## 当前项目已实现的迁移

|技术|本项目实现|边界|
|---|---|---|
|兼容请求凑批|按推理尺寸分桶；每请求独立种子与输出对应|已有独立图像attention边界；不是混合所有视觉token|
|阶段本地凑批|源图覆盖层定位、输出残留扫描可按1..4图批处理|左padding、同提示词预算；不同图分别解析|
|有界接纳|concurrency限制在途输入，batch_window限制每卡窗口|离线目录已有积压，不增加固定等待来凑批|
|尾部负载均衡|--adaptive-dispatch按总worker数缩小剩余窗口|主要改善小批次/尾部，不能声称万图全程大幅提速|
|失败隔离|检测/生成/检验按图记录错误；批次失败拆分|单图失败不会被当成通过|
|请求生命周期|返工与首轮分队列；暂停/取消不再调度下一次尝试|GPU内正在执行的forward仍需等待|
|候选与验收分离|只记录每图最终结果；保留最佳候选及逐次审计|检验是模型判断，不能代替真值|

没有替换为vLLM-Omni后端，没有把当前Diffusers执行称为步级连续凑批。
原生引擎迁移会涉及依赖/模型阶段实现/量化与attention适配，应在独立Linux环境验证后另行比较。
当前迁移保留已实测的uv锁定依赖、FP8和lean编码方案，避免破坏现有部署。

## A/B验证

scripts/benchmark-quality.py记录相同真实64图的全任务wall time、成功数、尺寸、验收及返工结果。
部署参数固定到.run/deployment.json，源图清单在每轮sources.json。
生成和检验分开计时；加载成本与任务耗时分别报告。
本轮结果见 [实测报告](quality-performance-2026-10-31.md)。
