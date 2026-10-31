from __future__ import annotations

import hashlib
import os
import time
import traceback

from .config import DeploymentConfig, ProcessingConfig, Region
from .detection import locate_watermarks_batch
from .images import (
    atomic_copy,
    blend_patch,
    composite_protected,
    context_box,
    load_photo,
    pixel_box,
    regions_for,
    save_output,
)
from .prompts import build_full_prompt, build_prompt


def load_pipeline(config, model_path):
    import torch
    from diffusers import QwenImage21Pipeline

    torch.set_num_threads(config.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = config.tf32
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查 NVIDIA 驱动及 PyTorch CUDA 环境")
    dtype = getattr(torch, config.dtype)
    pipe = QwenImage21Pipeline.from_pretrained(model_path, dtype=dtype, local_files_only=True)
    pipe.set_progress_bar_config(disable=True)
    if config.vae_tiling:
        pipe.vae.enable_tiling(
            tile_sample_min_height=config.vae_tile_size,
            tile_sample_min_width=config.vae_tile_size,
            tile_sample_stride_height=config.vae_tile_stride,
            tile_sample_stride_width=config.vae_tile_stride,
        )
    if config.vae_slicing:
        pipe.vae.enable_slicing()
    if config.offload == "model":
        pipe.enable_model_cpu_offload(gpu_id=0)
    elif config.offload == "sequential":
        pipe.enable_sequential_cpu_offload(gpu_id=0)
    else:
        pipe.to("cuda:0")
    if config.vae_channels_last:
        # Convert convolution weights only, not unrelated temporal/linear tensors.
        torch.nn.utils.convert_conv2d_weight_memory_format(pipe.vae, torch.channels_last)
        torch.nn.utils.convert_conv3d_weight_memory_format(pipe.vae, torch.channels_last_3d)
    if config.attention == "sdpa":
        from .attention import IndependentSDPAProcessor
        backend = config.attention_backend
        if backend.startswith("native_"):
            backend = "_" + backend
        # Selecting a dispatch name alone does not initialize Hub kernels.
        # The official context manager validates dependencies and downloads once.
        from diffusers.models.attention_dispatch import attention_backend
        with attention_backend(backend):
            pass
        pipe.transformer.set_attn_processor(IndependentSDPAProcessor(backend))
    if config.attention == "flex":
        from diffusers.models.transformers.transformer_qwenimage21 import (
            QwenImage21FlexAttnProcessor,
        )

        pipe.transformer.set_attn_processor(QwenImage21FlexAttnProcessor())
    if config.fuse_qkv:
        from .optimizations import fuse_projections
        fuse_projections(pipe.transformer)
    if config.quantization != "none":
        from .optimizations import quantize_transformer
        quantize_transformer(pipe.transformer, config.quantization, config.float8_granularity)
    pipe._watermark_prompt_encoder = config.prompt_encoder
    pipe._watermark_encoder_batch_size = config.encoder_batch_size
    if config.compile:
        if config.compile_scope == "projections":
            # Keep KV mutation, masks and Python segment branches outside compilation.
            for block in pipe.transformer.transformer_blocks:
                attn = block.attn
                linears = [attn.to_qkv] if hasattr(attn, "to_qkv") else [attn.to_q, attn.to_k, attn.to_v]
                for module in [*linears, attn.to_out[0], block.img_mlp]:
                    module.compile(mode=config.compile_mode, dynamic=config.compile_dynamic)
        else:
            compile_fn = (pipe.transformer.compile_repeated_blocks if config.compile_scope == "blocks"
                          else pipe.transformer.compile)
            compile_fn(mode=config.compile_mode, dynamic=config.compile_dynamic)
    if not getattr(pipe.config, "sample_sigmas", None):
        raise RuntimeError("模型缺少 Turbo sample_sigmas；请检查模型及 Diffusers 版本")
    return pipe


def process_batch(tasks, options, config, pipe, batch_limits=None):
    """Bounded lookahead; independently detect files, then batch compatible crop edits."""
    from collections import defaultdict

    from .batching import dimensions, independent_batch

    start = time.monotonic()
    batch_limits = {} if batch_limits is None else batch_limits
    states, results, edits = [], {}, []
    source_images, existing_images = {}, {}
    cuda = config.backend != "mock"
    if cuda:
        import torch
        torch.cuda.reset_peak_memory_stats()
    detections, decoded = {}, {}
    needs_detection = cuda and (
        (options.scope == "full_image" and options.preserve_background)
        or (options.scope == "regions" and options.preset == "auto")
    )
    if needs_detection:
        detect_tasks = []
        for task in tasks:
            if task.get("reuse_existing"):
                continue
            try:
                decoded[task["relative"]] = load_photo(task["source"])
                detect_tasks.append(task)
            except (OSError, ValueError):
                pass  # Normal per-file handling below records the decoding failure.
        started_detection = time.monotonic()
        found = locate_watermarks_batch(
            [decoded[t["relative"]][0] for t in detect_tasks],
            pipe.text_encoder, pipe.processor, pipe._execution_device,
            options.detection_resolution, config.detection_batch_size,
        )
        elapsed = (time.monotonic()-started_detection)/max(1, len(detect_tasks))
        detections = {t["relative"]: (r, elapsed) for t, r in zip(detect_tasks, found)}
    for task in tasks:
        state = None
        try:
            source, exif, icc, alpha = (decoded.pop(task["relative"]) if task["relative"] in decoded
                                       else load_photo(task["source"]))
            if options.quality_check:
                source_images[task["relative"]] = source
            if task.get("reuse_existing"):
                existing = load_photo(task["destination"])[0]
                existing_images[task["relative"]] = existing
                results[task["relative"]] = {
                    "file": task["relative"], "ok": True, "action": "existing_output_verified",
                    "regions": [], "detection_seconds": 0, "output_size": list(existing.size),
                }
                continue
            detection_start = time.monotonic()
            protected_regions = None
            if not cuda:
                regions = []
            elif options.scope == "full_image":
                if options.preserve_background:
                    found, _ = detections[task["relative"]]
                    if isinstance(found, Exception):
                        raise found
                    protected_regions, _ = found
                    regions = [Region(box=(0, 0, 1, 1))] if protected_regions else []
                else:
                    regions = [Region(box=(0, 0, 1, 1))]
            elif options.preset == "auto":
                found, _ = detections[task["relative"]]
                if isinstance(found, Exception):
                    raise found
                regions, _ = found
                regions = [r for r in regions if options.remove_logo or r.kind != "logo"]
            else:
                regions = regions_for(source, options)
            detected = round(detections[task["relative"]][1] if task["relative"] in detections
                             else time.monotonic() - detection_start, 3)
            if not regions:
                if options.output_width:
                    if alpha is not None:
                        source.putalpha(alpha)
                    output_size = save_output(source, task["source"], task["destination"], exif, icc, options)
                else:
                    atomic_copy(task["source"], task["destination"], options.overwrite)
                    from PIL import Image
                    output_size = list(source.size)
                results[task["relative"]] = {
                    "file": task["relative"], "ok": True, "regions": [],
                    "action": "no_watermark_detected" if cuda else "mock_copy",
                    "detection_seconds": detected, "output_size": output_size,
                    "background_preserved": bool(protected_regions is not None),
                }
                continue
            state = {"task": task, "source": source, "result": source.copy(), "exif": exif, "icc": icc,
                         "alpha": alpha, "regions": regions, "protected_regions": protected_regions,
                         "detection_seconds": detected,
                         "batches": [], "failed": None}
            states.append(state)
            stable = int.from_bytes(hashlib.sha256(task["relative"].encode()).digest()[:4], "big")
            for index, region in enumerate(regions):
                box = pixel_box(region.box, source.size)
                crop = context_box(box, source.size, options.context)
                reference = source.crop(crop)
                edits.append({
                    "state": state, "box": box, "crop": crop, "index": index, "image": reference,
                    "prompt": (build_full_prompt(options.extra_prompt) if options.scope == "full_image"
                               else build_prompt(region.kind, box, crop, options.extra_prompt)),
                    "seed": (options.seed + stable + index) % (2**32),
                    "size": dimensions(reference, options.resolution),
                })
        except Exception as error:  # noqa: BLE001 - isolate file and GPU failures
            if state is not None:
                state["failed"] = f"{type(error).__name__}: {error}"[:800]
            results[task["relative"]] = {
                "file": task["relative"], "ok": False,
                "error": f"{type(error).__name__}: {error}"[:800],
            }

    buckets = defaultdict(list)
    # Disable bucketing => adjacent compatible edits only; never mix different layouts.
    run, previous = 0, None
    for edit in edits:
        if edit["state"]["failed"]:
            continue
        if edit["size"] != previous:
            run += 1
        previous = edit["size"]
        key = edit["size"] if config.batch_bucketing else (run, edit["size"])
        buckets[key].append(edit)
    oom_retries = 0

    def execute(group):
        nonlocal oom_retries
        failure = None
        try:
            patches = independent_batch(pipe, group, options.resolution, config.kv_cache)
            if len(patches) != len(group):
                raise RuntimeError("Pipeline returned wrong image count")
            for edit, patch in zip(group, patches):
                # Store patches; apply in detection order even when buckets reorder edits.
                edit["patch"] = patch
                edit["state"]["batches"].append(len(group))
            return
        except Exception as error:  # noqa: BLE001 - isolate file and GPU failures
            failure = f"{type(error).__name__}: {error}"[:800]
            is_oom = isinstance(error, torch.cuda.OutOfMemoryError)
            split = len(group) > 1 and (not is_oom or config.oom_split_batch)
            if is_oom:
                oom_retries += 1
                if split:
                    batch_limits[(options.resolution, group[0]["size"])] = len(group) // 2
        # Release exception traceback and transient tensors before retrying.
        torch.cuda.empty_cache()
        if split:
            middle = len(group) // 2
            execute(group[:middle])
            execute(group[middle:])
        else:
            for edit in group:
                edit["state"]["failed"] = failure

    for bucket in buckets.values():
        offset = 0
        while offset < len(bucket):
            size = min(config.batch_size, batch_limits.get((options.resolution, bucket[0]["size"]), config.batch_size))
            execute(bucket[offset:offset + size])
            offset += size
    for edit in edits:
        state = edit["state"]
        if not state["failed"]:
            if options.scope == "full_image":
                from PIL import Image
                state["result"] = edit["patch"].convert("RGB").resize(
                    state["source"].size, Image.Resampling.LANCZOS
                )
                if state["protected_regions"] is not None:
                    state["result"] = composite_protected(
                        state["source"], state["result"], state["protected_regions"], options.feather,
                    )
            else:
                state["result"] = blend_patch(
                    state["result"], edit["patch"], edit["box"], edit["crop"], options.feather
                )
    for state in states:
        task = state["task"]
        result = {
            "file": task["relative"], "ok": not bool(state["failed"]),
            "regions": [r.model_dump() for r in state["regions"]],
            "detection_seconds": state["detection_seconds"], "batch_sizes": state["batches"],
            "background_preserved": state["protected_regions"] is not None or options.scope == "regions",
            "repair_regions": [r.model_dump() for r in (state["protected_regions"] or [])],
        }
        try:
            if state["failed"]:
                raise RuntimeError(state["failed"])
            output = state["result"]
            if output.size != state["source"].size:
                raise RuntimeError("Output dimensions differ from original")
            if state["alpha"] is not None:
                output.putalpha(state["alpha"])
            output_size = save_output(output, task["source"], task["destination"],
                                      state["exif"], state["icc"], options)
            result.update(action="restored", output_size=output_size)
        except Exception as error:  # noqa: BLE001 - isolate file and GPU failures
            result.update(ok=False, error=f"{type(error).__name__}: {error}"[:800])
        results[task["relative"]] = result
    if options.quality_check:
        from .quality import verify_batch
        pending = []
        for task in tasks:
            result = results[task["relative"]]
            if not result["ok"]:
                result["quality"] = {"status": "skipped", "reason": "processing_failed"}
            elif not cuda:
                result["quality"] = {"status": "skipped", "reason": "mock_backend"}
            else:
                pending.append(task)

        def review(group):
            try:
                pairs = [
                    (source_images[t["relative"]], existing_images[t["relative"]]
                     if t["relative"] in existing_images else load_photo(t["destination"])[0])
                    for t in group
                ]
                reviews = verify_batch(
                    pairs, pipe._quality_encoder, pipe._quality_processor, pipe._execution_device,
                    options.quality_resolution, options.quality_max_tokens,
                    reviewer=config.quality_backend, detail_patches=options.quality_detail_patches,
                    detection_batch_size=config.detection_batch_size,
                )
                for task, verdict in zip(group, reviews):
                    size = ([options.output_width, options.output_height] if options.output_width
                            else list(source_images[task["relative"]].size))
                    verdict["dimensions_match"] = results[task["relative"]]["output_size"] == size
                    if not verdict["dimensions_match"] and verdict["status"] != "error":
                        verdict.update(status="review", reason=(verdict["reason"]+" 输出尺寸与所选设置不符。").strip())
                    results[task["relative"]]["quality"] = verdict
            except Exception as error:  # noqa: BLE001 - quality failures must never become passes
                if isinstance(error, torch.cuda.OutOfMemoryError) and len(group) > 1:
                    torch.cuda.empty_cache()
                    mid = len(group) // 2
                    review(group[:mid])
                    review(group[mid:])
                else:
                    for task in group:
                        results[task["relative"]]["quality"] = {
                            "status": "error", "reason": f"{type(error).__name__}: {error}"[:400],
                        }

        for offset in range(0, len(pending), options.quality_batch_size):
            review(pending[offset:offset + options.quality_batch_size])
    if cuda:
        torch.cuda.synchronize()
    metrics = {
        "window_seconds": round(time.monotonic() - start, 3), "window_images": len(tasks),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated()/2**20) if cuda else 0,
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved()/2**20) if cuda else 0,
        "oom_retries": oom_retries,
    }
    return [results[task["relative"]] | metrics for task in tasks]


def process_image(task, options, config, pipe):
    return process_batch([task], options, config, pipe)[0]


def worker_main(index, gpu, settings, model_path, requests, events, load_gate, quality_path=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config = DeploymentConfig.model_validate(settings)
    if config.expandable_segments:
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    try:
        with load_gate:
            pipe = None if config.backend == "mock" else load_pipeline(config, model_path)
            if pipe is not None and quality_path:
                import torch
                from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
                torch.cuda.empty_cache()
                pipe._quality_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                    quality_path, dtype=getattr(torch, config.dtype), local_files_only=True,
                ).to(pipe._execution_device).eval()
                pipe._quality_processor = Qwen3VLProcessor.from_pretrained(
                    quality_path, local_files_only=True,
                )
            elif pipe is not None and config.quality_verifier:
                pipe._quality_encoder, pipe._quality_processor = pipe.text_encoder, pipe.processor
        events.put({"type": "ready", "worker": index})
    except Exception as error:  # noqa: BLE001 - isolate file and GPU failures
        traceback.print_exc()
        events.put({"type": "fatal", "worker": index, "error": str(error)[:800]})
        return
    batch_limits = {}
    while True:
        request = requests.get()
        if request is None:
            return
        try:
            from .retry import process_attempts
            results = process_attempts(
                request["tasks"], ProcessingConfig.model_validate(request["options"]), config, pipe,
                batch_limits, request["job"], process_batch,
            )
        except Exception as error:  # noqa: BLE001 - isolate file and GPU failures
            traceback.print_exc()
            results = [
                {"file": task["relative"], "ok": False,
                 "error": f"{type(error).__name__}: {error}"[:800]}
                for task in request["tasks"]
            ]
            if config.backend != "mock":
                import torch
                torch.cuda.empty_cache()
        events.put({"type": "results", "worker": index, "job": request["job"], "results": results})
