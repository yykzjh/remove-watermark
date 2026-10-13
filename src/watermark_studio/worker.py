from __future__ import annotations

import hashlib
import os
import time
import traceback

from .config import DeploymentConfig, ProcessingConfig, Region
from .detection import locate_watermarks
from .images import (
    atomic_copy,
    blend_patch,
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
    if config.attention == "sdpa":
        from .attention import IndependentSDPAProcessor
        backend = config.attention_backend
        if backend.startswith("native_"):
            backend = "_" + backend
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
        quantize_transformer(pipe.transformer, config.quantization)
    pipe._watermark_prompt_encoder = config.prompt_encoder
    pipe._watermark_encoder_batch_size = config.encoder_batch_size
    if config.compile:
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
    cuda = config.backend != "mock"
    if cuda:
        import torch
        torch.cuda.reset_peak_memory_stats()
    for task in tasks:
        state = None
        try:
            source, exif, icc, alpha = load_photo(task["source"])
            detection_start = time.monotonic()
            if not cuda:
                regions = []
            elif options.scope == "full_image":
                regions = [Region(box=(0, 0, 1, 1))]
            elif options.preset == "auto":
                regions, _ = locate_watermarks(
                    source, pipe.text_encoder, pipe.processor, pipe._execution_device,
                    options.detection_resolution,
                )
                regions = [r for r in regions if options.remove_logo or r.kind != "logo"]
            else:
                regions = regions_for(source, options)
            detected = round(time.monotonic() - detection_start, 3)
            if not regions:
                if options.output_width:
                    if alpha is not None:
                        source.putalpha(alpha)
                    output_size = save_output(source, task["source"], task["destination"], exif, icc, options)
                else:
                    atomic_copy(task["source"], task["destination"], options.overwrite)
                    from PIL import Image
                    with Image.open(task["source"]) as raw:
                        output_size = list(raw.size)
                results[task["relative"]] = {
                    "file": task["relative"], "ok": True, "regions": [],
                    "action": "no_watermark_detected" if cuda else "mock_copy",
                    "detection_seconds": detected, "output_size": output_size,
                }
                continue
            state = {"task": task, "source": source, "result": source.copy(), "exif": exif, "icc": icc,
                         "alpha": alpha, "regions": regions, "detection_seconds": detected,
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


def worker_main(index, gpu, settings, model_path, requests, events, load_gate):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config = DeploymentConfig.model_validate(settings)
    try:
        with load_gate:
            pipe = None if config.backend == "mock" else load_pipeline(config, model_path)
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
            results = process_batch(
                request["tasks"], ProcessingConfig.model_validate(request["options"]), config, pipe,
                batch_limits,
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
