from __future__ import annotations

import hashlib
import os
import time
import traceback

from .config import DeploymentConfig, ProcessingConfig
from .images import atomic_save, blend_patch, context_box, load_photo, pixel_box, regions_for
from .prompts import build_prompt


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
        pipe.vae.enable_tiling()
    if config.vae_slicing:
        pipe.vae.enable_slicing()
    if config.offload == "model":
        pipe.enable_model_cpu_offload(gpu_id=0)
    elif config.offload == "sequential":
        pipe.enable_sequential_cpu_offload(gpu_id=0)
    else:
        pipe.to("cuda:0")
    if config.attention == "flex":
        from diffusers.models.transformers.transformer_qwenimage21 import (
            QwenImage21FlexAttnProcessor,
        )

        pipe.transformer.set_attn_processor(QwenImage21FlexAttnProcessor())
    if config.compile:
        pipe.transformer.compile(mode=config.compile_mode)
    if not getattr(pipe.config, "sample_sigmas", None):
        raise RuntimeError("模型缺少 Turbo sample_sigmas；请检查模型及 Diffusers 版本")
    return pipe


def process_image(task, options, config, pipe):
    start = time.monotonic()
    source, exif, icc, alpha = load_photo(task["source"])
    result = source.copy()
    stable_seed = int.from_bytes(hashlib.sha256(task["relative"].encode()).digest()[:4], "big")
    for index, region in enumerate(regions_for(source, options)):
        box = pixel_box(region.box, source.size)
        crop = context_box(box, source.size, options.context)
        reference = source.crop(crop)
        if config.backend == "mock":
            # Explicit test backend: copying is not claimed to remove any watermark.
            patch = reference
        else:
            import torch

            with torch.inference_mode():
                patch = pipe(
                    prompt=build_prompt(region.kind, box, crop, options.extra_prompt),
                    image=reference,
                    output_resolution=options.resolution,
                    true_cfg_scale=1.0,
                    use_kv_cache=config.kv_cache,
                    generator=torch.Generator("cpu").manual_seed(
                        (options.seed + stable_seed + index) % (2**32)
                    ),
                ).images[0]
        result = blend_patch(result, patch, box, crop, options.feather)
    if alpha is not None:
        result.putalpha(alpha)
    atomic_save(result, task["destination"], exif, icc, options.overwrite)
    return {"file": task["relative"], "ok": True, "seconds": round(time.monotonic() - start, 3)}


def worker_main(index, gpu, settings, model_path, requests, events, load_gate):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config = DeploymentConfig.model_validate(settings)
    try:
        with load_gate:
            pipe = None if config.backend == "mock" else load_pipeline(config, model_path)
        events.put({"type": "ready", "worker": index})
    except Exception as error:  # noqa: BLE001 - isolate worker/service failures
        traceback.print_exc()
        events.put({"type": "fatal", "worker": index, "error": str(error)[:800]})
        return
    while True:
        request = requests.get()
        if request is None:
            return
        try:
            result = process_image(
                request["task"], ProcessingConfig.model_validate(request["options"]), config, pipe
            )
        except Exception as error:  # noqa: BLE001 - isolate worker/service failures
            traceback.print_exc()
            result = {
                "file": request["task"]["relative"],
                "ok": False,
                "error": f"{type(error).__name__}: {error}"[:800],
            }
            if config.backend != "mock":
                import torch

                torch.cuda.empty_cache()
        events.put({"type": "result", "worker": index, "job": request["job"], **result})
