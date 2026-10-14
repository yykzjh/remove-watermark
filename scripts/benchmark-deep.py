"""Research matrix on one GPU per process. Never reuse image-conditioned embeddings."""
import argparse
import gc
import json
import os
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--expandable-segments", action="store_true")
parser.add_argument("--gpu", type=int)
parser.add_argument("--lane", type=int, required=True)
parser.add_argument("--case", default="")
parser.add_argument("--repeats", type=int, default=2)
args = parser.parse_args()
gpu = args.lane if args.gpu is None else args.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if args.expandable_segments:
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import torch
from PIL import Image

from watermark_studio.batching import independent_batch
from watermark_studio.config import DeploymentConfig
from watermark_studio.optimizations import lean_encode
from watermark_studio.prompts import build_full_prompt
from watermark_studio.worker import load_pipeline

root = Path(".run/deep-tuning")
root.mkdir(exist_ok=True)
samples = sorted(Path(".run/user-samples/input").glob("*.jpg"))
assert len(samples) == 2
items = [{"image": Image.open(samples[i % 2]).convert("RGB"), "prompt": build_full_prompt(),
          "seed": 42 + i} for i in range(8)]
base = {"gpus": [gpu], "expandable_segments": args.expandable_segments, "kv_cache": True, "vae_tiling": True, "vae_slicing": True, "tf32": True,
            "prompt_encoder": "lean", "encoder_batch_size": 2, "vae_tile_size": 384, "vae_tile_stride": 288}
# label, configuration overrides, inference batch, resolution
matrices = {
    0: [
        ("stock-b1", {"prompt_encoder": "stock", "vae_tile_size": 256, "vae_tile_stride": 192}, 1, 1024),
        ("stock-b2", {"prompt_encoder": "stock", "vae_tile_size": 256, "vae_tile_stride": 192}, 2, 1024),
        ("lean-serial-b1", {"encoder_batch_size": 1, "vae_tile_size": 256, "vae_tile_stride": 192}, 1, 1024),
        ("lean-serial-b2", {"encoder_batch_size": 1, "vae_tile_size": 256, "vae_tile_stride": 192}, 2, 1024),
        ("lean-batched-b2", {"vae_tile_size": 256, "vae_tile_stride": 192}, 2, 1024),
        ("lean-batched-b4", {"encoder_batch_size": 4, "vae_tile_size": 256, "vae_tile_stride": 192}, 4, 1024),
    ],
    1: [
        ("tile256", {"vae_tile_size": 256, "vae_tile_stride": 192}, 2, 1024),
        ("tile384", {}, 2, 1024),
        ("tile512", {"vae_tile_size": 512, "vae_tile_stride": 384}, 2, 1024),
        ("tile768", {"vae_tile_size": 768, "vae_tile_stride": 576}, 2, 1024),
        ("no-tiling-b2", {"vae_tiling": False}, 2, 1024),
        ("no-tiling-b1", {"vae_tiling": False}, 1, 1024),
    ],
    2: [(f"batch{n}", {}, n, 1024) for n in [1, 2, 3, 4, 6, 8]],
    3: [
        ("unfused", {}, 2, 1024),
        ("fused-b2", {"fuse_qkv": True}, 2, 1024),
        ("fused-b3", {"fuse_qkv": True}, 3, 1024),
        ("fused-b4", {"fuse_qkv": True}, 4, 1024),
    ],
    4: [
        ("compile-full", {"compile": True}, 2, 1024),
        ("compile-blocks", {"compile": True, "compile_scope": "blocks"}, 2, 1024),
        ("compile-blocks-dynamic", {"compile": True, "compile_scope": "blocks", "compile_dynamic": True}, 2, 1024),
        ("compile-blocks-autotune", {"compile": True, "compile_scope": "blocks",
                                         "compile_mode": "max-autotune-no-cudagraphs"}, 2, 1024),
    ],
    5: [
        ("fp8-b2", {"quantization": "float8"}, 2, 1024),
        ("int8-b2", {"quantization": "int8"}, 2, 1024),
        ("fp8-b4", {"quantization": "float8"}, 4, 1024),
        ("fp8-compile", {"quantization": "float8", "compile": True, "compile_scope": "blocks"}, 2, 1024),
    ],
    6: [(f"resolution{n}", {}, 2, n) for n in [512, 768, 1024, 1280, 1536, 2048]],
    7: [(f"threads{n}", {"cpu_threads": n}, 2, 1024) for n in [1, 2, 4, 8]] + [
        ("tile384-stride320", {"vae_tile_stride": 320}, 2, 1024),
        ("no-slicing", {"vae_slicing": False}, 2, 1024),
    ],
}
matrices[8] = [(f"kernel-{backend}", {"attention_backend": backend}, 2, 1024)
               for backend in ["native", "native_flash", "native_cudnn", "native_efficient", "flash_hub", "sage_hub"]]
matrices[9] = [
    ("allocator-b4", {}, 4, 1024),
    ("allocator-compile-full", {"compile": True}, 2, 1024),
    ("allocator-fp8-b4", {"quantization": "float8"}, 4, 1024),
    ("allocator-compile-b1", {"compile": True, "compile_scope": "blocks"}, 1, 1024),
    ("allocator-compile-b3", {"compile": True, "compile_scope": "blocks"}, 3, 1024),
]
matrices[10] = [(f"fp8-compiled-b{n}", {"quantization": "float8", "compile": True, "compile_scope": "blocks"}, n, 1024) for n in [1, 3, 4]]
matrices[11] = [
    ("fp8-projections", {"quantization": "float8", "compile": True, "compile_scope": "projections"}, 2, 1024),
    ("fp8-projections-dynamic", {"quantization": "float8", "compile": True, "compile_scope": "projections", "compile_dynamic": True}, 2, 1024),
]
# Second-pass candidates: compare on the deployed FP8 compiled recipe.
second = {"vae_tile_size": 256, "vae_tile_stride": 192, "quantization": "float8",
          "compile": True, "compile_scope": "projections", "compile_dynamic": True}
matrices.update({
    12: [("v2-control-b2", second, 2, 1024),
         ("v2-row-b2", second | {"float8_granularity": "row"}, 2, 1024)],
    13: [("v2-layout-b2", second | {"vae_channels_last": True}, 2, 1024),
         ("v2-cudnn-b2", second | {"cudnn_benchmark": True}, 2, 1024)],
    14: [("v2-layout-cudnn-b2", second | {"vae_channels_last": True, "cudnn_benchmark": True}, 2, 1024),
         ("v2-row-layout-b2", second | {"float8_granularity": "row", "vae_channels_last": True}, 2, 1024)],
    15: [("v2-projections-b1", second, 1, 1024),
         ("v2-projections-b3", second, 3, 1024)],
    16: [("v2-blocks-b1", second | {"compile_scope": "blocks", "compile_dynamic": False}, 1, 1024),
         ("v2-blocks-b2", second | {"compile_scope": "blocks", "compile_dynamic": False}, 2, 1024)],
    17: [("v2-row-blocks-b1", second | {"compile_scope": "blocks", "float8_granularity": "row"}, 1, 1024),
         ("v2-row-blocks-b2", second | {"compile_scope": "blocks", "float8_granularity": "row"}, 2, 1024)],
    18: [("v2-autotune-b1", second | {"compile_mode": "max-autotune-no-cudagraphs"}, 1, 1024),
         ("v2-autotune-b2", second | {"compile_mode": "max-autotune-no-cudagraphs"}, 2, 1024)],
    19: [("v2-row-b1", second | {"float8_granularity": "row"}, 1, 1024),
         ("v2-row-b3", second | {"float8_granularity": "row"}, 3, 1024)],
})
model = Path.home() / ".cache/watermark-models/models/Qwen--Qwen-Image-2.1-Turbo/snapshots/master"
for label, overrides, batch, resolution in matrices[args.lane]:
    if args.case and label != args.case:
        continue
    config = DeploymentConfig(**(base | overrides))
    pipe = None
    result = {"label": label, "gpu": gpu, "batch": batch, "resolution": resolution,
              "config": config.model_dump()}
    try:
        torch.cuda.empty_cache()
        started = time.monotonic()
        pipe = load_pipeline(config, str(model))
        result["load_s"] = round(time.monotonic() - started, 3)
        if args.lane == 0 and label == "stock-b1":
            resized = pipe.image_processor.resize(items[0]["image"].convert("RGBA"), width=576, height=448)
            with torch.inference_mode():
                expected = pipe.encode_prompt(prompt=items[0]["prompt"], image=[resized])
                actual = lean_encode(pipe, [{"prompt": items[0]["prompt"], "image": resized}])[0]
                torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
                if expected[1] is None:
                    assert actual[1].all()
                else:
                    torch.testing.assert_close(actual[1], expected[1])
                torch.testing.assert_close(actual[2], expected[2])
                paired = lean_encode(pipe, [{"prompt": items[0]["prompt"], "image": resized}] * 2)
                result["lean_single_embedding_exact"] = True
                result["lean_batched_embedding_maxdiff"] = float((paired[0][0] - expected[0]).abs().max())
                del expected, actual, paired
        times = []
        torch.cuda.reset_peak_memory_stats()
        for iteration in range(args.repeats + 1):
            torch.cuda.synchronize()
            started = time.monotonic()
            outputs = independent_batch(pipe, items[:batch], resolution, config.kv_cache)
            torch.cuda.synchronize()
            times.append(time.monotonic() - started)
        for index, image in enumerate(outputs[:2]):
            image.save(root / f"{label}-{index}.png")
        result.update(cold_s=round(times[0], 3), hot_s=[round(s, 3) for s in times[1:]],
                      images_per_s=round(batch / float(np.median(times[1:])), 4),
                      allocated_mib=round(torch.cuda.max_memory_allocated() / 2**20),
                      reserved_mib=round(torch.cuda.max_memory_reserved() / 2**20),
                      steps=len(pipe.config.sample_sigmas))
        del outputs
    except Exception as error:  # noqa: BLE001 - record unsupported/OOM configurations
        result["error"] = f"{type(error).__name__}: {error}"[:1200]
    with (root / f"lane-{args.lane}.jsonl").open("a") as stream:
        stream.write(json.dumps(result) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "config"}), flush=True)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    torch._dynamo.reset()
