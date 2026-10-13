"""GPU-local reproducible microbatch tuning; writes artifacts locally, prints small summaries."""
import argparse
import json
import os
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--profile", choices=["base", "compile", "offload", "optimized"], default="base")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
import numpy as np
import torch
from PIL import Image, ImageDraw

from watermark_studio.batching import independent_batch
from watermark_studio.config import DeploymentConfig
from watermark_studio.worker import load_pipeline

root = Path(".run/tuning")
root.mkdir(parents=True, exist_ok=True)
model = Path.home()/".cache/watermark-models/models/Qwen--Qwen-Image-2.1-Turbo/snapshots/master"
config = DeploymentConfig(gpus=[args.gpu], kv_cache=True, vae_slicing=True, vae_tiling=True)
pipe = load_pipeline(config, str(model))
items = []
for i in range(4):
    image = Image.new("RGB", (800, 600), [(88,130,64),(140,95,66),(67,109,141),(142,140,71)][i])
    draw = ImageDraw.Draw(image)
    for y in range(0, 600, 24):
        draw.line((0,y,800,y+130), fill=(110+i*12,115,90), width=3)
    draw.rectangle((20,400,360,570), fill=(220,220,220))
    draw.text((35,420), "CONSTRUCTION SITE 2024 / WATERMARK", fill="black")
    image.save(root/f"reference-{i}.png")
    items.append({"image": image, "prompt": "Remove the digital watermark information panel. "
                  "Naturally reconstruct the covered soil and grass. Preserve all other content."
                  + (" Keep the original lighting unchanged." if i % 2 else ""),
                  "seed": 123+i})

def run(label, batch=2, resolution=1024, kv=True, repeats=3):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        start = time.monotonic()
        outputs = independent_batch(pipe, items[:batch], resolution, kv)
        torch.cuda.synchronize()
        cold = time.monotonic()-start
        seconds = []
        for _ in range(repeats):
            start = time.monotonic()
            outputs = independent_batch(pipe, items[:batch], resolution, kv)
            torch.cuda.synchronize()
            seconds.append(time.monotonic()-start)
        for i, output in enumerate(outputs):
            output.save(root/f"{label}-{i}.png")
        result = {"label":label, "batch":batch, "resolution":resolution,
                  "cold_s":round(cold,3), "median_s":round(float(np.median(seconds)),3),
                  "images_per_s":round(batch/float(np.median(seconds)),3),
                  "peak_allocated_mib":round(torch.cuda.max_memory_allocated()/2**20),
                  "peak_reserved_mib":round(torch.cuda.max_memory_reserved()/2**20)}
    except Exception as error:  # noqa: BLE001 - record benchmark failure
        result = {"label": label, "error": f"{type(error).__name__}: {error}"[:400]}
    with (root/f"{args.profile}.jsonl").open("a") as stream:
        stream.write(json.dumps(result)+"\n")
    print(json.dumps(result), flush=True)
    return result

if args.profile in {"base", "optimized"}:
    from watermark_studio.attention import attention
    q = torch.randn(2,23,2,32,device="cuda",dtype=torch.bfloat16)
    k = torch.randn(2,31,2,32,device="cuda",dtype=torch.bfloat16)
    v = torch.randn_like(k)
    mask = torch.ones(2,1,1,31,device="cuda",dtype=torch.bool)
    mask[0,:,:,10:15] = False
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),attn_mask=mask
    ).transpose(1,2)
    torch.testing.assert_close(attention(q,k,v,mask), reference, atol=.004, rtol=.02)
    # Verify singleton adapter agrees with the stock independent-image call.
    expected = pipe(image=items[0]["image"], prompt=items[0]["prompt"],
                    output_resolution=512, true_cfg_scale=1, use_kv_cache=True,
                    generator=torch.Generator("cpu").manual_seed(123)).images[0]
    actual = independent_batch(pipe, items[:1], 512, True)[0]
    difference = float(np.abs(np.asarray(expected).astype(float)-np.asarray(actual)).mean())
    assert difference < 0.1, difference
    pair = independent_batch(pipe, items[:2], 512, True)
    single = [independent_batch(pipe, [item], 512, True)[0] for item in items[:2]]
    errors = [float(np.abs(np.asarray(a).astype(float)-np.asarray(b)).mean())
              for a,b in zip(pair,single)]
    # BF16 GEMM batch rounding changes generative pixels; detect swapped/shared references.
    own = [float(np.abs(np.asarray(pair[i]).astype(float)-np.asarray(single[i])).mean())
           for i in range(2)]
    crossed = [float(np.abs(np.asarray(pair[i]).astype(float)-np.asarray(single[1-i])).mean())
               for i in range(2)]
    assert all(a < b for a,b in zip(own,crossed)), (own,crossed)
    print(json.dumps({"verification":"independent_conditions","singleton_mae":difference,
                      "batch_single_mae":errors,"crossed_mae":crossed}), flush=True)
    for batch in (1,2,3):
        run(f"b{batch}-1024", batch=batch)
    run("b2-no-kv", kv=False)
    for threads in (1,8):
        torch.set_num_threads(threads)
        run(f"threads-{threads}")
    torch.set_num_threads(4)
    pipe.vae.disable_tiling()
    run("no-vae-tiling")
    pipe.vae.disable_slicing()
    run("no-vae-slicing-tiling")
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    torch.backends.cuda.matmul.allow_tf32 = True
    run("tf32")
    for resolution in (768,1280):
        run(f"b2-{resolution}", resolution=resolution)
elif args.profile == "compile":
    pipe.transformer.compile(mode="default")
    run("compile-sdpa", repeats=2)
    pipe.transformer._compiled_call_impl = None
    torch._dynamo.reset()
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21FlexAttnProcessor
    pipe.transformer.set_attn_processor(QwenImage21FlexAttnProcessor())
    pipe.transformer.compile(mode="default")
    run("compile-flex", repeats=2)
else:
    pipe.enable_model_cpu_offload(gpu_id=0)
    run("model-offload", repeats=2)
