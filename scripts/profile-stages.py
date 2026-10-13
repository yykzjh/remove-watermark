"""GPU stage timing; use only an idle GPU, not a production worker."""
import argparse
import json
import os
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=2)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
import torch
from PIL import Image

from watermark_studio.batching import independent_batch
from watermark_studio.config import DeploymentConfig
from watermark_studio.prompts import build_full_prompt
from watermark_studio.worker import load_pipeline

model = Path.home() / ".cache/watermark-models/models/Qwen--Qwen-Image-2.1-Turbo/snapshots/master"
pipe = load_pipeline(DeploymentConfig(gpus=[args.gpu], kv_cache=True, vae_tiling=True,
                                     vae_slicing=True, prompt_encoder="lean"), str(model))
photo = next(Path(".run/user-samples/input").glob("*.jpg"))
items = [{"image": Image.open(photo).convert("RGB"), "prompt": build_full_prompt(), "seed": 42}]
independent_batch(pipe, items, 1024, True)
events = []
def wrapper(fn, name):
    def measured(*args, **kwargs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn(*args, **kwargs)
        end.record()
        events.append((name, start, end))
        return result
    return measured

for module, name, attr in [(pipe.text_encoder.model, "encoder", "forward"),
                           (pipe.vae, "vae_encode", "encode"), (pipe.vae, "vae_decode", "decode"),
                           (pipe.transformer, "transformer", "forward")]:
    setattr(module, attr, wrapper(getattr(module, attr), name))
torch.cuda.synchronize()
start = time.monotonic()
independent_batch(pipe, items, 1024, True)
torch.cuda.synchronize()
summary = {"wall_s": round(time.monotonic() - start, 3)}
for name in {row[0] for row in events}:
    times = [a.elapsed_time(b) / 1000 for key, a, b in events if key == name]
    summary[name] = {"calls": len(times), "total_s": round(sum(times), 3),
                     "first_s": round(times[0], 3)}
Path(".run/deep-tuning/stages.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary))
