"""Verify that the deployed encoder can locate overlays without fixed positions."""
import json
import os
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from PIL import Image, ImageDraw
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from watermark_studio.detection import locate_watermarks

root = Path.home()/".cache/watermark-models/models/Qwen--Qwen-Image-2.1-Turbo/snapshots/master"
encoder = Qwen3VLForConditionalGeneration.from_pretrained(
    root/"text_encoder", dtype=torch.bfloat16, local_files_only=True,
).to("cuda").eval()
processor = Qwen3VLProcessor.from_pretrained(root/"processor", local_files_only=True)
base = Image.open(".run/smoke/input/synthetic.png").convert("RGB")
panel = base.crop((10, 420, 346, 561))
ImageDraw.Draw(base).rectangle((10, 420, 346, 561), fill=(135, 125, 90))
for name, position in [("upper_right", (420, 12)), ("center", (210, 230))]:
    image = base.copy()
    image.paste(panel, position)
    start = time.monotonic()
    regions, text = locate_watermarks(image, encoder, processor, "cuda")
    print(json.dumps({"case": name, "seconds": round(time.monotonic()-start, 2),
                      "regions": [r.model_dump() for r in regions], "raw": text}),
          flush=True)
