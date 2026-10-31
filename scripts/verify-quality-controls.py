"""Check that the reviewer rejects unchanged watermarks and obvious content damage."""
import json
import os
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from PIL import Image, ImageDraw, ImageOps
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from watermark_studio.download import download_model
from watermark_studio.quality import review_batch

snapshot = Path.home()/".cache/watermark-models/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master"
if not all((snapshot/f"model-{i:05d}-of-00002.safetensors").is_file() for i in (1, 2)):
    snapshot = Path(download_model(
        "Qwen/Qwen3-VL-4B-Instruct", "modelscope", "master", "~/.cache/watermark-models",
    ))
encoder = Qwen3VLForConditionalGeneration.from_pretrained(
    snapshot, dtype=torch.bfloat16, local_files_only=True,
).to("cuda").eval()
processor = Qwen3VLProcessor.from_pretrained(snapshot, local_files_only=True)
root = Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031/quality-benchmark")
sources = sorted((root/"input").glob("*"))
cases = []
for index in (0, 1, 3, 6, 12, 19):
    with Image.open(sources[index]) as im:
        before = ImageOps.exif_transpose(im).convert("RGB")
    cases.append((f"watermark_retained_{index}", before, before.copy()))
for index in (3, 12):
    with Image.open(sources[index]) as im:
        before = ImageOps.exif_transpose(im).convert("RGB")
    damaged = before.copy()
    ImageDraw.Draw(damaged).rectangle((0, 0, before.width//2, before.height//2), fill=(0, 255, 0))
    cases.append((f"content_destroyed_{index}", before, damaged))
for resolution in (512, 768, 1024):
    for label, before, after in cases:
        started = time.monotonic()
        review = review_batch([(before, after)], encoder, processor, "cuda", resolution)[0]
        row = {"case": label, "model": "Qwen/Qwen3-VL-4B-Instruct", "resolution": resolution, "seconds": round(time.monotonic()-started, 3),
               **review}
        with (root/"controls-independent.jsonl").open("a") as out:
            out.write(json.dumps(row, ensure_ascii=False)+"\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
