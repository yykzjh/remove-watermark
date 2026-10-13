"""Synthetic fixture generated on the GPU machine; never sends image bytes over SSH."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from watermark_studio.cli import request

parser = argparse.ArgumentParser()
parser.add_argument("--resolution", type=int, default=512)
parser.add_argument("--url", default="http://127.0.0.1:8765")
args = parser.parse_args()
root = Path(".run/smoke").resolve()
source, target = root/"input", root/"output"
source.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
pixels = np.clip(rng.normal([125, 113, 74], 15, size=(576, 768, 3)), 0, 255).astype("uint8")
image = Image.fromarray(pixels)
draw = ImageDraw.Draw(image)
draw.rectangle((0, 0, 768, 170), fill=(167, 192, 210))
draw.polygon([(0, 170), (90, 60), (160, 170)], fill=(58, 93, 47))
draw.polygon([(580, 170), (670, 35), (767, 170)], fill=(52, 87, 45))
draw.polygon([(320, 170), (400, 170), (660, 576), (50, 576)], fill=(156, 142, 109))
draw.rectangle((10, 420, 345, 560), fill=(212, 215, 215))
draw.rectangle((10, 420, 345, 460), fill=(50, 125, 209))
draw.text((25, 435), "SITE INSPECTION", fill="white")
draw.text((25, 478), "Date: 2024.03.21   14:15", fill="black")
draw.text((25, 520), "Location: construction site", fill="black")
image.save(source/"synthetic.png")
health = request(args.url, "/health")
if health["backend"] != "qwen" or health["stage"] != "ready":
    raise SystemExit("A ready Qwen backend is required")
start = time.monotonic()
request(args.url, "/jobs", {
    "input_dir": str(source), "output_dir": str(target),
    "options": {"preset": "custom", "regions": [
        {"kind": "panel", "box": [0.01, 0.72, 0.46, 0.98]}],
        "concurrency": 1, "resolution": args.resolution, "overwrite": True}})
while time.monotonic() - start < 1800:
    state = request(args.url, "/health")
    job = state["job"]
    if job["state"] in {"completed", "failed", "cancelled"}:
        if job["succeeded"] != 1:
            raise SystemExit(json.dumps(job, ensure_ascii=False))
        with Image.open(target/"synthetic.png") as result:
            delta = np.abs(np.asarray(result).astype(float)-np.asarray(image).astype(float))
            assert delta[:300].max() == 0, "Pixels outside repair area changed"
            assert delta[420:560, 10:345].mean() > 0.5, "No generated edit observed"
        print(json.dumps({"ok": True, "seconds": round(time.monotonic()-start, 2),
                          "resolution": args.resolution,
                          "changed_mean": round(float(delta[420:560, 10:345].mean()), 2),
                          "job": job["id"]}))
        break
    time.sleep(5)
else:
    raise SystemExit("Inference timeout; inspect remote service log")
