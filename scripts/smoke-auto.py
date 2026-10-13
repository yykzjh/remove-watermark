"""End-to-end automatic localization on varied positions and an unmarked image."""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from watermark_studio.cli import request
from watermark_studio.config import Region
from watermark_studio.images import pixel_box

root = Path(".run/smoke-auto").resolve()
source, target = root/"input", root/"output"
source.mkdir(parents=True, exist_ok=True)
base = Image.open(".run/smoke/input/synthetic.png").convert("RGB")
panel = base.crop((10, 420, 346, 561))
ImageDraw.Draw(base).rectangle((10, 420, 346, 561), fill=(135, 125, 90))
truth = {}
for name, position in [
    ("upper_right", (420, 12)), ("center", (210, 230)),
    ("upper_left", (12, 12)), ("lower_right", (420, 420)),
    ("lower_left", (12, 420)), ("middle_right", (420, 230)),
]:
    image = base.copy()
    image.paste(panel, position)
    image.save(source/f"{name}.png")
    x, y = position
    truth[f"{name}.png"] = (x, y, x+336, y+141)
multiple = base.copy()
multiple.paste(panel, (12, 12))
draw = ImageDraw.Draw(multiple)
font = ImageFont.load_default(size=18)
draw.text((530, 500), "TODAY WATERMARK", fill="white", font=font)
draw.text((530, 525), "CAMERA TRUE TIME", fill="white", font=font)
draw.text((530, 550), "CODE A123456", fill="white", font=font)
multiple.save(source/"multiple.png")
truth["multiple.png"] = (12, 12, 348, 153)
base.save(source/"unmarked.png")
url = "http://127.0.0.1:8765"
response = request(url, "/jobs", {
    "input_dir": str(source), "output_dir": str(target),
    "options": {"preset": "auto", "concurrency": 7, "resolution": 768, "overwrite": True},
})
start = time.monotonic()
while time.monotonic()-start < 600:
    job = request(url, "/health")["job"]
    if job["state"] in {"completed", "failed", "cancelled"}:
        assert job["succeeded"] == 8, job
        results = [
            json.loads(line) for line in
            (Path.home()/".local/state/watermark-studio"/job["id"]/"results.jsonl")
            .read_text().splitlines()
        ]
        assert len({r["worker"] for r in results}) == 7, results
        for result in results:
            name = result["file"]
            if name == "unmarked.png":
                assert result["action"] == "no_watermark_detected", result
                assert (source/name).read_bytes() == (target/name).read_bytes()
                continue
            assert result["action"] == "restored", result
            if name == "multiple.png":
                assert len(result["regions"]) == 2, result
                assert any(r["kind"] == "logo" for r in result["regions"]), result
            original = np.asarray(Image.open(source/name))
            output = np.asarray(Image.open(target/name))
            allowed = np.zeros(original.shape[:2], dtype=bool)
            tx0, ty0, tx1, ty1 = truth[name]
            best_coverage = 0
            for value in result["regions"]:
                region = Region.model_validate(value)
                x0, y0, x1, y1 = pixel_box(region.box, (original.shape[1], original.shape[0]))
                overlap = max(0, min(x1, tx1)-max(x0, tx0))*max(0, min(y1, ty1)-max(y0, ty0))
                best_coverage = max(best_coverage, overlap/((tx1-tx0)*(ty1-ty0)))
                allowed[max(0,y0-12):y1+12, max(0,x0-12):x1+12] = True
            assert best_coverage > 0.95, result
            assert np.array_equal(original[~allowed], output[~allowed]), result
            assert not np.array_equal(original[ty0:ty1,tx0:tx1], output[ty0:ty1,tx0:tx1])
        print(json.dumps({"ok": True, "seconds": round(time.monotonic()-start, 2), "job": job["id"], "results": results}, ensure_ascii=False))
        break
    time.sleep(2)
else:
    raise SystemExit("Automatic batch timed out")
