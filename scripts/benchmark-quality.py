"""Run real end-to-end quality-stage throughput comparisons on an existing service."""
import json
import os
import random
import sys
import time
from pathlib import Path

from PIL import Image

from watermark_studio.cli import request

url = "http://127.0.0.1:18770"
root = Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031/quality-benchmark")
inputs = root / "input"
inputs.mkdir(parents=True, exist_ok=True)
data = Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031/八标照片集")
manifest = json.loads((data / ".download-state/manifest.json").read_text())
cloud = "/codex-download-1mQf6F6g-20261031/八标照片集/"
candidates = [data / r["path"].removeprefix(cloud) for r in manifest
              if Path(r["path"]).suffix.lower() in {".jpg", ".png"}]
random.Random(42).shuffle(candidates)
chosen = []
for f in candidates:
    if not f.is_file() or not f.stat().st_size:
        continue
    try:
        with Image.open(f) as im:
            im.load()
    except OSError:
        continue
    target = inputs / (str(len(chosen)).zfill(3) + f.suffix)
    if not target.exists():
        os.link(f, target)
    chosen.append(str(f))
    if len(chosen) == 24:
        break
assert len(chosen) == 24
(root / "sources.json").write_text(json.dumps(chosen, ensure_ascii=False))
if "--contact" in sys.argv:
    from PIL import ImageDraw, ImageOps
    for page in range(3):
        sheet = Image.new("RGB", (768, 336), "white")
        draw = ImageDraw.Draw(sheet)
        for index, f in enumerate(chosen[page*8:page*8+8]):
            with Image.open(f) as image:
                thumb = ImageOps.contain(image.convert("RGB"), (192, 144))
            x, y = (index % 4)*192, (index // 4)*168
            sheet.paste(thumb, (x, y+20))
            draw.text((x+3, y+2), str(page*8+index), fill="black")
        sheet.save(root / f"contact-{page}.jpg", quality=45)
        assert (root / f"contact-{page}.jpg").stat().st_size < 65536
    print(str(root))
    raise SystemExit
while request(url, "/health")["stage"] != "ready":
    time.sleep(2)
profiles = [
    ("baseline-cold", False, 768, 1),
    ("baseline-hot", False, 768, 1),
    ("quality-512-b1", True, 512, 1),
    ("quality-512-b2", True, 512, 2),
    ("quality-768-b1", True, 768, 1),
    ("quality-768-b2", True, 768, 2),
]
for label, enabled, resolution, batch in profiles:
    started = time.monotonic()
    job_id = request(url, "/jobs", {
        "input_dir": str(inputs), "output_dir": str(root / label),
        "options": {"concurrency": 20, "resolution": 1024, "overwrite": True,
                    "quality_check": enabled, "quality_resolution": resolution,
                    "quality_batch_size": batch},
    })["id"]
    while True:
        state = request(url, "/health")
        job = state["job"]
        if state["stage"] == "failed" or job["state"] in {"completed", "failed", "cancelled"}:
            break
        if time.monotonic() - started > 1800:
            raise TimeoutError(label)
        time.sleep(.5)
    rows = [json.loads(line) for line in
            (Path.home()/".local/state/watermark-studio"/job_id/"results.jsonl").read_text().splitlines()]
    record = {"label": label, "seconds": round(time.monotonic()-started, 3),
              "job": job_id, "success": job["succeeded"], "failed": job["failed"],
              "quality_counts": job.get("quality_counts", {}),
              "quality_seconds": round(sum(r.get("quality", {}).get("seconds", 0) for r in rows), 3),
              "bad_sizes": sum(Image.open(inputs/r["file"]).size != Image.open(root/label/r["file"]).size
                               for r in rows if r["ok"]),
              "quality_errors": [r["quality"] for r in rows
                                 if r.get("quality", {}).get("status") == "error"][:2]}
    with (root / "results.jsonl").open("a") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)
    if job["failed"] or state["stage"] == "failed":
        raise RuntimeError("Benchmark job failed")
