"""End-to-end eight-GPU batch/concurrency benchmark; all fixtures remain on server."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from watermark_studio.cli import request
from watermark_studio.config import Region
from watermark_studio.images import pixel_box

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["custom", "auto"], default="custom")
args = parser.parse_args()
root = Path(".run/tuning/e2e").resolve()
source = root/args.mode/"input"
source.mkdir(parents=True, exist_ok=True)
# Different dimensions, including non-multiples of 32, portrait and original 4K photos.
sizes = [(800,600), (1600,1200), (4000,3000), (1707,1279)]
for i in range(32 if args.mode == "custom" else 16):
    width, height = sizes[i % len(sizes)]
    if args.mode == "auto" and i % 5 == 0:
        width, height = height, width
    image = Image.new("RGB", (width,height), (100+i%10,130,75))
    draw = ImageDraw.Draw(image)
    for y in range(0,height, max(1,height//40)):
        draw.line((0,y,width,y+height//5), fill=(125,105,65), width=max(1,width//250))
    panel = (int(width*.05),int(height*.65),int(width*.45),int(height*.94))
    draw.rectangle(panel, fill=(225,225,225))
    # Render at a fixed logical scale then enlarge, so detection can read the text.
    overlay = Image.new("RGB",(400,174),(225,225,225))
    od = ImageDraw.Draw(overlay)
    from PIL import ImageFont
    font = ImageFont.load_default(size=22)
    for j,text in enumerate(["CONSTRUCTION SITE", "DATE: 2024-03-21", "LOCATION: FARM ROAD", "CAMERA WATERMARK"]):
        od.text((10,10+j*36),text,fill="black",font=font)
    image.paste(overlay.resize((panel[2]-panel[0],panel[3]-panel[1])), panel[:2])
    folder = source/f"nested-{i%3}"
    folder.mkdir(exist_ok=True)
    image.save(folder/f"{i:02}.png")
(source/"ignore.txt").write_text("non-image")
(source/"invalid.jpg").write_text("invalid image")
url = "http://127.0.0.1:8765"
health = request(url,"/health")
for concurrency in ([8,16,32,64] if args.mode == "custom" else [16,32]):
    if concurrency > health["capacity"]:
        continue
    for detection in ([1280] if args.mode == "custom" else [768,1280]):
        target = root/args.mode/f"output-c{concurrency}-d{detection}"
        options = {
            "concurrency": concurrency, "preset":args.mode, "resolution":1024,
            "detection_resolution":detection, "remove_logo":True, "overwrite":True,
        }
        if args.mode == "custom":
            options["regions"] = [{"box":[.05,.65,.45,.94]}]
        response = request(url,"/jobs",{
            "input_dir":str(source),"output_dir":str(target),"options":options,
        })
        start = time.monotonic()
        while time.monotonic()-start < 1800:
            job = request(url,"/health")["job"]
            if job["id"] != response["id"]:
                raise RuntimeError("Another job replaced the benchmark")
            if job["state"] in {"completed","failed","cancelled"}:
                break
            time.sleep(.25)
        assert job["state"] == "completed" and job["failed"] == 0, job
        elapsed = time.monotonic()-start
        rows = [json.loads(line) for line in
                (Path.home()/".local/state/watermark-studio"/job["id"]/"results.jsonl")
                .read_text().splitlines()]
        assert len({row["worker"] for row in rows}) == 8, "Not all GPUs were used"
        coverages = []
        for row in rows:
            original = Image.open(source/row["file"])
            output = Image.open(target/row["file"])
            assert original.size == output.size, row
            before, after = np.asarray(original), np.asarray(output)
            allowed = np.zeros(before.shape[:2],dtype=bool)
            covered = np.zeros_like(allowed)
            for value in row["regions"]:
                x0,y0,x1,y1 = pixel_box(Region.model_validate(value).box,original.size)
                allowed[max(0,y0-12):y1+12,max(0,x0-12):x1+12] = True
                covered[y0:y1,x0:x1] = True
            assert np.array_equal(before[~allowed],after[~allowed]), row
            width, height = original.size
            truth = covered[int(height*.65):int(height*.94), int(width*.05):int(width*.45)]
            coverages.append(float(truth.mean()))
            if row["action"] == "restored":
                assert not np.array_equal(before[allowed],after[allowed]), row
        report = {
            "mode":args.mode, "concurrency":concurrency, "detection_resolution":detection,
            "seconds":round(elapsed,3), "images":len(rows),
            "job":job["id"], "workers":len({r["worker"] for r in rows}),
            "batch_sizes":sorted({b for r in rows for b in r.get("batch_sizes",[])}),
            "peak_allocated_mib":max(r["peak_allocated_mib"] for r in rows),
            "peak_reserved_mib":max(r["peak_reserved_mib"] for r in rows),
            "min_watermark_coverage":round(min(coverages),4),
            "undercovered_images":sum(value < .95 for value in coverages),
            "no_watermark":job["no_watermark"], "dimensions_exact":True,
        }
        with (root/"results.jsonl").open("a") as f:
            f.write(json.dumps(report)+"\n")
        print(json.dumps(report),flush=True)
