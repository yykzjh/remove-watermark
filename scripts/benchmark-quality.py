"""Run real end-to-end quality-stage throughput comparisons on an existing service."""
import argparse
import json
import os
import random
import time
from pathlib import Path

from PIL import Image

from watermark_studio.cli import request

parser = argparse.ArgumentParser()
parser.add_argument("--tag", default="")
parser.add_argument("--images", type=int, default=24)
parser.add_argument("--concurrency", type=int, default=20)
parser.add_argument("--preserve-background", action="store_true")
parser.add_argument("--first-pass-mode",choices=["whole","focused","thorough","complete"],default="whole")
parser.add_argument("--repair-blending", choices=["feather","harmonic"], default="harmonic")
parser.add_argument("--quality-cross-review", action="store_true")
parser.add_argument("--quality-auto-retry", action="store_true")
parser.add_argument("--quality-max-attempts", type=int, default=3)
parser.add_argument("--url", default="http://127.0.0.1:18770")
parser.add_argument("--detail-patches", type=int, default=4)
parser.add_argument("--profiles", nargs="+")
parser.add_argument("--contact", action="store_true")
args = parser.parse_args()
url = args.url
root = Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031") / ("quality-benchmark" + ("-" + args.tag if args.tag else ""))
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
    if len(chosen) == args.images:
        break
assert len(chosen) == args.images
(root / "sources.json").write_text(json.dumps(chosen, ensure_ascii=False))
if args.contact:
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
    if args.profiles and label not in args.profiles:
        continue
    started = time.monotonic()
    job_id = request(url, "/jobs", {
        "input_dir": str(inputs), "output_dir": str(root / label),
        "options": {"concurrency": args.concurrency, "resolution": 1024, "overwrite": True,
                    "quality_check": enabled, "quality_cross_review": enabled and args.quality_cross_review,
                    "quality_auto_retry": enabled and args.quality_auto_retry,
                    "quality_max_attempts": args.quality_max_attempts, "preserve_background": args.preserve_background,
                    "first_pass_mode":args.first_pass_mode,"repair_blending": args.repair_blending, "quality_resolution": resolution,
                    "quality_batch_size": batch, "quality_detail_patches": args.detail_patches},
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
    quality_attempts = [attempt.get("quality") or {} for row in rows
                        for attempt in row.get("attempt_history", [{"quality": row.get("quality", {})}])]
    feedback_retries = sum(bool(attempt.get("repair_feedback_used")) for row in rows
                           for attempt in row.get("attempt_history", []))
    record = {"label": label, "tag": args.tag, "images": args.images, "detail_patches": args.detail_patches, "seconds": round(time.monotonic()-started, 3),
              "job": job_id, "success": job["succeeded"], "failed": job["failed"],
              "quality_counts": job.get("quality_counts", {}),
              "retry_attempts": job.get("retry_attempts", 0),
              "retry_exhausted": job.get("retry_exhausted", 0),
              "first_pass_mode":args.first_pass_mode,
              "first_pass_counts":job.get("first_pass_counts",{}),
              "first_pass_existing":job.get("first_pass_existing",0),
              "preserve_background": args.preserve_background,
              "repair_blending": args.repair_blending,
              "cross_review_attempts": sum(q.get("cross_review", {}).get("attempted", False) for q in quality_attempts),
              "cross_review_corrected": sum(q.get("cross_review", {}).get("status") == "corrected" for q in quality_attempts),
              "cross_review_errors": sum(q.get("cross_review", {}).get("status") == "error" for q in quality_attempts),
              "cross_review_seconds": round(sum(q.get("cross_review", {}).get("seconds", 0) for q in quality_attempts), 3),
              "explained_attempts": sum(q.get("model_explanation", {}).get("status") in {"ok", "summary_only", "adjudicated"} for q in quality_attempts),
              "explanation_errors": sum(q.get("model_explanation", {}).get("status") == "error" for q in quality_attempts),
              "explanation_seconds": round(sum(q.get("explanation_seconds", 0) for q in quality_attempts), 3),
              "explanation_oom_splits_max": max((q.get("model_explanation", {}).get("oom_splits", 0) for q in quality_attempts), default=0),
              "feedback_retries": feedback_retries,
              "quality_seconds": round(sum(r.get("quality_total_seconds", r.get("quality", {}).get("seconds", 0)) for r in rows), 3),
              "bad_sizes": sum(Image.open(inputs/r["file"]).size != Image.open(root/label/r["file"]).size
                               for r in rows if r["ok"]),
              "quality_oom_splits_max": max((r.get("quality_oom_splits",0) for r in rows),default=0),
              "peak_allocated_mib": max((r.get("peak_allocated_mib",0) for r in rows),default=0),
              "quality_errors": [r["quality"] for r in rows
                                 if r.get("quality", {}).get("status") == "error"][:2]}
    with (root / "results.jsonl").open("a") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)
    if job["failed"] or state["stage"] == "failed":
        raise RuntimeError("Benchmark job failed")
