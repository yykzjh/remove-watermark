"""Validate overview/detail reviews on known retained overlays and local defects."""
import argparse
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("QUALITY_TEST_GPU", "1")

import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from watermark_studio.quality import verify_batch

torch.set_num_threads(1)

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=["resident", "independent"], default="resident")
parser.add_argument("--combined", action="store_true", help="Include removed overlays with subtle new defects")
parser.add_argument("--compare-details", action="store_true", help="Experimental paired content detail comparison")
args = parser.parse_args()
if args.compare_details and not args.combined:
    parser.error("--compare-details requires --combined")
folder = "Qwen--Qwen-Image-2.1-Turbo" if args.backend == "resident" else "Qwen--Qwen3-VL-4B-Instruct"
snapshot = Path.home()/".cache/watermark-models/models"/folder/"snapshots/master"
encoder = Qwen3VLForConditionalGeneration.from_pretrained(
    snapshot/"text_encoder" if args.backend == "resident" else snapshot, dtype=torch.bfloat16, local_files_only=True,
).to("cuda").eval()
processor = Qwen3VLProcessor.from_pretrained(
    snapshot/"processor" if args.backend == "resident" else snapshot, local_files_only=True)
root = Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031/quality-benchmark")
files = sorted((root/"input").glob("*"))
cases = []
def photo(index):
    with Image.open(files[index]) as im:
        return ImageOps.exif_transpose(im).convert("RGB")
for index in (0, 1, 3, 12, 19):
    before = photo(index)
    cases.append((f"retained_watermark_{index}", before, before.copy(), "review"))
clean = photo(6)
cases.append(("clean_unchanged", clean, clean.copy(), "passed"))
for index in (3, 12):
    before = photo(index)
    after = before.copy()
    ImageDraw.Draw(after).rectangle((0, 0, before.width//2, before.height//2), fill="lime")
    cases.append((f"flat_fill_{index}", before, after, "review"))
w, h = clean.size
after = clean.copy()
overlay = Image.new("RGBA", clean.size, (0, 0, 0, 0))
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(20, w//100))
ImageDraw.Draw(overlay).text((int(w*.58), int(h*.85)), "SITE CAMERA 2026", font=font, fill=(255,255,255,32))
after = Image.alpha_composite(after.convert("RGBA"), overlay).convert("RGB")
cases.append(("faint_text_residue", clean, after, "review"))
after = clean.copy()
box = (int(w*.80), int(h*.25), int(w*.94), int(h*.55))
after.paste(after.crop(box).filter(ImageFilter.GaussianBlur(max(2, w/900))), box)
cases.append(("localized_person_blur", clean, after, "review"))
after = clean.copy()
box = (int(w*.46), int(h*.51), int(w*.66), int(h*.71))
patch = after.crop(box)
after.paste(patch, (box[0]+max(3, w//500), box[1]))
cases.append(("small_edge_displacement", clean, after, "review"))
after = clean.copy()
box = (int(w*.85), int(h*.27), int(w*.90), int(h*.34))
patch = after.crop((box[0]-int(w*.07), box[1], box[2]-int(w*.07), box[3]))
after.paste(patch, box)
cases.append(("small_head_region_replaced", clean, after, "review"))
# Known-good removals: overlays are added to a real clean photo, then the exact
# clean photo is used as output. Hidden ground truth is known for these fixtures.
combined_before = None
for label, location in (("bottom_left", (.02, .72)), ("top_right", (.60, .02)), ("center", (.30, .42))):
    panel = Image.new("RGBA", (int(w*.36), int(h*.25)), (225, 225, 225, 225))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel.width, panel.height//3), fill=(30, 110, 220, 255))
    draw.text((12, 8), "SITE INFO", font=font, fill="white")
    draw.text((12, panel.height//2), "2026-10-31", font=font, fill="black")
    layer = Image.new("RGBA", clean.size, (0, 0, 0, 0))
    layer.paste(panel, (int(w*location[0]), int(h*location[1])))
    before = Image.alpha_composite(clean.convert("RGBA"), layer).convert("RGB")
    cases.append(("known_clean_removal_"+label, before, clean.copy(), "passed"))
    if label == "bottom_left":
        combined_before = before
if args.combined:
    for label in ("faint_text_residue", "localized_person_blur",
                  "small_edge_displacement", "small_head_region_replaced"):
        after = next(c[2] for c in cases if c[0] == label)
        cases.append(("removed_plus_"+label, combined_before, after, "review"))
if args.compare_details:
    from watermark_studio import quality
    original_vqa = quality._structured_vqa
    active_pair = None
    def with_details(chats, fields, *arguments):
        if fields == ("content_preserved", "uncertain"):
            sheets, _ = quality.detail_sheets(*active_pair, 4)
            content = chats[0][0]["content"]
            for item, sheet in zip(content[:2], sheets):
                overview = item["image"]
                canvas = Image.new("RGB", (max(overview.width, sheet.width),
                                           overview.height+sheet.height+24), (48, 48, 48))
                canvas.paste(overview, ((canvas.width-overview.width)//2, 0))
                canvas.paste(sheet, ((canvas.width-sheet.width)//2, overview.height+24))
                item["image"] = canvas
            content[-1]["text"] += ("\n两图均为上方总览、下方同位置放大细节。"
                                    "只对比原来可见的真实内容；不要把原图数字水印被背景替代当成内容破坏。"
                                    "灰色分隔和白边只是排版。")
        return original_vqa(chats, fields, *arguments)
    quality._structured_vqa = with_details
out = root/("subtle-controls-"+args.backend+("-combined-details" if args.compare_details else "-combined") if args.combined else "subtle-controls-"+args.backend+"-v2")
out.mkdir(exist_ok=True)
for detail in ((4,) if args.combined else (0, 4)):
    records = []
    for label, before, after, expected in cases:
        active_pair = (before, after)
        result = verify_batch([(before, after)], encoder, processor, "cuda",
                              resolution=768, max_tokens=128, reviewer=args.backend,
                              detail_patches=detail)[0]
        row = {"case": label, "expected": expected, "details": detail, **result}
        records.append(row)
        with (out/"results.jsonl").open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False)+"\n")
    print(json.dumps({"backend": args.backend, "details": detail, "cases": len(records),
                      "bad_false_pass": [r["case"] for r in records if r["expected"]=="review" and r["status"]=="passed"],
                      "clean_status": [r["status"] for r in records if r["expected"]=="passed"],
                      "errors": sum(r["status"]=="error" for r in records),
                      "seconds": round(sum(r["seconds"] for r in records), 3)}), flush=True)
