"""Compare compact reviewer prompts on the image pipeline's resident encoder."""
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from PIL import Image, ImageDraw, ImageOps
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from watermark_studio import quality
from watermark_studio.detection import locate_watermarks

snapshot = Path.home()/".cache/watermark-models/models/Qwen--Qwen-Image-2.1-Turbo/snapshots/master"
encoder = Qwen3VLForConditionalGeneration.from_pretrained(
    snapshot/"text_encoder", dtype=torch.bfloat16, local_files_only=True,
).to("cuda").eval()
processor = Qwen3VLProcessor.from_pretrained(snapshot/"processor", local_files_only=True)
root = Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031/quality-benchmark")
files = sorted((root/"input").glob("*"))
cases = []
for index in (0, 1, 3, 6, 12, 19):
    with Image.open(files[index]) as im:
        before = ImageOps.exif_transpose(im).convert("RGB")
    cases.append((f"watermark_retained_{index}", before, before.copy()))
for index in (3, 12):
    with Image.open(files[index]) as im:
        before = ImageOps.exif_transpose(im).convert("RGB")
    after = before.copy()
    ImageDraw.Draw(after).rectangle((0, 0, before.width//2, before.height//2), fill="lime")
    cases.append((f"content_destroyed_{index}", before, after))
prompts = {
    "short_cn": """图1是原图，图2是去水印结果。严格验收图2：
1. 所有数字水印、蓝灰信息卡、黄底文字、今日水印和小字是否完全消失？
2. 修补是否自然，有无纯色填块、模糊、接缝、人体畸形？
3. 原来可见的人物、机械、标牌、道路、植物是否保持不变？
水印还在则watermark_removed=false，有涂块则natural=false，删改主体则content_preserved=false。
返回JSON，字段watermark_removed、natural、content_preserved、uncertain为布尔值，reason为具体问题和位置。不确定就明确标记，不要默认通过。""",
    "evidence_cn": """图1为原图，图2为处理结果。请认真逐项观察图2，并与图1比较。
先在reason中写出看到的残留水印文字或异常位置、修补缺陷、主体变化，再给出判断。
返回JSON：
watermark_removed（是否完全无数字水印），natural（修补是否自然），content_preserved（原来可见内容是否未变），uncertain（是否难以判断）均为布尔值。
有任何水印卡片、相机品牌、小字、底纹即未去净；出现大块纯色遮盖、少人少物、场景替换不能通过。只有实际证据充分才通过。""",
}
for name, prompt in prompts.items():
    quality.PROMPT = prompt
    rows = []
    for label, before, after in cases:
        result = quality.review_batch([(before, after)], encoder, processor, "cuda", 768, 192)[0]
        # Report the VLM-only verdict separately from deterministic guards.
        raw_pass = all(result.get(k, False) for k in ("watermark_removed", "natural", "content_preserved"))
        row = {"prompt": name, "case": label, "vlm_pass": raw_pass, **result}
        rows.append(row)
        with (root/"resident-prompts.jsonl").open("a") as out:
            out.write(json.dumps(row, ensure_ascii=False)+"\n")
    print(json.dumps({"prompt": name, "bad_cases": len(rows),
                      "vlm_false_pass": sum(r["vlm_pass"] for r in rows),
                      "guarded_false_pass": sum(r["status"]=="passed" for r in rows)}), flush=True)
for label, _, after in cases:
    try:
        regions, raw = locate_watermarks(after, encoder, processor, "cuda", 768)
        row = {"case": label, "detected": len(regions), "raw": raw}
    except (ValueError, TypeError, RuntimeError) as exc:
        row = {"case": label, "error": type(exc).__name__}
    with (root/"resident-detection-controls.jsonl").open("a") as out:
        out.write(json.dumps(row, ensure_ascii=False)+"\n")
print("Resident comparisons complete", flush=True)
