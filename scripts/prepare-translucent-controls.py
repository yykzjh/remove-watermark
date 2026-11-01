"""Known-background synthetic development controls from exposed source-copy pairs.

Never a qualification holdout. Existing source files and frozen snapshots are
read-only; only the new work directory is written.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from watermark_studio.frozen_candidates import bounded, digest, record_digest
from watermark_studio.images import load_photo


def card_overlay(clean, position, transmission):
    if position not in range(4) or transmission not in (.45, .65):
        raise ValueError("Fixed control positions and opacity levels required")
    width, height = clean.size
    margin = max(4, round(min(clean.size)*.02))
    card_width, card_height = round(width*.38), round(height*.27)
    left = margin if position % 2 == 0 else width-margin-card_width
    top = margin if position < 2 else height-margin-card_height
    box = (left,top,left+card_width,top+card_height)
    original = clean.convert("RGB")
    result = original.copy()
    translucent = Image.blend(original.crop(box), Image.new("RGB",(card_width,card_height),"white"),1-transmission)
    result.paste(translucent, box)
    draw = ImageDraw.Draw(result)
    header_height = max(16, round(card_height*.25))
    draw.rectangle((left,top,left+card_width-1,top+header_height-1),fill=(45,120,225))
    font_size = max(12, round(height*.022))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf",font_size)
    except OSError:
        font = ImageFont.load_default(size=font_size)
    draw.text((left+8,top+max(1,(header_height-font_size)//2)), "SITE RECORD",font=font,fill="white")
    for index, text in enumerate(("PROJECT 08 - FIELD INSPECTION","DATE 2026-10-01 14:15","LOCATION: VILLAGE ROAD","NOTE: WORK IN PROGRESS")):
        draw.text((left+8,top+header_height+4+index*max(14,round(font_size*1.35))),text,font=font,fill="black")
    # A separate small camera footer tests two independently positioned overlays.
    footer = (width-round(width*.22)-margin,height-round(height*.055)-margin,width-margin,height-margin)
    draw.rectangle(footer,fill=(60,60,60))
    small = ImageFont.load_default(size=max(10,round(height*.015)))
    draw.text((footer[0]+3,footer[1]+2),"CAMERA | REAL TIME",font=small,fill="white")
    draw.text((footer[0]+3,footer[1]+max(12,round(height*.022))),"ID T24R9M6X",font=small,fill="white")
    return result, {"panel":list(box),"header_bottom":top+header_height,
                    "footer":list(footer),"transmission":transmission,"foreground":[255,255,255],
                    "known_background":"original source RGB before synthetic additions"}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--snapshot",type=Path,required=True)
    p.add_argument("--work",type=Path,required=True)
    p.add_argument("--sources",type=int,default=4,choices=[2,4,8])
    a = p.parse_args()
    if a.work.exists():
        p.error("Fresh development directory required")
    summary = json.loads((a.snapshot/"summary.json").read_text())
    if summary.get("snapshot_schema") != 2:
        p.error("Frozen first snapshot required")
    manifest = {r["file"]:r for r in json.loads((a.snapshot/"manifest.json").read_text())}
    rows = [json.loads(line) for line in (a.snapshot/"results.jsonl").read_text().splitlines()]
    selected = sorted([r for r in rows if r.get("processing_path") == "source_copy"
                       and r.get("quality",{}).get("status") == "passed"
                       and not r.get("quality",{}).get("source_regions")],
                      key=lambda r:manifest[r["file"]]["source_sha256"])
    chosen = []
    for row in selected:
        source = bounded(Path(summary["input_dir"]),row["file"])
        item = manifest[row["file"]]
        if digest(source) != item["source_sha256"] or record_digest(row) != item["record_sha256"]:
            raise ValueError("Frozen source-copy proof changed")
        photo = load_photo(source)[0]
        if photo.width < 800 or photo.height < 600 or photo.width < photo.height:
            continue
        if item["source_sha256"] in {c[2] for c in chosen}:
            continue
        chosen.append((row,photo,item["source_sha256"],source))
        if len(chosen) == a.sources:
            break
    if len(chosen) != a.sources:
        p.error("Insufficient distinct exposed source-copy photographs")
    (a.work/"input").mkdir(parents=True)
    (a.work/"ground-truth").mkdir()
    cases, hashes = [], {}
    for index, (row,photo,sha,source) in enumerate(chosen):
        clean_name = f"{index:02d}.png"
        photo.save(a.work/"ground-truth"/clean_name)
        for position in range(4):
            for transmission in (.45,.65):
                name = f"{index:02d}-p{position}-t{round(transmission*100)}.png"
                planted, geometry = card_overlay(photo,position,transmission)
                planted.save(a.work/"input"/name)
                hashes[name] = digest(a.work/"input"/name)
                cases.append({"file":name,"ground_truth":clean_name,"geometry":geometry,
                              "origin":row["file"],"origin_sha256":sha,
                              "ground_truth_sha256":digest(a.work/"ground-truth"/clean_name)})
        if digest(source) != sha:
            raise ValueError("Original changed during control preparation")
    selection = {"scope":"synthetic known-background development, NOT qualification or human labels",
                 "development_reuse":True,"images":len(hashes),"unique_sha256":len(set(hashes.values())),
                 "prior_sha256_overlap":0,"source_sha256":hashes,"source_snapshot":str(a.snapshot)}
    (a.work/"source-selection.json").write_text(json.dumps(selection,indent=2))
    (a.work/"fixture.json").write_text(json.dumps(cases,ensure_ascii=False,indent=2))
    (a.work/"sources.json").write_text(json.dumps([str(c[3]) for c in chosen],ensure_ascii=False))
    print(json.dumps({"cases":len(cases),"unique_origins":len(chosen),"scope":selection["scope"]}))


if __name__ == "__main__":
    main()
