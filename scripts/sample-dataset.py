"""Stratified, bounded previews of a dataset already on the development machine."""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

p = argparse.ArgumentParser()
p.add_argument("--count", type=int, default=128)
args = p.parse_args()
base = Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031")
data = base / "八标照片集"
cloud = "/codex-download-1mQf6F6g-20261031/八标照片集/"
manifest = json.loads((data / ".download-state/manifest.json").read_text())
groups = defaultdict(list)
for row in manifest:
    rel = row["path"].removeprefix(cloud)
    f = data / rel
    if f.suffix.lower() not in {".jpg", ".png"}:
        continue
    if not f.is_file() or not row["size"] or f.stat().st_size != row["size"]:
        continue
    parts = Path(rel).parts
    match = re.match(r"(.+?(?:镇|乡|街道))(?:[-—_ ]+)?(.*)", parts[0])
    town = match[1] if match else parts[0]
    village = match[2] if match and match[2] else (parts[1] if len(parts)>2 else "未明确村名")
    named = re.match(r"(.+?(?:村|社区))", village)
    key = town + "/" + (named[1] if named else village)

    groups[key].append(f)
rng = random.Random(20261031)
keys = sorted(groups)
rng.shuffle(keys)
for group in groups.values():
    rng.shuffle(group)
selected = []
while keys and len(selected) < args.count:
    next_keys = []
    for key in keys:
        group = groups[key]
        while group:
            f = group.pop()
            try:
                with Image.open(f) as im:
                    im.load()
                    size = list(im.size)
            except (OSError, ValueError):
                continue
            selected.append({"path": str(f), "group": key, "size": size})
            break
        if group:
            next_keys.append(key)
        if len(selected) == args.count:
            break
    keys = next_keys
out = base / "sample-review"
out.mkdir(exist_ok=True)
(out / "sources.json").write_text(json.dumps(selected, ensure_ascii=False))
for page in range((len(selected)+7)//8):
    sheet = Image.new("RGB", (768, 336), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(selected[page*8:page*8+8]):
        with Image.open(row["path"]) as image:
            thumb = ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"), (192, 144))
        x, y = i%4*192, i//4*168
        sheet.paste(thumb, (x, y+20))
        draw.text((x+3, y+2), str(page*8+i), fill="black")
    target = out / f"contact-{page:02d}.jpg"
    sheet.save(target, quality=42)
    assert target.stat().st_size < 65536
print(json.dumps({"samples": len(selected), "directory_groups": len({r["group"] for r in selected}),
                  "town_directories": len({Path(r["group"]).parts[0] for r in selected}),
                  "pages": (len(selected)+7)//8}))
