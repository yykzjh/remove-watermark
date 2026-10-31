"""Small diagnostic previews of the two explicitly requested review cases."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

root = Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031/quality-benchmark-protected-v3-64")
for name in ("010.jpg", "057.jpg"):
    photos = []
    for folder in ("input", "quality-768-b2"):
        with Image.open(root/folder/name) as im:
            photos.append(ImageOps.exif_transpose(im).convert("RGB"))
    canvas = Image.new("RGB", (768, 840), "white")
    draw = ImageDraw.Draw(canvas)
    for column, image in enumerate(photos):
        for top, region, height, label in (
            (24, (0, 0, 1, 1), 288, "whole"),
            (340, (0, .55, .43, 1), 320, "panel"),
            (690, (.78, .90, 1, 1), 145, "camera corner"),
        ):
            w, h = image.size
            crop = image.crop(tuple(round(v*s) for v,s in zip(region,(w,h,w,h))))
            thumb = ImageOps.contain(crop, (384, height))
            canvas.paste(thumb, (column*384, top))
            draw.text((column*384+3,top-18), ("original " if column==0 else "output ")+label, fill="black")
    path = Path(".run")/("review-"+name)
    for quality in (65, 55, 45, 35, 25):
        canvas.save(path, quality=quality, optimize=True)
        if path.stat().st_size <= 65536:
            break
    assert path.stat().st_size <= 65536
    print(path.name, path.stat().st_size)
