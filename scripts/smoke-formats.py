"""Verify GPU outputs preserve storage dimensions, EXIF orientation, alpha and formats."""
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from watermark_studio.cli import request

root = Path(".run/tuning/formats").resolve()
source, target = root/"input", root/"output"
source.mkdir(parents=True, exist_ok=True)
for index, suffix in enumerate(["jpg","jpeg","png","webp","bmp","tif","tiff","png"]):
    size = (803+index*2, 601+index*2)
    image = Image.new("RGB",size,(110,140,70))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20,400,320,550),fill=(220,220,220))
    draw.text((30,420),"WATERMARK CAMERA",fill="black")
    kwargs = {}
    if suffix in {"jpg","jpeg"}:
        exif = image.getexif()
        exif[274] = 6 if index == 0 else 8
        kwargs["exif"] = exif
    if index == 7:
        image.putalpha(180)
    image.save(source/f"sample-{index}.{suffix}",**kwargs)
url = "http://127.0.0.1:8765"
response = request(url,"/jobs",{
    "input_dir":str(source),"output_dir":str(target),
    "options":{"concurrency":16,"preset":"custom","regions":[{"box":[.1,.1,.4,.4]}],
               "resolution":768,"overwrite":True},
})
deadline = time.monotonic()+600
while time.monotonic()<deadline:
    job = request(url,"/health")["job"]
    assert job["id"] == response["id"]
    if job["state"] in {"completed","failed","cancelled"}:
        break
    time.sleep(.5)
assert job["state"] == "completed" and job["succeeded"] == 8, job
for path in source.iterdir():
    with Image.open(path) as before, Image.open(target/path.name) as after:
        assert before.size == after.size, path
        assert ImageOps.exif_transpose(before).size == ImageOps.exif_transpose(after).size, path
        assert before.getexif().get(274) == after.getexif().get(274), path
        if "A" in before.getbands():
            assert before.getchannel("A").tobytes() == after.getchannel("A").tobytes()
print(json.dumps({"formats_ok":True,"images":8,"job":job["id"]}),flush=True)
