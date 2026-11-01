"""Small diagnostic contact sheet; inputs and frozen outputs remain read-only."""
import argparse
import json
import runpy
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from watermark_studio.boundary_views import boundary_ribbons
from watermark_studio.config import Region
from watermark_studio.images import context_box, load_photo, pixel_box

p=argparse.ArgumentParser()
p.add_argument("--job",required=True)
p.add_argument("--files",nargs="+",required=True)
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
if not 1<=len(a.files)<=6 or a.output.exists():
    p.error("One to six cases and a fresh preview file required")
helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
cases=helper["frozen_job_cases"](Path.home()/".local/state/watermark-studio",a.job,a.files)
hashes=helper["file_hashes"](cases)
canvas=Image.new("RGB",(1440,320*len(cases)),"white")
draw=ImageDraw.Draw(canvas)
for index,c in enumerate(cases):
    source=load_photo(c["source"])[0]
    result=load_photo(c["output"])[0]
    regions=[Region.model_validate(r) for r in c["regions"]]
    if not regions:
        box=(0,0,source.width,source.height)
    else:
        region=max(regions,key=lambda r:(r.box[2]-r.box[0])*(r.box[3]-r.box[1]))
        box=context_box(pixel_box(region.box,source.size),source.size,.22)
    ribbon,_=boundary_ribbons(result,regions)
    views=[source.crop(box),result.crop(box),ribbon or result]
    for column,image in enumerate(views):
        thumb=ImageOps.contain(image,(478,292))
        canvas.paste(thumb,(column*480+(480-thumb.width)//2,index*320+25))
    draw.text((4,index*320+4),c["file"]+" SOURCE / OUTPUT / OUTPUT BOUNDARIES",fill="black")
a.output.parent.mkdir(parents=True,exist_ok=True)
for quality in (70,60,50,40,30):
    canvas.save(a.output,quality=quality)
    if a.output.stat().st_size<=98304:
        break
if a.output.stat().st_size>98304:
    a.output.unlink()
    raise RuntimeError("Preview exceeds the small transfer budget")
assert hashes==helper["file_hashes"](cases),"Frozen image changed during preview"
print(json.dumps({"cases":len(cases),"bytes":a.output.stat().st_size,"sources_and_outputs_unchanged":True}))
