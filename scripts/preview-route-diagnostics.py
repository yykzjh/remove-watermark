"""Render four diagnostic pairs; never modify originals or use QA flags as labels."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from watermark_studio.images import load_photo

p=argparse.ArgumentParser()
p.add_argument("--job",required=True)
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
assert not a.output.exists()
root=Path.home()/".local/state/watermark-studio"/a.job
s=json.loads((root/"summary.json").read_text())
rs=[json.loads(x) for x in (root/"results.jsonl").read_text().splitlines()]
chosen=[]
for tag in ["document","repair_rectangle_phase","repair_texture_collapse","source_physical_ambiguity"]:
    pool=[r for r in rs if r.get("quality",{}).get("status")=="review"
          and r["file"] not in [v[1]["file"] for v in chosen]
          and (r.get("source_scene")==tag or tag in r["quality"].get("judgment_basis",{}).get("local_guard_codes",[]))]
    if pool:chosen.append((tag,pool[0]))
canvas=Image.new("RGB",(1280,340*len(chosen)),"#202020");draw=ImageDraw.Draw(canvas)
audit=[]
for i,(tag,row) in enumerate(chosen):
    before=load_photo(str(Path(s["input_dir"])/row["file"]))[0]
    after=load_photo(str(Path(s["output_dir"])/row["file"]))[0]
    region=max(row.get("regions",[]) or [{"box":[0,.5,1,1]}],
               key=lambda r:(r["box"][2]-r["box"][0])*(r["box"][3]-r["box"][1]))["box"]
    crop=tuple(round(v*z) for v,z in zip(region,(before.width,before.height)*2))
    draw.text((4,i*340+4),f"{i}: {tag} | source | result | source ROI | result ROI",fill="white")
    for j,photo in enumerate([before,after,before.crop(crop),after.crop(crop)]):
        thumb=ImageOps.contain(photo,(310,310))
        canvas.paste(thumb,(j*320,i*340+24))
    audit.append({"index":i,"tag":tag,"file":row["file"],"region":region})
canvas.save(a.output,quality=70,optimize=True)
a.output.with_suffix(".json").write_text(json.dumps(audit,ensure_ascii=False,indent=2))
print(json.dumps({"bytes":a.output.stat().st_size,"cases":len(chosen),"output":str(a.output)}))
