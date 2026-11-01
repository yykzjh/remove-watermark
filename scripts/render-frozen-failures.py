"""Render a few guarded first candidates for native visual inspection; never write originals."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from watermark_studio.frozen_candidates import digest, record_digest
from watermark_studio.images import load_photo

p=argparse.ArgumentParser()
p.add_argument("--snapshot",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--limit",type=int,choices=[1,2,3,4],default=2)
a=p.parse_args()
if a.output.exists():p.error("Fresh preview required")
summary=json.loads((a.snapshot/"summary.json").read_text())
rows=[json.loads(s) for s in (a.snapshot/"results.jsonl").read_text().splitlines()]
manifest={r["file"]:r for r in json.loads((a.snapshot/"manifest.json").read_text())}
chosen=[]
for guard in ["repair_texture_collapse","repair_rectangle_phase","repair_boundary_step"]:
    candidates=[r for r in rows if guard in r["quality"].get("guard_codes",[])
                and r["quality"].get("model_checks",{}).get("natural") is True
                and r["file"] not in [q[0]["file"] for q in chosen]]
    if candidates:
        candidates.sort(key=lambda r:r["quality"].get("repair_texture",{}).get("count",0),reverse=True)
        chosen.append((candidates[0],guard))
    if len(chosen)==a.limit:break
if not chosen:p.error("No guarded candidates")
canvas=Image.new("RGB",(1440,24+500*len(chosen)),"#202020")
draw=ImageDraw.Draw(canvas);draw.text((8,4),"SOURCE | FIRST OUTPUT | SOURCE native | FIRST OUTPUT native",fill="white")
audit=[]
for index,(row,guard) in enumerate(chosen):
    name=row["file"];source=Path(summary["input_dir"])/name;candidate=Path(summary["output_dir"])/name
    expected=manifest[name]
    assert digest(source)==expected["source_sha256"] and digest(candidate)==expected["candidate_sha256"]
    assert record_digest(row)==expected["record_sha256"]
    before=load_photo(str(source))[0];after=load_photo(str(candidate))[0]
    q=row["quality"];region=max(row["repair_regions"],key=lambda r:(r["box"][2]-r["box"][0])*(r["box"][3]-r["box"][1]))["box"]
    if guard=="repair_texture_collapse":
        box=q["repair_texture"]["suspect_segments"][0]["box"]
        cx,cy=(box[0]+box[2])*.5*before.width,(box[1]+box[3])*.5*before.height
    elif guard=="repair_rectangle_phase":
        sides=q["repair_rectangle_phase"]["findings"][0]["side_evidence"]
        side="top" if "top" in sides else next(iter(sides))
        cx,cy=(region[0]+region[2])*.5*before.width,(region[1]+region[3])*.5*before.height
        if side in ["top","bottom"]:cy=sides[side]["position"]*before.height
        else:cx=sides[side]["position"]*before.width
    else:
        cx,cy=(region[0]+region[2])*.5*before.width,region[1]*before.height
    edge=min(480,before.width,before.height)
    x=max(0,min(before.width-edge,round(cx-edge/2)));y=max(0,min(before.height-edge,round(cy-edge/2)))
    crop=(x,y,x+edge,y+edge);top=24+index*500
    draw.text((8,top),f"case {index+1} {guard} source={before.size}",fill="white")
    for j,photo in enumerate([before,after]):
        thumb=photo.copy();ImageDraw.Draw(thumb).rectangle(crop,outline="red",width=max(2,round(before.width/500)))
        thumb=ImageOps.contain(thumb,(240,470));canvas.paste(thumb,(j*240,top+20))
        canvas.paste(photo.crop(crop),(480+j*480,top+20))
    assert digest(source)==expected["source_sha256"] and digest(candidate)==expected["candidate_sha256"]
    audit.append({"file":name,"guard":guard,"crop":crop,"source_size":before.size})
canvas.save(a.output,quality=78,optimize=True)
a.output.with_suffix(".json").write_text(json.dumps({"snapshot":str(a.snapshot),"cases":audit},ensure_ascii=False,indent=2))
print(json.dumps({"preview_bytes":a.output.stat().st_size,"cases":[{k:v for k,v in r.items() if k!="file"} for r in audit],"originals_unchanged":True}))
