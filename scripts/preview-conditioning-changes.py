"""Bounded two-source native diagnostic montage; never changes outputs or QA."""
import argparse
import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw

from watermark_studio.images import load_photo

p=argparse.ArgumentParser()
p.add_argument("--work",type=Path,required=True)
p.add_argument("--source-job",required=True)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--candidate",default="split")
a=p.parse_args();assert not a.output.exists()
assert Path(a.candidate).name==a.candidate
root=Path.home()/".local/state/watermark-studio"/a.source_job
source=Path(json.loads((root/"summary.json").read_text())["input_dir"])
raw={r["file"]:r for r in map(json.loads,(a.work/"raw/results.valid.jsonl").read_text().splitlines())}
candidate=a.work/a.candidate
records=candidate/"results.valid.jsonl"
if not records.exists():records=candidate/"results.jsonl"
split={r["file"]:r for r in map(json.loads,records.read_text().splitlines())}
raw={name:r for name,r in raw.items() if name in split}
chosen=[]
for up in (True,False):
    for name,r in raw.items():
        before=r["quality"];after=split[name]["quality"]
        if ((before["status"]!="passed" and after["status"]=="passed") if up else
                (before["status"]=="passed" and after["status"]!="passed")):
            q=before if up else after
            segments=[seg for k in ("repair_texture","repair_boundary","repair_rectangle_phase")
                      for seg in q.get(k,{}).get("suspect_segments",[]) if "box" in seg]
            if segments:chosen.append((name,segments[0]["box"],up));break
if len(chosen)==1:
    # A candidate with no regressions still gets a second independent recovery.
    for name,r in raw.items():
        if name==chosen[0][0] or r["quality"]["status"]=="passed" or split[name]["quality"]["status"]!="passed":
            continue
        q=r["quality"]
        segments=[seg for k in ("repair_texture","repair_boundary","repair_rectangle_phase")
                  for seg in q.get(k,{}).get("suspect_segments",[]) if "box" in seg]
        if segments:chosen.append((name,segments[0]["box"],True));break
assert len(chosen)==2
tile=512;height=tile+34
canvas=Image.new("RGB",(tile*3,height*2),"white");draw=ImageDraw.Draw(canvas)
audits=[]
for row,(name,box,up) in enumerate(chosen):
    paths=[source/name,a.work/"raw/output"/name,candidate/"output"/name]
    original=load_photo(str(paths[0]))[0];w,h=original.size
    center=((box[0]+box[2])*w/2,(box[1]+box[3])*h/2)
    cw,ch=min(tile,w),min(tile,h)
    x=max(0,min(w-cw,round(center[0]-cw/2)));y=max(0,min(h-ch,round(center[1]-ch/2)))
    crop=(x,y,x+cw,y+ch)
    for col,(path,label) in enumerate(zip(paths,("ORIGINAL","RAW ROI",a.candidate.upper()),strict=True)):
        photo=load_photo(str(path))[0];assert photo.size==original.size
        image=photo.crop(crop)
        canvas.paste(image,(col*tile,row*height+34))
        status="" if col==0 else raw[name]["quality"]["status"] if col==1 else split[name]["quality"]["status"]
        draw.text((col*tile+8,row*height+8),label+" "+status,fill="black")
    audits.append({"file":name,"native_crop":crop,"up":up,
        "source_sha256":hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        "raw":raw[name]["quality"]["status"],"split":split[name]["quality"]["status"]})
for quality in (88,84,80,76,72,68,64):
    buf=io.BytesIO();canvas.save(buf,format="JPEG",quality=quality,optimize=True)
    if len(buf.getvalue())<=280000:break
assert len(buf.getvalue())<=280000
a.output.write_bytes(buf.getvalue())
a.output.with_suffix(".json").write_text(json.dumps(audits,ensure_ascii=False,indent=2))
print(json.dumps({"bytes":len(buf.getvalue()),"source_cases":2,"native_pixels":True,"quality":quality}))
