"""Adversarial verifier controls with known restoration targets; never rewrite originals."""
import argparse
import collections
import hashlib
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from watermark_studio.cli import request

parser=argparse.ArgumentParser()
parser.add_argument("--url",default="http://127.0.0.1:8765")
parser.add_argument("--tag",required=True)
parser.add_argument("--prepare-only",action="store_true")
args=parser.parse_args()
if not args.tag.replace("-","").replace("_","").isalnum():
    parser.error("Simple tag required")
root=Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031")
work=root/("adversarial-"+args.tag)
inputs,outputs=work/"input",work/"output"
inputs.mkdir(parents=True,exist_ok=True);outputs.mkdir(parents=True,exist_ok=True)
fixture=work/"fixture.json"
if not fixture.exists():
    bases=[]
    hold=root/"first-pass-holdout-neutral-v1/input"
    for name in ("0010.jpg","0065.jpg","0027.jpg","0042.jpg","0051.jpg","0082.jpg"):
        with Image.open(hold/name) as raw:
            bases.append((name,ImageOps.exif_transpose(raw).convert("RGB")))
    old=root/"quality-benchmark-cross-review-v7-64/input"
    for label,name,box in (("sign","010.jpg",(.40,.35,.80,.85)),
                           ("brand","057.jpg",(.48,.14,.78,.40)),
                           ("receipt","054.jpg",(.08,.12,.95,.55))):
        with Image.open(old/name) as raw:
            image=ImageOps.exif_transpose(raw).convert("RGB")
        native=tuple(round(v*s) for v,s in zip(box,(image.width,image.height,image.width,image.height)))
        bases.append((label,image.crop(native)))
    records=[]
    contact=Image.new("RGB",(960,900),"white")
    draw=ImageDraw.Draw(contact)
    for index,(name,clean) in enumerate(bases):
        clean.thumbnail((1536,1536),Image.Resampling.LANCZOS)
        thumb=ImageOps.contain(clean,(320,270))
        contact.paste(thumb,((index%3)*320,(index//3)*300+20))
        draw.text(((index%3)*320+3,(index//3)*300+2),name,fill="black")
        w,h=clean.size
        font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",max(14,w//45))
        small=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",max(12,w//90))
        x,y=((.02,.66),(.60,.03),(.30,.38),(.03,.03),(.60,.67))[index%5]
        box=(int(x*w),int(y*h),int((x+.37)*w),int(min(1,y+.25)*h))
        def overlay(style):
            layer=Image.new("RGBA",clean.size,(0,0,0,0))
            d=ImageDraw.Draw(layer)
            if style in {"panel","background"}:
                d.rectangle(box,fill=(220,220,220,220))
                d.rectangle((box[0],box[1],box[2],box[1]+max(20,int(h*.07))),fill=(30,100,220,235))
            if style in {"panel","text"}:
                d.text((box[0]+5,box[1]+5),"SITE CAMERA",font=font,fill=(255,255,255,255))
                d.text((box[0]+5,box[1]+int(h*.1)),"2026 / PROJECT",font=small,fill=(0,0,0,240))
            if style=="code":
                d.text((box[0],box[3]-max(16,h//55)),"VERIFY 8F2P9X",font=small,fill=(255,255,255,220))
            return Image.alpha_composite(clean.convert("RGBA"),layer).convert("RGB")
        marked=overlay("panel")
        variants=[("identity",clean,clean,"passed"),
                  ("exact_removal",marked,clean,"passed"),
                  ("exact_jpeg",marked,clean,"passed"),
                  ("retained_panel",marked,marked,"review"),
                  ("residual_text",marked,overlay("text"),"review"),
                  ("residual_background",marked,overlay("background"),"review"),
                  ("small_code",marked,overlay("code"),"review")]
        for opacity in (18,32):
            layer=Image.new("RGBA",clean.size,(0,0,0,0))
            ImageDraw.Draw(layer).rectangle(box,fill=(0,0,0,opacity))
            dark=Image.alpha_composite(clean.convert("RGBA"),layer).convert("RGB")
            variants.append(("shadow_"+str(opacity),marked,dark,"review"))
        blur=clean.copy();blur.paste(clean.crop(box).filter(ImageFilter.GaussianBlur(7)),box)
        variants.append(("repair_blur",marked,blur,"review"))
        fill=clean.copy();ImageDraw.Draw(fill).rectangle(box,fill=(128,128,128))
        variants.append(("flat_repair",marked,fill,"review"))
        shifted=clean.copy()
        outside=(int(w*.05),int(h*.35),int(w*.25),int(h*.55))
        patch=clean.crop(outside)
        shifted.paste(patch,(outside[0]+max(5,w//180),outside[1]))
        variants.append(("visible_displacement",marked,shifted,"review"))
        for kind,before,after,expected in variants:
            if kind in {"repair_blur","visible_displacement"}:
                import numpy as np
                target=box if kind=="repair_blur" else (
                    outside[0],outside[1],outside[2]+max(5,w//180),outside[3])
                change=np.abs(np.asarray(clean.crop(target),dtype=float)-
                              np.asarray(after.crop(target),dtype=float)).mean()
                if change<2: continue  # No meaningful injected defect in a uniform patch.
            filename=f"{index:02d}-{kind}.png"
            before.save(inputs/filename)
            if kind=="exact_jpeg":
                # Preserve a PNG container while carrying actual JPEG quantization.
                import io
                encoded=io.BytesIO();after.save(encoded,format="JPEG",quality=97,subsampling=0)
                after=Image.open(io.BytesIO(encoded.getvalue())).convert("RGB")
            after.save(outputs/filename)
            records.append({"file":filename,"background":name,"kind":kind,"expected":expected})
    for quality in (65,55,45,35):
        contact.save(work/"background-contact.jpg",quality=quality)
        if (work/"background-contact.jpg").stat().st_size<=98304: break
    assert (work/"background-contact.jpg").stat().st_size<=98304
    fixture.write_text(json.dumps(records,indent=2))
if args.prepare_only:
    print(json.dumps({"fixtures":len(json.loads(fixture.read_text())),"contact":str(work/"background-contact.jpg")}))
    raise SystemExit
if (work/"result.json").exists():
    raise RuntimeError("Fresh tag required for measured controls")
records=json.loads(fixture.read_text())
health=request(args.url,"/health")
if health["stage"]!="ready" or (health.get("job") or {}).get("state") in {"running","scanning","paused"}:
    raise RuntimeError("Ready idle service required")
def hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs.iterdir()}
original_hashes=hashes()
started=time.monotonic()
job=request(args.url,"/jobs",{"input_dir":str(inputs),"output_dir":str(outputs),"options":{
    "concurrency":20,"quality_check":True,"quality_batch_size":2,"quality_detail_patches":4,
    "preserve_background":True,"quality_cross_review":True}})["id"]
last=started
while True:
    state=request(args.url,"/health");status=state["job"]
    assert status["id"]==job
    if status["state"] in {"completed","failed","cancelled"}: break
    if time.monotonic()-last>=30:
        print(json.dumps({"controls":status["completed"],"total":status["total"]}),flush=True)
        last=time.monotonic()
    if time.monotonic()-started>1800: raise TimeoutError(job)
    time.sleep(.5)
assert hashes()==original_hashes,"Read-only review modified controlled outputs"
rows={r["file"]:r for r in map(json.loads,(Path.home()/".local/state/watermark-studio"/job/"results.jsonl").read_text().splitlines())}
for r in records:
    r["actual"]=rows[r["file"]].get("quality",{}).get("status","error")
    r["reason"]=rows[r["file"]].get("quality",{}).get("reason")
    r["guards"]=rows[r["file"]].get("quality",{}).get("guard_codes",[])
summary={"images":len(records),"job":job,"code_revision":state["code_revision"],
    "seconds":round(time.monotonic()-started,3),"output_unchanged":True,
    "clean_controls":sum(r["expected"]=="passed" for r in records),
    "false_rejects":sum(r["expected"]=="passed" and r["actual"]!="passed" for r in records),
    "defect_controls":sum(r["expected"]=="review" for r in records),
    "false_accepts":sum(r["expected"]=="review" and r["actual"]=="passed" for r in records),
    "errors":sum(r["actual"]=="error" for r in records),
    "confusion_by_kind":{kind:dict(collections.Counter(r["actual"] for r in records if r["kind"]==kind))
                         for kind in sorted({r["kind"] for r in records})}}
(work/"result.json").write_text(json.dumps(summary,indent=2))
(work/"cases.json").write_text(json.dumps(records,ensure_ascii=False,indent=2))
print(json.dumps(summary),flush=True)
