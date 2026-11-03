"""Create retained digital-overlays among unchanged real printing; output must be rejected."""
import argparse
import importlib.util
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from watermark_studio.config import Region
from watermark_studio.images import load_photo

p=argparse.ArgumentParser(__doc__);p.add_argument("--native-manifest",type=Path,required=True)
p.add_argument("--output",type=Path,required=True);a=p.parse_args()
assert not a.output.exists()
spec=importlib.util.spec_from_file_location("verifier_benchmark",Path(__file__).with_name("benchmark-verifier-known-source.py"))
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
original=helper.manifest_cases(a.native_manifest)
tokens=["1号机耕道 (3)","补路 (2)","石山包村县级验收","10号生产路 (1)"]
selected=[next(c for c in original if token in c["file"]) for token in tokens]
(a.output/"input").mkdir(parents=True);(a.output/"output").mkdir()
cases=[]
for index,case in enumerate(selected):
    clean=load_photo(case["output"])[0].convert("RGB");w,h=clean.size
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",max(24,round(max(w,h)/100)))
    for variant,(alpha,location,style) in enumerate([(255,(.62,.92),"white"),(64,(.60,.04),"white"),
        (32,(.36,.12),"white"),(255,(.55,.78),"gray"),(255,(.02,.02),"blue"),
        (32,(.52,.45),"white")]):
        layer=Image.new("RGBA",clean.size,(0,0,0,0));draw=ImageDraw.Draw(layer)
        text="SITE CAMERA 2026";tw=int(draw.textlength(text,font=font));th=font.size+16
        x=min(w-tw-20,round(w*location[0]));y=min(h-th-10,round(h*location[1]))
        if style!="white":
            draw.rectangle((x-8,y-8,x+tw+8,y+th),fill=(220,220,220,180) if style=="gray" else (30,110,220,220))
        draw.text((x,y),text,font=font,fill=(0,0,0,alpha) if style=="gray" else (255,255,255,alpha))
        candidate=Image.alpha_composite(clean.convert("RGBA"),layer).convert("RGB")
        source=candidate.copy();sd=ImageDraw.Draw(source)
        mask=(.01,.70,.38,.98);box=tuple(round(v*d) for v,d in zip(mask,(w,h,w,h)))
        sd.rectangle(box,fill=(225,225,225));sd.rectangle((box[0],box[1],box[2],box[1]+(box[3]-box[1])//3),fill=(30,110,220))
        sd.text((box[0]+12,box[1]+12),"SITE INFO",fill="white",font=font)
        sd.text((box[0]+12,box[1]+(box[3]-box[1])//2),"DATE 2026",fill="black",font=font)
        name=f"{index:02d}-{variant:02d}-{style}-alpha{alpha}.png"
        src=a.output/"input"/name;out=a.output/"output"/name
        source.save(src,compress_level=1);candidate.save(out,compress_level=1)
        cases.append({"file":name,"source":str(src.resolve()),"output":str(out.resolve()),
            "regions":[Region(kind="panel",box=mask).model_dump()],"scene":"outdoor",
            "cohort":"inquiry_stress","expected":"review","preserved":True,
            "source_cores":[],"injected_remaining_overlay":{"style":style,"alpha":alpha,
                "box":[x/w,y/h,(x+tw)/w,(y+th)/h]},"clean_parent":case["output"]})
manifest={"version":1,"cases":cases,"file_sha256":helper.file_hashes(cases),
    "scope":"All 24 outputs contain known retained digital overlays; synthetic initial-dispute stress only"}
(a.output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print(json.dumps({"cases":len(cases),"manifest":str(a.output/"manifest.json")}),flush=True)
