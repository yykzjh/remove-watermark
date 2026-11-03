"""Known synthetic residuals over real readable printing; never training or production geometry."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def overlap(a,b):
    return max(a[0],b[0])<min(a[2],b[2]) and max(a[1],b[1])<min(a[3],b[3])


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--controls",type=Path,required=True)
    p.add_argument("--witnesses",type=Path,required=True)
    p.add_argument("--target",type=Path,required=True)
    a=p.parse_args()
    assert not a.target.exists()
    helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    cases=helper["manifest_cases"](a.controls);frozen=helper["file_hashes"](cases)
    report=json.loads((a.witnesses/"report.json").read_text())
    assert report["state"]=="completed_read_only_witnesses" and report["images"]==len(cases)
    witnesses={r["file"]:r for r in map(json.loads,(a.witnesses/"results.jsonl").read_text().splitlines())}
    font_path=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    assert font_path.is_file()
    chosen=[]
    for case in cases:
        if case["expected"]!="passed" or "exact_removal" not in case["file"]:continue
        words=[w["output"] for w in witnesses[case["file"]]["same_readable_words"] if w["pixels"]["unchanged"]]
        words=[w for w in words if w["confidence"]>=.9 and not w["app_marker"]]
        if words:
            word=max(words,key=lambda w:(w["box"][2]-w["box"][0])*(w["box"][3]-w["box"][1]))
            chosen.append((case,word))
    assert len(chosen)>=4
    from watermark_studio.images import load_photo, pixel_box
    a.target.mkdir(parents=True);generated=[]
    for parent_index,(case,word) in enumerate(chosen):
        clean=load_photo(case["output"])[0].convert("RGB");w,h=clean.size
        l,t,r,b=pixel_box(word["box"],clean.size)
        for mode in ("coincident","adjacent"):
            for alpha in (24,32,48,64,96,160):
                text=("N4X7","P8K2","R6M3","H9T5")[parent_index%4]
                size=max(8,round((b-t)*.65));font=ImageFont.truetype(str(font_path),size)
                while size>8 and font.getbbox(text)[2]-font.getbbox(text)[0]>r-l-4:
                    size-=1;font=ImageFont.truetype(str(font_path),size)
                glyph=font.getbbox(text);tw,th=glyph[2]-glyph[0],glyph[3]-glyph[1]
                assert tw<=r-l and th<=b-t
                x=l+max(0,(r-l-tw)//2);y=t+max(0,(b-t-th)//2)
                placement=mode
                if mode=="adjacent":
                    ys=[b+3,t-th-3]
                    valid=[at for at in ys if 0<=at and at+th<=h]
                    if valid:y=valid[0]
                    else:placement="coincident_edge_fallback"
                layer=Image.new("RGBA",clean.size);draw=ImageDraw.Draw(layer)
                backdrop=np.asarray(clean.crop((x,y,x+tw,y+th)),dtype=np.float32).mean()
                color=(255,255,255) if backdrop<160 else (0,0,0)
                draw.text((x-glyph[0],y-glyph[1]),text,font=font,fill=(*color,alpha))
                result=Image.alpha_composite(clean.convert("RGBA"),layer).convert("RGB")
                delta=np.abs(np.asarray(result,dtype=np.int16)-np.asarray(clean,dtype=np.int16)).max(axis=2)
                if int((delta>1).sum())<16 or int(delta.max())<4:
                    raise ValueError("Invisible synthetic control cannot receive a defect label")
                faint=[x/w,y/h,(x+tw)/w,(y+th)/h]
                word_box=word["box"]
                regions=[]
                for box in ([.02,.02,.34,.20],[.66,.02,.98,.20],[.02,.80,.34,.98],[.66,.80,.98,.98]):
                    if not overlap(box,faint) and not overlap(box,word_box):
                        regions=[{"kind":"panel","box":box}];break
                if not regions:raise ValueError("No independent original camera-card location")
                source=result.copy();draw=ImageDraw.Draw(source)
                cl,ct,cr,cb=pixel_box(regions[0]["box"],source.size)
                draw.rectangle((cl,ct,cr-1,cb-1),fill=(225,225,225))
                draw.rectangle((cl,ct,cr-1,ct+max(8,(cb-ct)//3)),fill=(65,125,215))
                tiny=ImageFont.truetype(str(font_path),max(8,min(16,(cb-ct)//5)))
                draw.text((cl+4,ct+max(10,(cb-ct)//3)+3),"SITE 2026",font=tiny,fill=(10,10,10))
                name=f"{parent_index:02d}-{mode}-alpha{alpha}.png"
                before=a.target/("source-"+name);after=a.target/("result-"+name)
                source.save(before);result.save(after)
                generated.append({"file":name,"source":str(before.resolve()),"output":str(after.resolve()),
                    "regions":regions,"scene":case["scene"],"expected":"review",
                    "expected_overlay_present":True,"cohort":"colocated-readable-scene-text",
                    "parent_control":case["file"],"physical_word_box":word_box,
                    "injected_remaining_overlay":{"box":faint,"alpha":alpha,"color":color,
                        "placement":placement,"kind":"unknown_floating_alphanumeric_overlay"},
                    "visibility":{"pixels_over1":int((delta>1).sum()),"maximum_delta":int(delta.max())}})
    assert helper["file_hashes"](cases)==frozen
    manifest={"version":1,"cases":generated,"file_sha256":helper["file_hashes"](generated),
              "parent_controls_manifest_sha256":hashlib.sha256(a.controls.read_bytes()).hexdigest(),
              "font_sha256":hashlib.sha256(font_path.read_bytes()).hexdigest(),
              "scope":"Negative QA controls only. Inserted locations never supplied as production geometry.",
              "training_used":False,"parents":len(chosen),"images":len(generated)}
    (a.target/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps({"parents":len(chosen),"images":len(generated),"training_used":False}),flush=True)


if __name__=="__main__":
    main()
