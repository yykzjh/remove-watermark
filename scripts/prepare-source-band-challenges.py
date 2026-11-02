"""Construct held-out digital-matte band challenges with exact clean-pixel provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

from PIL import Image, ImageDraw

from watermark_studio.images import load_photo


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--controls",type=Path,required=True)
    p.add_argument("--target",type=Path,required=True)
    a=p.parse_args()
    h=runpy.run_path(str(Path(__file__).with_name("calibrate-source-side-proof.py")))
    labels=json.loads((a.controls/"fixture.json").read_text())
    clean=[r for r in labels if r["kind"]=="exact_removal"]
    if len(clean)!=9:
        p.error("Exactly nine independently established clean backgrounds required")
    a.target.mkdir(exist_ok=False);inputs=a.target/"input";inputs.mkdir()
    cases=[];source_hashes={};seed=2026110203
    for index,record in enumerate(clean):
        original=(a.controls/"output"/record["file"]).resolve()
        source_hashes[str(original)]=digest(original)
        photo=load_photo(original)[0].convert("RGB");w,hgt=photo.size
        for variant in range(12):
            # Exact source geometry varies; model prompts receive neither labels nor alpha.
            alpha=(.18,.35,.60,.85)[variant%4]
            extra=(0,16,40)[variant//4]
            cardw=round(w*(.30+.04*(variant%3)));cardh=round(hgt*.24)
            l=round(w*(.07+.12*(variant%4)));t=round(hgt*(.07+.21*((variant//2)%3)))
            r=l+cardw;b=t+cardh;header=t+round(cardh*.24)
            gray_right=min(w-48,r+extra)
            card=Image.new("RGBA",photo.size,(0,0,0,0));draw=ImageDraw.Draw(card)
            fg=((220,220,220),(242,242,242),(200,210,220))[variant//4]
            draw.rectangle((l,header,gray_right-1,b-1),fill=(*fg,round(alpha*255)))
            blue=((30,110,220),(35,80,150),(30,150,210))[variant%3]
            draw.rectangle((l,t,r-1,header-1),fill=(*blue,245))
            draw.text((l+10,t+5),"SITE CAMERA",fill="white")
            draw.text((l+12,header+8),"2026 / PROJECT",fill=(15,15,15,255))
            image=Image.alpha_composite(photo.convert("RGBA"),card).convert("RGB")
            name=f"{index:02d}-matte-{variant:02d}.png";path=inputs/name;image.save(path)
            core=[l/w,t/hgt,r/w,b/hgt]
            mask=[max(0,core[0]-.025),max(0,core[1]-.025),
                  min(1,core[2]+.025),min(1,core[3]+.025)]
            # Native flat bands exclude lettering, but include real translucent texture.
            native={
                "clear_photo":(gray_right+8,header+16,min(w,gray_right+32),b-12),
                "gray_backing":(l+12,header+round(cardh*.25),r-12,header+round(cardh*.25)+20),
                "blue_backing":(r-round(cardw*.22),t+round((header-t)*.55),r-8,header-5),
            }
            if extra>=16:
                native["wider_gray_backing"]=(r+8,header+16,min(gray_right,r+32),b-12)
            sha=digest(path)
            for cohort,box in native.items():
                if min(box[2]-box[0],box[3]-box[1])<8:
                    continue
                proposal=h["labelled_proposal"](image,mask,core,box)
                cases.append({"cohort":cohort,"file":name+"/"+cohort,"source":str(path.resolve()),
                    "proposal":proposal,"expected_clear":cohort=="clear_photo","expected_digital":True,
                    "source_file_sha256":sha})
    if any(digest(path)!=sha for path,sha in source_hashes.items()):
        raise ValueError("Clean source changed")
    manifest={"version":1,"seed":seed,"images":108,"cases":cases,
        "known_clean_source_sha256":source_hashes,"quality_permission":False,
        "scope":"Offline source-band challenge only; never training or production data"}
    path=a.target/"fixture.json";path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps({"images":108,"band_cases":len(cases),"manifest":str(path),
        "manifest_sha256":digest(path)}))


if __name__=="__main__":
    main()
