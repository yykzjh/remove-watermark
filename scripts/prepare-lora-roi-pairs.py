"""Derive crop-relative repair supervision from screened native clean parents."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from watermark_studio.images import load_photo
from watermark_studio.roi_supervision import make_roi_supervision
from watermark_studio.training_features import digest, validate_pairs
from watermark_studio.training_pairs import render_profile_pair

p=argparse.ArgumentParser(__doc__)
p.add_argument("--parent-manifest",type=Path,required=True)
p.add_argument("--sealed",type=Path,required=True)
p.add_argument("--target",type=Path,required=True)
p.add_argument("--train-originals",type=int,default=48)
p.add_argument("--validation-originals",type=int,default=12)
p.add_argument("--main-resolution",type=int,choices=[512,768,1024],default=768)
p.add_argument("--font",type=Path,default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
p.add_argument("--seed",type=int,default=2026110319)
a=p.parse_args()
assert not a.target.exists() and 8<=a.train_originals<=96 and 4<=a.validation_originals<=24
parent,_=validate_pairs(a.parent_manifest)
assert digest(a.font)==parent["font_sha256"]
sealed=json.loads(a.sealed.read_text())
sealed_hashes=set(sealed["source_sha256"].values())
forbidden=set(json.loads((a.parent_manifest.parent/"forbidden-sha256.json").read_text()))|sealed_hashes
parents={}
for row in parent["pairs"]:
    if row["source_original_sha256"] not in forbidden:
        previous=parents.setdefault(row["source_original_sha256"],row)
        assert previous["split"]==row["split"]
selected=[]
for split,count in (("train",a.train_originals),("validation",a.validation_originals)):
    values=[r for r in parents.values() if r["split"]==split]
    values.sort(key=lambda r:hashlib.sha256((str(a.seed)+r["source_original_sha256"]).encode()).hexdigest())
    assert len(values)>=count
    selected+=values[:count]
a.target.mkdir(parents=True);work=a.target.resolve()
for folder in ("pairs","targets","masks"):(work/folder).mkdir()
(work/"forbidden-sha256.json").write_text(json.dumps(sorted(forbidden)))
pairs=[];source_hashes={}
for parent_row in selected:
    source=Path(parent_row["source_original"]);sha=parent_row["source_original_sha256"]
    assert digest(source)==sha
    original=load_photo(source)[0];native=original.crop(tuple(parent_row["crop"])).convert("RGB")
    prior=load_photo(parent_row["target"])[0]
    assert np.array_equal(np.asarray(native.resize(prior.size,Image.Resampling.LANCZOS)),np.asarray(prior))
    source_hashes[str(source)]=sha
    for variant in (0,1):
        seed=(a.seed+int(sha[:16],16)+variant)%(2**32)
        edited,mask,render=render_profile_pair(native,variant,random.Random(seed),a.font,profile="camera-v2")
        l,t,r,b=render["panel"]
        regions=[("panel",(l,t,r,b))]
        remaining=np.asarray(mask).copy();remaining[t:b,l:r]=0
        yy,xx=np.where(remaining>0)
        if len(xx):
            logo=(int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1)
            if max(l,logo[0])<min(r,logo[2]) and max(t,logo[1])<min(b,logo[3]):
                regions=[("panel",(min(l,logo[0]),min(t,logo[1]),max(r,logo[2]),max(b,logo[3])))]
            else:
                regions.append(("logo",logo))
        for group,(kind,box) in enumerate(regions):
            budget=512 if kind=="logo" else a.main_resolution
            image,truth,support,geometry=make_roi_supervision(edited,native,box,budget,kind)
            crop=geometry["crop"];expanded=geometry["expanded"];prompt=geometry["prompt"]
            name=sha[:24]+"-roi-"+str(variant*2+group)
            ip=work/"pairs"/(name+".png");tp=work/"targets"/(name+".png");mp=work/"masks"/(name+".png")
            image.save(ip);truth.save(tp);support.save(mp)
            pairs.append({"id":name,"split":parent_row["split"],"input":str(ip),"input_sha256":digest(ip),
                "target":str(tp),"target_sha256":digest(tp),"mask":str(mp),"mask_sha256":digest(mp),
                "source_original":str(source),"source_original_sha256":sha,"crop":parent_row["crop"],
                "roi_context_crop_native_parent":list(crop),"roi_group_native_parent":list(expanded),
                "seed":seed,"render":render|{"kind":"roi_"+kind},"training_crop_kind":"roi",
                "model_input_resolution":budget,"edit_prompt":prompt,"edit_prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest()})
    print(json.dumps({"parents":len(source_hashes),"needed":len(selected),"pairs":len(pairs)}),flush=True)
assert len(pairs)<=640 and all(digest(path)==sha for path,sha in source_hashes.items())
manifest={"version":1,"state":"completed","pairs":pairs,"prompt_profile":"roi_focused_v1",
    "parent_manifest":str(a.parent_manifest),"parent_manifest_sha256":digest(a.parent_manifest),
    "font_sha256":digest(a.font),"train_originals":a.train_originals,"validation_originals":a.validation_originals,
    "source_screening":parent.get("source_screening","inherited frozen dual-source/OCR-screened clean parents"),
    "sealed_source_sha256":hashlib.sha256(a.sealed.read_bytes()).hexdigest(),"sealed_source_overlap":0,
    "scope":"Source-disjoint ROI supervision; only the active padded rectangle is replaced with known clean pixels"}
path=work/"manifest.json";path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
validate_pairs(path)
(work/"source-selection.json").write_text(json.dumps({"source_sha256":source_hashes,"purpose":"ROI LoRA training parents; exclude from any future holdout"},ensure_ascii=False,indent=2))
print(json.dumps({"state":"completed","pairs":len(pairs),"manifest":str(path),"sealed_source_overlap":0}),flush=True)
