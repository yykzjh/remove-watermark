"""Derive new digital overlays from immutable screened clean targets; preserve parent splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

from watermark_studio.images import load_photo
from watermark_studio.training_pairs import render_profile_pair


def digest(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--target",type=Path,required=True)
    p.add_argument("--font",type=Path,default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    p.add_argument("--render-profile",choices=["camera-v2"],required=True)
    a=p.parse_args();raw=a.manifest.read_bytes();parent=json.loads(raw)
    if parent.get("state")!="completed" or parent.get("source_overlap")!=0 or parent.get("previous_exposure_overlap")!=0:
        p.error("Completed source-disjoint screened parent required")
    if parent.get("font_sha256")!=digest(a.font):p.error("Exact screened font artifact required")
    rows=parent["pairs"];unique={};train=set();validation=set()
    for row in rows:
        if row["split"] not in ("train","validation"):raise ValueError("Unknown source split")
        sha=row["source_original_sha256"]
        (train if row["split"]=="train" else validation).add(sha)
        previous=unique.setdefault(sha,row)
        if any(previous[k]!=row[k] for k in ("split","target","target_sha256","crop","source_original")):
            raise ValueError("Conflicting clean target for one parent source")
    if train&validation:raise ValueError("Parent source split overlaps")
    if len(train)!=parent["train_originals"] or len(validation)!=parent["validation_originals"]:
        raise ValueError("Parent split count differs")
    a.target.mkdir(exist_ok=False);work=a.target.resolve()
    (work/"pairs").mkdir();(work/"masks").mkdir();pairs=[];began=time.monotonic()
    for sha,row in unique.items():
        if digest(row["target"])!=row["target_sha256"] or digest(row["source_original"])!=sha:
            raise ValueError("Screened clean or original bytes changed")
        target=load_photo(row["target"])[0]
        for variant in range(4):
            seed=int.from_bytes(hashlib.sha256((sha+"/camera-v2/"+str(variant)).encode()).digest()[:4],"big")
            image,mask,render=render_profile_pair(target,variant,random.Random(seed),a.font,profile=a.render_profile)
            name=sha[:24]+"-"+str(variant)+".png"
            ip=work/"pairs"/name;mp=work/"masks"/name;image.save(ip);mask.save(mp)
            pairs.append({**{k:row[k] for k in ("split","target","target_sha256","crop","source_original","source_original_sha256")},
                "id":name.removesuffix(".png"),"input":str(ip),"input_sha256":digest(ip),
                "mask":str(mp),"mask_sha256":digest(mp),"render":render,"seed":seed})
    if a.manifest.read_bytes()!=raw:raise ValueError("Parent manifest changed")
    manifest={**parent,"revision":subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip(),
        "pairs":pairs,"render_profile":a.render_profile,"parent_manifest":str(a.manifest.resolve()),
        "parent_manifest_sha256":hashlib.sha256(raw).hexdigest(),
        "new_original_sources":0,"split_policy":"Exact parent source splits; no native evaluation images added"}
    (work/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    report={"state":"completed","train_pairs":4*len(train),"validation_pairs":4*len(validation),
        "train_originals":len(train),"validation_originals":len(validation),"source_overlap":0,
        "new_original_sources":0,"parent_manifest_sha256":manifest["parent_manifest_sha256"],
        "render_profile":a.render_profile,"seconds":round(time.monotonic()-began,3)}
    (work/"report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=="__main__":main()
