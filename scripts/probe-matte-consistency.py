"""Offline physical-matte calibration; no image generation and no approval decisions."""
from __future__ import annotations

import argparse
import io
import json
import random
import runpy
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from watermark_studio.config import Region
from watermark_studio.images import load_photo
from watermark_studio.matte_consistency import matte_consistency
from watermark_studio.training_features import digest, validate_pairs


def main():
    p=argparse.ArgumentParser(__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--pair-manifest",type=Path)
    group.add_argument("--dataset-manifest",type=Path)
    p.add_argument("--target",type=Path,required=True)
    p.add_argument("--font",type=Path,default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    a=p.parse_args();a.target.mkdir(exist_ok=False);began=time.monotonic();records=[]
    totals=defaultdict(Counter);manifest=a.pair_manifest or a.dataset_manifest;manifest_sha=digest(manifest)
    def measure(name,before,after,regions,cohort,expected=None):
        started=time.monotonic();audit=matte_consistency(before,after,regions)
        flagged=audit["count"]>0
        row={"file":name,"cohort":cohort,"expected":expected,"added_review":flagged,"audit":audit,
            "seconds":round(time.monotonic()-started,4),"quality_permission":False}
        records.append(row);totals[cohort]["cases"]+=1;totals[cohort]["flagged"]+=int(flagged)
        totals[cohort]["observable"]+=int(audit["observable_tiles"]>0)
        print(json.dumps({"completed":len(records),"last_flagged":flagged}),flush=True)
    if a.pair_manifest:
        helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
        cases=helper["manifest_cases"](a.pair_manifest)
        for case in cases:
            before=load_photo(case["source"])[0];after=load_photo(case["output"])[0]
            measure(case["file"],before,after,[Region.model_validate(r) for r in case["regions"]],
                case.get("cohort","unknown"),case.get("expected"))
        helper["manifest_cases"](a.pair_manifest)
    else:
        from watermark_studio.artifact_pairs import physical_shadow_background
        from watermark_studio.training_pairs import render_pair
        data,_=validate_pairs(a.dataset_manifest)
        if data.get("font_sha256")!=digest(a.font):raise ValueError("Exact original font required")
        for row in data["pairs"]:
            if row["split"]!="validation":continue
            panel=row["render"].get("render",{}).get("panel")
            if panel is None:continue
            before=load_photo(row["input"])[0];clean=load_photo(row["target"])[0]
            width,height=clean.size;l,t,r,b=panel
            regions=[Region(kind="panel",box=(l/width,t/height,r/width,b/height))]
            variant=int(row["id"].rsplit("-",1)[-1])
            rendered,_,_,overlay=render_pair(clean,variant,random.Random(row["seed"]),a.font,return_overlay=True)
            if rendered.tobytes()!=before.tobytes():raise ValueError("Overlay replay failed")
            measure(row["id"]+"/clean",before,clean,regions,"known_clean","passed")
            stream=io.BytesIO();clean.save(stream,format="JPEG",quality=95);stream.seek(0)
            with Image.open(stream) as image:jpeg=image.convert("RGB")
            measure(row["id"]+"/jpeg95",before,jpeg,regions,"known_jpeg","passed")
            existing=physical_shadow_background(clean,panel,row["seed"]^0x1839)
            causal_source=Image.alpha_composite(existing.convert("RGBA"),overlay).convert("RGB")
            measure(row["id"]+"/physical_shadow",causal_source,existing,regions,"known_physical_shadow","passed")
            with Image.open(row["mask"]) as image:mask=np.asarray(image.convert("L"))>0
            for label,delta in [("dark18",-18),("dark32",-32),("bright18",18)]:
                pixels=np.asarray(clean,dtype=np.float32).copy();pixels[mask]+=delta
                damaged=Image.fromarray(np.clip(np.rint(pixels),0,255).astype(np.uint8))
                measure(row["id"]+"/"+label,before,damaged,regions,label,"review")
        validate_pairs(a.dataset_manifest)
    if digest(manifest)!=manifest_sha:raise ValueError("Manifest changed")
    report={"version":1,"state":"completed","manifest":str(manifest.resolve()),"manifest_sha256":manifest_sha,
        "cases":len(records),"cohorts":dict(totals),"seconds":round(time.monotonic()-began,3),
        "extra_generations":0,"quality_permission":False,
        "scope":"Exact causal synthetic controls or frozen native pairs; diagnostic can only add review"}
    (a.target/"results.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    (a.target/"report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=="__main__":main()
