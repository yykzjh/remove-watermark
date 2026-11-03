"""Source-only gray-card calibration diagnostic; no generation, repair, or QA approval."""
from __future__ import annotations

import argparse
import collections
import json
import runpy
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d


def edge_fit(photo,region,sigma):
    from watermark_studio.translucent_recovery import fit_transmission
    w,h=photo.size
    l,t,r,b=[round(x*s) for x,s in zip(region.box,(w,h,w,h))]
    if min(r-l,b-t)<100 or (r-l)*(b-t)>.4*w*h:
        return {"ok":False,"reason":"unsupported_extent"}
    rgb=np.asarray(photo.convert("RGB"),dtype=np.float32)
    inset=max(12,round((r-l)*.1))
    row=np.median(rgb[t:b,l+inset:r-inset],axis=1)
    neutral=(row.max(axis=1)-row.min(axis=1)<45)&(row.mean(axis=1)>130)
    starts=[i for i in range(len(row)-19) if neutral[i:i+20].mean()>=.9]
    if not starts:return {"ok":False,"reason":"no_neutral_body"}
    top=t+starts[0]+8;lo,hi=top+12,b-20
    if hi-lo<128:return {"ok":False,"reason":"short_body"}
    radius=max(20,round(.008*w)+6)
    edges=[]
    for side,expected,sign in [("left",l,1),("right",r,-1)]:
        if not 20<expected<w-20:continue
        first,last=max(20,expected-radius),min(w-20,expected+radius)
        # Smooth ONLY parallel to the edge: no card/background alpha leakage.
        strip=gaussian_filter1d(rgb[lo:hi,first-9:last+10],sigma,axis=0,mode="nearest")
        candidates=[]
        for at in range(first,last+1):
            x=strip[:,at-first+9-sign*6-1:at-first+9-sign*6+2].mean(axis=1)
            y=strip[:,at-first+9+sign*6-1:at-first+9+sign*6+2].mean(axis=1)
            candidates.append((float(np.median(np.linalg.norm(y-x,axis=1))),at,x,y))
        contrast,at,x,y=max(candidates,key=lambda v:v[0])
        fit=fit_transmission(x,y)
        halves=[fit_transmission(a,c) for a,c in zip(np.array_split(x,2),np.array_split(y,2))]
        consistent=(fit["ok"] and all(v["ok"] for v in halves) and
            abs(halves[0]["transmission"]-halves[1]["transmission"])<=.05 and
            np.max(np.abs(np.array(halves[0]["foreground"])-halves[1]["foreground"]))<=12)
        edges.append({"side":side,"position":at,"contrast":contrast,"fit":fit,
                      "halves":halves,"consistent":bool(consistent)})
    return {"ok":any(e["consistent"] for e in edges),"edges":edges,
            "body":[l,top,r,b],"reason":"source_parallel_calibration_only"}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--target",type=Path,required=True)
    a=p.parse_args()
    assert not a.target.exists()
    helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    cases=helper["manifest_cases"](a.manifest);hashes=helper["file_hashes"](cases)
    from watermark_studio.config import Region
    from watermark_studio.images import load_photo
    a.target.mkdir(parents=True);counts=collections.Counter();rows=[];started=time.monotonic()
    with (a.target/"results.jsonl").open("x") as log:
        for i,case in enumerate(cases):
            photo=load_photo(case["source"])[0]
            results=[]
            if case.get("scene")=="outdoor":
                for value in case.get("regions",[]):
                    region=Region.model_validate(value)
                    if region.kind!="panel":continue
                    for sigma in [2,4,8,12,16,24]:
                        fit=edge_fit(photo,region,sigma)
                        if fit["ok"]:counts[str(sigma)]+=1
                        results.append({"sigma":sigma,"box":region.box,**fit})
            record={"file":case["file"],"fits":results,"baseline_status":case.get("baseline_status")}
            rows.append(record);log.write(json.dumps(record)+"\n");log.flush()
            if (i+1)%16==0:print(json.dumps({"done":i+1,"fits":dict(counts)}),flush=True)
    assert helper["file_hashes"](cases)==hashes
    report={"state":"diagnostic_completed","images":len(rows),"seconds":time.monotonic()-started,
            "fit_regions_by_sigma":dict(counts),"source_bytes_unchanged":True,
            "additional_generations":0,"production_adopted":False,
            "scope":"Source-only physical calibration hypothesis; no repaired images or quality approvals"}
    (a.target/"report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
