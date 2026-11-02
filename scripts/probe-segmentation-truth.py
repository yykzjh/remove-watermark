"""Test source masks against exact known overlays; never modify image outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import subprocess
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np


def digest(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--fixture",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--sha256",required=True)
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--jitter",type=float,default=0.)
    p.add_argument("--tag",required=True)
    a=p.parse_args()
    if a.gpu<0 or not 0<=a.jitter<=.025 or not a.tag.replace("-","").replace("_","").isalnum():
        p.error("Explicit GPU, bounded jitter and new simple tag required")
    helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as response:
        isolation=helper["require_disjoint_server_gpus"](json.load(response),[a.gpu],True)
    data=json.loads(a.fixture.read_text());root=a.fixture.resolve().parent
    if data.get("version")!=1 or data.get("quality_permission") is not False:
        p.error("Diagnostic exact-source fixture required")
    hashes=data["known_clean_source_sha256"].copy();unique={}
    for row in data["cases"]:
        path=Path(row["source"]).resolve(strict=True)
        if not path.is_relative_to(root) or digest(path)!=row["source_file_sha256"]:
            raise ValueError("Fixture source changed or escaped")
        unique.setdefault(str(path),row);hashes[str(path)]=row["source_file_sha256"]
    if len(unique)!=data["images"]:
        raise ValueError("Unique source fixture coverage differs")
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
    import cv2
    import torch

    from watermark_studio.images import load_photo
    from watermark_studio.segmentation_probe import load_segmenter, mask_truth, segment_whole_photo
    torch.set_num_threads(1)
    clean={Path(path).name.split("-")[0]:load_photo(path)[0] for path in data["known_clean_source_sha256"]}
    work=a.fixture.parent.parent/a.tag;work.mkdir(exist_ok=False)
    revision=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    began=time.monotonic();model,processor,identity=load_segmenter(a.checkpoint,a.sha256,"cuda:0")
    load_seconds=time.monotonic()-began;began=time.monotonic();rows=[];full=Counter()
    with (work/"results.jsonl").open("x") as stream:
        for path,case in unique.items():
            photo=load_photo(path)[0];original=clean[Path(path).name.split("-")[0]]
            if photo.size!=original.size:raise ValueError("Native clean counterpart dimensions differ")
            truth=np.any(np.asarray(photo)!=np.asarray(original),axis=2)
            started=time.monotonic()
            masks,audit=segment_whole_photo(photo,case["proposal"]["core"],model,processor,"cuda:0",a.jitter)
            metrics={}
            radius=max(1,round(max(photo.size)*.004))
            for label,mask in masks.items():
                dilated=cv2.dilate(mask.astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8)).astype(bool)
                for key,value in ((label,mask),(label+"_dilated",dilated)):
                    metrics[key]=mask_truth(value,truth);full[key]+=metrics[key]["full_coverage"]
            row={"file":Path(path).name,"audit":audit,"metrics":metrics,"dilation_pixels":radius,
                "seconds":round(time.monotonic()-started,4),"approved":False}
            rows.append(row);stream.write(json.dumps(row,ensure_ascii=False)+"\n");stream.flush()
            print(json.dumps({"completed":len(rows),"total":len(unique)}),flush=True)
    if (any(digest(path)!=sha for path,sha in hashes.items()) or
            revision!=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()):
        raise ValueError("Source or code changed")
    with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as response:
        if isolation!=helper["require_disjoint_server_gpus"](json.load(response),[a.gpu],True):
            raise ValueError("Server reservation changed")
    report={"scope":"Mask-only exact-pixel coverage test; no image generation, approvals or output edits",
        "revision":revision,"images":len(rows),"full_coverage":dict(full),"jitter":a.jitter,
        "seconds":round(time.monotonic()-began,3),"load_seconds":round(load_seconds,3),
        "segmenter_identity":identity,"sources_unchanged":True}
    (work/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!="segmenter_identity"}))


if __name__=="__main__":
    main()
