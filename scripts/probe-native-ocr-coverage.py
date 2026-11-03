"""Measure OCR candidate coverage at known injected locations; never a watermark verdict."""
from __future__ import annotations

import argparse
import collections
import json
import runpy
import time
from pathlib import Path

import numpy as np


def location_covered(target, candidate):
    overlap=max(0,min(target[2],candidate[2])-max(target[0],candidate[0]))*max(
        0,min(target[3],candidate[3])-max(target[1],candidate[1]))
    target_area=(target[2]-target[0])*(target[3]-target[1])
    candidate_area=(candidate[2]-candidate[0])*(candidate[3]-candidate[1])
    return overlap/max(target_area,1e-12)>=.15 and overlap/max(candidate_area,1e-12)>=.3


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--target",type=Path,required=True)
    p.add_argument("--backend",choices=["rapid","ppocrv6","ppocrv5-server"],default="rapid")
    p.add_argument("--model-path",type=Path)
    p.add_argument("--model-sha256",default="")
    p.add_argument("--gpu",type=int,choices=[0,1,2,3])
    p.add_argument("--resolution",type=int,choices=[2048,3072,4096],default=4096)
    p.add_argument("--box-threshold",type=float,default=.3)
    p.add_argument("--pixel-threshold",type=float,default=.1)
    p.add_argument("--clahe-clip",type=float,default=0)
    p.add_argument("--contrast-space",choices=["gray","lab"],default="gray")
    a=p.parse_args()
    assert not a.target.exists() and .1<=a.box_threshold<=.7 and .05<=a.pixel_threshold<=.5
    assert 0<=a.clahe_clip<=8
    if bool(a.model_path)!=bool(a.model_sha256) or (a.model_path and a.backend not in {"ppocrv6","ppocrv5-server"}):
        p.error("Pin both candidate model path and SHA256 for a native OCR backend")
    if a.backend in {"ppocrv6","ppocrv5-server"}:
        if a.gpu is None:p.error("Native GPU locator requires an authorized GPU")
        from watermark_studio.training_reservation import offline_reservation
        reservation=offline_reservation(a.gpu)
        import os
        os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
    elif a.contrast_space!="gray":
        p.error("Lab contrast pilot requires a pinned native OCR backend")
    elif a.gpu is not None:
        p.error("CPU RapidOCR does not allocate a GPU")
    from PIL import Image
    from rapidocr_onnxruntime import RapidOCR

    from watermark_studio.images import load_photo

    helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    cases=helper["manifest_cases"](a.manifest)
    assert all(c.get("injected_remaining_overlay") for c in cases)
    hashes=helper["file_hashes"](cases)
    locator=None
    if a.backend in {"ppocrv6","ppocrv5-server"}:
        import torch

        from watermark_studio.native_text import OFFICIAL_SERVER_DET_SHA256, NativeTextLocator
        torch.set_num_threads(1)
        folder=("PaddlePaddle--PP-OCRv6_small_det_safetensors" if a.backend=="ppocrv6"
                else "PaddlePaddle--PP-OCRv5_server_det_safetensors")
        path=a.model_path or Path.home()/".cache/watermark-models/models"/folder/"snapshots/master"
        sha=a.model_sha256 if a.model_path else (OFFICIAL_SERVER_DET_SHA256 if a.backend=="ppocrv5-server" else None)
        locator=NativeTextLocator(path,resolution=a.resolution,box_threshold=a.box_threshold,
                                  pixel_threshold=a.pixel_threshold,clahe_clip=a.clahe_clip,contrast_space=a.contrast_space,
                                  **({"weights_sha256":sha} if sha else {}))
        expected="pp_ocrv6_small_det" if a.backend=="ppocrv6" else "pp_ocrv5_server_det"
        if locator.architecture!=expected:
            raise ValueError("Requested OCR backend differs from the pinned architecture")
        effective={"backend":a.backend,"resolution":a.resolution,"box_threshold":a.box_threshold,
                   "pixel_threshold":a.pixel_threshold,"clahe_clip":a.clahe_clip,
                   "contrast_space":a.contrast_space,"weight_sha256":locator.weights_sha256}
    else:
        engine=RapidOCR(det_use_cuda=False,rec_use_cuda=False,cls_use_cuda=False,
            intra_op_num_threads=1,inter_op_num_threads=1,det_limit_type="max",
            det_limit_side_len=a.resolution,max_side_len=a.resolution,
            det_box_thresh=a.box_threshold,det_thresh=a.pixel_threshold,use_cls=False,use_rec=False)
        assert engine.max_side_len==a.resolution and engine.text_det.limit_side_len==a.resolution
        effective={"resolution":a.resolution,"box_threshold":engine.text_det.postprocess_op.box_thresh,
                   "pixel_threshold":engine.text_det.postprocess_op.thresh,"clahe_clip":a.clahe_clip}
        assert effective["box_threshold"]==a.box_threshold and effective["pixel_threshold"]==a.pixel_threshold
    a.target.mkdir(parents=True);rows=[]
    began=time.monotonic()
    with (a.target/"results.jsonl").open("x") as log:
        for case in cases:
            photo=load_photo(case["output"])[0];started=time.monotonic()
            details={}
            if locator is not None:
                details=locator.inspect(photo)
                boxes=[record["box"] for record in details["candidates"]]
            else:
                diagnostic=photo
                if a.clahe_clip:
                    import cv2
                    gray=cv2.createCLAHE(clipLimit=a.clahe_clip,tileGridSize=(16,16)).apply(np.asarray(photo.convert("L")))
                    diagnostic=Image.fromarray(gray).convert("RGB")
                detected,_=engine(np.asarray(diagnostic)[...,::-1].copy(),use_det=True,use_cls=False,use_rec=False)
                boxes=[]
                for quad in ([] if detected is None else detected):
                    points=np.asarray(quad,dtype=float)
                    if points.shape!=(4,2) or not np.isfinite(points).all():
                        raise ValueError("Invalid native OCR quad")
                    lo,hi=points.min(axis=0),points.max(axis=0)
                    boxes.append([float(lo[0]/photo.width),float(lo[1]/photo.height),
                                  float(hi[0]/photo.width),float(hi[1]/photo.height)])
            target=case["injected_remaining_overlay"]["box"]
            row={"file":case["file"],"alpha":case["injected_remaining_overlay"]["alpha"],
                 "style":case["injected_remaining_overlay"].get("style",case["injected_remaining_overlay"].get("kind","unspecified")),"target":target,
                 "covered":any(location_covered(target,b) for b in boxes),"boxes":boxes,"detector":details,
                 "seconds":time.monotonic()-started,"source":case["source"],"output":case["output"]}
            rows.append(row);log.write(json.dumps(row)+"\n");log.flush()
            print(json.dumps({"done":len(rows),"coverage":sum(r["covered"] for r in rows),
                              "last_boxes":len(boxes),"last_seconds":round(row["seconds"],3)}),flush=True)
    assert helper["file_hashes"](cases)==hashes
    if locator is not None:assert reservation==offline_reservation(a.gpu)
    alpha=collections.defaultdict(lambda:{"cases":0,"covered":0})
    for row in rows:
        alpha[row["alpha"]]["cases"]+=1;alpha[row["alpha"]]["covered"]+=int(row["covered"])
    report={"state":"completed","images":len(rows),"effective_config":effective,
            "seconds":time.monotonic()-began,"by_alpha":dict(alpha),
            "missed":[r["file"] for r in rows if not r["covered"]],
            "mean_boxes":sum(len(r["boxes"]) for r in rows)/len(rows),
            "p95_seconds":float(np.quantile([r["seconds"] for r in rows],.95)),
            "scope":"Location coverage only. OCR text-like boxes do not prove a digital watermark or a successful rejection."}
    (a.target/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
