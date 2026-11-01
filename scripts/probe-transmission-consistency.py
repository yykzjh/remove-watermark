"""Read-only physical consistency of a translucent card and a first candidate."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from watermark_studio.frozen_candidates import digest
from watermark_studio.images import load_photo, pixel_box
from watermark_studio.text_presence import TargetTextPresence
from watermark_studio.translucent_recovery import fit_transmission

p=argparse.ArgumentParser()
p.add_argument("--cases",type=Path,required=True,help="Existing read-only SR screen report")
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
if a.output.exists():p.error("Fresh report required")
cases=json.loads(a.cases.read_text())["rows"]
if not 1<=len(cases)<=16:p.error("Bounded diagnostic cohort required")
from rapidocr_onnxruntime import RapidOCR

detector=TargetTextPresence(RapidOCR(det_use_cuda=False,rec_use_cuda=False,cls_use_cuda=False,
    intra_op_num_threads=1,inter_op_num_threads=1,det_limit_type="max",
    det_limit_side_len=1536,max_side_len=1536,det_box_thresh=.3,use_cls=False,use_rec=False))
at=time.monotonic();reports=[]
for row in cases:
    source=Path(row["source"]);candidate=Path(row["first_candidate"])
    assert digest(source)==row["source_sha256"]
    assert digest(candidate)==row["first_candidate_sha256"]
    before=load_photo(source)[0];after=load_photo(candidate)[0]
    largest=max(row["protected_core_boxes"],
                key=lambda b:(b["core"][2]-b["core"][0])*(b["core"][3]-b["core"][1]))["core"]
    box=pixel_box(largest,before.size)
    crop=before.crop(box);size=crop.size
    crop.thumbnail((512,512),Image.Resampling.LANCZOS)
    scale=(crop.width/size[0],crop.height/size[1])
    observed=np.asarray(crop.filter(ImageFilter.GaussianBlur(1)),dtype=np.float32)
    predicted=np.asarray(after.crop(box).resize(crop.size,Image.Resampling.LANCZOS).filter(
        ImageFilter.GaussianBlur(1)),dtype=np.float32)
    scan=detector.inspect(before,[largest])
    mask=Image.new("L",crop.size,0);draw=ImageDraw.Draw(mask)
    for target in scan.get("regions",[]):
        for text in target["textlike_boxes"]:
            x0,y0,x1,y1=pixel_box(text,before.size)
            draw.rectangle(((x0-box[0])*scale[0],(y0-box[1])*scale[1],
                            (x1-box[0])*scale[0],(y1-box[1])*scale[1]),fill=255)
    blocked=np.asarray(mask.filter(ImageFilter.MaxFilter(9)))>0
    valid=(~blocked)&(np.ptp(observed,axis=2)<45)&(observed.mean(axis=2)>135)&(observed.mean(axis=2)<250)
    valid[:12]=False;valid[-12:]=False;valid[:,:12]=False;valid[:,-12:]=False
    yy,xx=np.mgrid[:crop.height,:crop.width]
    halves=[valid,valid&(xx<crop.width//2),valid&(xx>=crop.width//2)]
    fits=[fit_transmission(predicted[v],observed[v]) for v in halves]
    reports.append({"file":row["file"],"source_sha256":row["source_sha256"],
        "candidate_sha256":row["first_candidate_sha256"],"source_text_scan":scan["status"],
        "valid_pixels":int(valid.sum()),"valid_fraction":float(valid.mean()),"fits":fits})
    assert digest(source)==row["source_sha256"] and digest(candidate)==row["first_candidate_sha256"]
report={"scope":"physical-fit diagnostics only; no source/output changes or quality approval",
        "seconds":round(time.monotonic()-at,3),"cases":len(reports),"rows":reports}
a.output.write_text(json.dumps(report,indent=2))
print(json.dumps({"cases":len(reports),"seconds":report["seconds"],
    "fits":[{"file":r["file"],"valid_fraction":round(r["valid_fraction"],3),
             "fits":[{k:f.get(k) for k in ["ok","transmission","r2","rmse","reason"]} for f in r["fits"]]} for r in reports]}),flush=True)
