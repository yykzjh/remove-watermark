"""Optimistic classical inpaint screen using exact synthetic masks and known clean targets."""
import argparse
import hashlib
import json
import math
import runpy
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from watermark_studio.config import Region
from watermark_studio.images import load_photo, pixel_box
from watermark_studio.quality import repair_boundary_warnings, repair_texture_warnings

p=argparse.ArgumentParser()
p.add_argument("--controls",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
assert not a.output.exists()
cv2.setNumThreads(1)
helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
cases=[c for c in helper["known_control_cases"](a.controls) if c["kind"]=="exact_removal"]
assert cases
rows=[]
for c in cases:
    source=load_photo(c["source"])[0];truth=load_photo(c["output"])[0]
    digest=hashlib.sha256(Path(c["source"]).read_bytes()).hexdigest()
    rgb=np.asarray(source);gt=np.asarray(truth).astype(np.float32)
    mask=np.zeros(rgb.shape[:2],np.uint8)
    for core in c["source_cores"]:
        x0,y0,x1,y1=pixel_box(core["core"],source.size);mask[y0:y1,x0:x1]=255
    began=time.monotonic()
    out=cv2.inpaint(rgb,mask,3,cv2.INPAINT_TELEA)
    seconds=time.monotonic()-began
    sel=mask>0;err=out.astype(np.float32)-gt
    mae=float(np.abs(err[sel]).mean());mse=float(np.square(err[sel]).mean())
    output_gray=cv2.cvtColor(out,cv2.COLOR_RGB2GRAY)
    truth_gray=cv2.cvtColor(np.asarray(truth),cv2.COLOR_RGB2GRAY)
    core=cv2.erode(mask,np.ones((9,9),np.uint8))>0
    oa=np.abs(cv2.Laplacian(output_gray,cv2.CV_32F))[core].mean()
    ta=np.abs(cv2.Laplacian(truth_gray,cv2.CV_32F))[core].mean()
    assert digest==hashlib.sha256(Path(c["source"]).read_bytes()).hexdigest()
    assert np.array_equal(out[~sel],rgb[~sel])
    regions=[Region.model_validate(r) for r in c["regions"]]
    def guards(photo,regions=regions,source=source):
        return {"texture":repair_texture_warnings(photo,regions,before=source)["count"],
                "boundary":repair_boundary_warnings(photo,regions,before=source)["count"]}
    measured=guards(Image.fromarray(out));clean_guards=guards(truth)
    rows.append({"numeric_guards":measured,"truth_numeric_guards":clean_guards,"file":c["file"],"size":source.size,"seconds":seconds,"masked_mae":mae,
                 "masked_psnr":10*math.log10(255**2/max(mse,1e-12)),
                 "texture_energy_ratio":float(oa/max(ta,1e-6))})
    if len(rows)<=3:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(out).save(a.output.parent/("telea-"+c["file"]))
report={"scope":"exact-mask known-ground-truth CPU screen; no detector/QA cost included, not real-dataset acceptance",
        "cases":len(rows),"seconds":sum(c["seconds"] for c in rows),"mean_masked_mae":sum(c["masked_mae"] for c in rows)/len(rows),
        "mean_masked_psnr":sum(c["masked_psnr"] for c in rows)/len(rows),
        "mean_texture_energy_ratio":sum(c["texture_energy_ratio"] for c in rows)/len(rows),
        "new_numeric_vetoes":sum(any(c["numeric_guards"][k] and not c["truth_numeric_guards"][k] for k in ["texture","boundary"]) for c in rows),
        "originals_unchanged":True,"outside_masks_unchanged":True,"rows":rows}
a.output.write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!="rows"}))
