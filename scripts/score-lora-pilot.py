"""Re-score immutable synthetic pilot outputs with exact targets; no inference."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from PIL import Image

from watermark_studio.exact_metrics import reconstruction_metrics
from watermark_studio.images import load_photo
from watermark_studio.training_features import digest, validate_pairs


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--experiment",type=Path,required=True)
    a=p.parse_args()
    data,_=validate_pairs(a.manifest)
    pairs={r["id"]:r for r in data["pairs"] if r["split"]=="validation"}
    report=json.loads((a.experiment/"report.json").read_text())
    if report.get("state")!="completed" or digest(a.manifest)!=report["dataset_manifest_sha256"]:
        p.error("Completed same-dataset experiment required")
    records=[json.loads(x) for x in (a.experiment/"results.jsonl").read_text().splitlines()]
    if len(records)!=report["total"] or len({r["id"] for r in records})!=len(records):
        p.error("Missing/repeated validation outputs")
    values=[]
    for record in records:
        row=pairs[record["id"]];output=Path(record["output"]).resolve(strict=True)
        if (not output.is_relative_to((a.experiment/"output").resolve()) or
                digest(output)!=record["output_sha256"] or record["generation_count"]!=1):
            p.error("Exact once-generated output bytes required")
        actual=np.asarray(load_photo(output)[0].convert("RGB"))
        expected=np.asarray(load_photo(row["target"])[0].convert("RGB"))
        with Image.open(row["mask"]) as mask_image:mask=np.asarray(mask_image.convert("L"))>0
        values.append({"id":row["id"],"kind":row["render"]["kind"],
            "overlay":reconstruction_metrics(actual,expected,mask),
            "content":reconstruction_metrics(actual,expected,~mask),
            "output_sha256":record["output_sha256"]})
    cohorts={}
    for kind in sorted({r["kind"] for r in values}):
        cohort=[r for r in values if r["kind"]==kind]
        summary={"count":len(cohort)}
        for scope in ("overlay","content"):
            valid=[r[scope] for r in cohort if r[scope]["pixels"]]
            summary[scope]={key:statistics.mean(r[key] for r in valid) for key in
                ("psnr","mae","low_frequency_rmse")} if valid else None
        cohorts[kind]=summary
    result={"version":2,"dataset_manifest_sha256":digest(a.manifest),"experiment":str(a.experiment.resolve()),
        "cohorts":cohorts,"cases":values,"quality_permission":False,"image_generations":0}
    destination=a.experiment/"exact-metrics-v2.json"
    with destination.open("x") as stream:stream.write(json.dumps(result,indent=2))
    print(json.dumps({"cohorts":cohorts,"image_generations":0,"quality_permission":False}))


if __name__=="__main__":
    main()
