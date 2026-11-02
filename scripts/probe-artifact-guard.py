"""Frozen paired CNN calibration on known controls/native pairs; vetoes only."""
from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

from watermark_studio.artifact_pairs import CLASSES, build_guard, guard_decision, guard_input
from watermark_studio.images import load_photo, pixel_box
from watermark_studio.training_features import digest


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--pair-manifest",type=Path,required=True)
    p.add_argument("--training-report",type=Path,required=True)
    p.add_argument("--checkpoint-sha256",required=True)
    p.add_argument("--step",type=int,default=200)
    p.add_argument("--target",type=Path,required=True)
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--tf32",action="store_true")
    p.add_argument("--defect-types",nargs="+",choices=list(CLASSES[1:]),default=list(CLASSES[1:]),
        help="Explicit independently validated heads; raw probabilities are never renormalized")
    a=p.parse_args()
    if a.gpu<0 or not 1<=a.batch_size<=128:p.error("Isolated GPU and bounded batch required")
    training=json.loads(a.training_report.read_text())
    if training.get("state")!="completed" or training.get("model_type")!="paired_resnet18_v1":
        p.error("Completed versioned trained guard required")
    checkpoint=next((r for r in training["checkpoints"] if r["step"]==a.step),None)
    if not checkpoint or checkpoint["sha256"]!=a.checkpoint_sha256 or digest(checkpoint["path"])!=a.checkpoint_sha256:
        p.error("Exact frozen adapter identity required")
    input_profile=training.get("input_profile","context")
    if input_profile not in ("context","boundary-strips-v1","boundary-strips-v2"):p.error("Recognized training preprocessing required")
    if any(kind not in training.get("trained_defect_types",CLASSES[1:]) for kind in a.defect_types):
        p.error("Only explicitly trained detector heads may be tested")
    threshold=training["veto_threshold"]
    if threshold!=.995:p.error("Precommitted conservative experimental veto threshold required")
    helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    cases=helper["manifest_cases"](a.pair_manifest)
    with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as response:
        isolation=helper["require_disjoint_server_gpus"](json.load(response),[a.gpu],True)
    revision=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    a.target.mkdir(exist_ok=False);work=a.target.resolve()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=a.tf32
    torch.backends.cudnn.allow_tf32=a.tf32
    with safe_open(checkpoint["path"],framework="pt") as weights:
        if weights.metadata().get("input_profile","context")!=input_profile:
            raise ValueError("Checkpoint/report preprocessing identity differs")
    began=time.monotonic();model=build_guard(False)
    model.load_state_dict(load_file(checkpoint["path"]),strict=True);model.eval().requires_grad_(False).to("cuda:0")
    load_seconds=time.monotonic()-began
    inputs=[];positions=[]
    records=[{"file":c["file"],"cohort":c.get("cohort","unknown"),"expected":c.get("expected"),
        "regions":[],"would_veto":False,"image_generations":0,"quality_permission":False} for c in cases]
    began=time.monotonic()
    for index,case in enumerate(cases):
        before=load_photo(case["source"])[0];after=load_photo(case["output"])[0]
        if before.size!=after.size:raise ValueError("Native image dimensions changed")
        for region in case["regions"]:
            box=pixel_box(region["box"],before.size)
            if input_profile.startswith("boundary-strips-"):
                l,t,r,b=box
                if (region["kind"]!="panel" or (r-l)*(b-t)<.025*before.width*before.height
                        or not (l>1 or r<before.width-1) or not (t>1 or b<before.height-1)):
                    records[index]["regions"].append({"box":region["box"],"would_veto":False,
                        "quality_permission":False,"reason":"boundary_profile_not_observable_in_trained_scope"})
                    continue
            inputs.append(guard_input(before,after,box,training["arguments"]["resolution"],profile=input_profile))
            positions.append((index,region))
    prep_seconds=time.monotonic()-began;began=time.monotonic()
    with torch.inference_mode():
        for offset in range(0,len(inputs),a.batch_size):
            batch=torch.from_numpy(np.stack(inputs[offset:offset+a.batch_size])).to("cuda:0")
            scores=model(batch).float().softmax(-1).cpu().numpy()
            if not np.isfinite(scores).all():raise ValueError("Nonfinite guard score")
            for (index,region),values in zip(positions[offset:offset+a.batch_size],scores,strict=True):
                decision=guard_decision(values,a.defect_types,threshold)
                records[index]["regions"].append({"box":region["box"],**decision})
                records[index]["would_veto"]|=decision["would_veto"]
    inference_seconds=time.monotonic()-began
    if revision!=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip():
        raise ValueError("Code changed during guard calibration")
    helper["manifest_cases"](a.pair_manifest)
    with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as response:
        if isolation!=helper["require_disjoint_server_gpus"](json.load(response),[a.gpu],True):
            raise ValueError("GPU reservations changed")
    cohorts={}
    for cohort in sorted({r["cohort"] for r in records}):
        group=[r for r in records if r["cohort"]==cohort]
        cohorts[cohort]={"cases":len(group),"vetoes":sum(r["would_veto"] for r in group),
            "clean_cases":sum(r["expected"]=="passed" for r in group),
            "clean_vetoed":[r["file"] for r in group if r["expected"]=="passed" and r["would_veto"]],
            "defects_flagged":sum(r["expected"]=="review" and r["would_veto"] for r in group),
            "defect_cases":sum(r["expected"]=="review" for r in group),
            "types":dict(Counter(region["type"] for r in group for region in r["regions"] if region["would_veto"]))}
    report={"state":"completed","version":1,"checkpoint_sha256":a.checkpoint_sha256,
        "training_report_sha256":digest(a.training_report),"pair_manifest_sha256":digest(a.pair_manifest),
        "code_revision":revision,"input_profile":input_profile,"threshold":threshold,"defect_types":a.defect_types,"scores_are_uncalibrated":True,"cohorts":cohorts,
        "load_seconds":round(load_seconds,3),"preprocessing_seconds":round(prep_seconds,3),
        "inference_seconds":round(inference_seconds,3),"patches":len(inputs),"cases":len(cases),
        "quality_permission":False,"image_generations":0,"production_changed":False}
    (work/"results.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    (work/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
