"""Independent exact-target shadow controls for a frozen boundary classifier; no approvals."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import runpy
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from watermark_studio.artifact_pairs import (
    build_guard,
    guard_decision,
    guard_input,
    physical_shadow_background,
)
from watermark_studio.images import load_photo
from watermark_studio.training_features import digest, validate_pairs
from watermark_studio.training_pairs import render_profile_pair


def controls(clean,box):
    """Known offsets are applied AFTER the actual background, never to source identity."""
    yield "exact",clean.copy(),"passed"
    stream=io.BytesIO();clean.save(stream,format="JPEG",quality=95);stream.seek(0)
    with Image.open(stream) as image:yield "jpeg95",image.convert("RGB"),"passed"
    l,t,r,b=box
    for name,delta in [("dark18",-18),("dark32",-32),("bright18",18)]:
        value=clean.copy();pixels=np.asarray(clean.crop(box),dtype=np.float32)
        edited=np.clip(pixels+delta,0,255).astype(np.uint8)
        if not np.any(np.abs(edited.astype(np.int16)-pixels)>2):continue
        value.paste(Image.fromarray(edited),(l,t,r,b))
        yield name,value,"review"


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--source-split",choices=["train","validation"],default="validation")
    p.add_argument("--original-limit",type=int,default=0)
    p.add_argument("--sampling-seed",type=int,default=2026110206)
    p.add_argument("--training-report",type=Path,required=True)
    p.add_argument("--checkpoint-sha256",required=True)
    p.add_argument("--step",type=int,required=True)
    p.add_argument("--target",type=Path,required=True)
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--font",type=Path,default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    a=p.parse_args()
    if a.gpu<0 or not 0<=a.original_limit<=128:p.error("Explicit disjoint GPU and bounded original count required")
    training=json.loads(a.training_report.read_text());data,files=validate_pairs(a.manifest)
    input_profile=training.get("input_profile")
    if training.get("state")!="completed" or input_profile not in ("boundary-strips-v1","boundary-strips-v2"):
        p.error("Completed versioned boundary-strips model required")
    if training["veto_threshold"]!=.995 or training["trained_defect_types"]!=["shadow_or_tint"]:
        p.error("Frozen shadow-only protocol required")
    if digest(a.font)!=data["font_sha256"]:p.error("Exact original font required")
    checkpoint=next((r for r in training["checkpoints"] if r["step"]==a.step),None)
    if not checkpoint or checkpoint["sha256"]!=a.checkpoint_sha256 or digest(checkpoint["path"])!=a.checkpoint_sha256:
        p.error("Exact frozen checkpoint required")
    if digest(training["dataset_manifest"])!=training["dataset_manifest_sha256"]:
        p.error("Frozen training manifest changed")
    original_training,_=validate_pairs(Path(training["dataset_manifest"]))
    used={r["source_original_sha256"] for r in original_training["pairs"]}
    validation=[r for r in data["pairs"] if r["split"]==a.source_split and
        (r["render"].get("render",{}).get("panel") or r["render"].get("panel"))]
    selected=sorted({r["source_original_sha256"] for r in validation},
        key=lambda sha:hashlib.sha256((str(a.sampling_seed)+sha).encode()).hexdigest())
    if a.original_limit:selected=selected[:a.original_limit]
    if a.original_limit and len(selected)!=a.original_limit:p.error("Insufficient untouched original candidates")
    validation=[r for r in validation if r["source_original_sha256"] in set(selected)]
    if not validation or {r["source_original_sha256"] for r in validation}&used:
        p.error("Independent original-SHA validation panels required")
    helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as response:
        reserved=helper["require_disjoint_server_gpus"](json.load(response),[a.gpu],True)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=training["arguments"].get("tf32",False)
    torch.backends.cudnn.allow_tf32=training["arguments"].get("tf32",False)
    with safe_open(checkpoint["path"],framework="pt") as weight:
        if weight.metadata().get("input_profile")!=input_profile:
            raise ValueError("Preprocessing identity mismatch")
    a.target.mkdir(exist_ok=False);work=a.target.resolve();began=time.monotonic()
    model=build_guard(False).eval().requires_grad_(False).to("cuda:0")
    model.load_state_dict(load_file(checkpoint["path"]),strict=True)
    load_seconds=time.monotonic()-began;began=time.monotonic();pending=[];records=[];preprocess_seconds=0.
    def flush():
        if not pending:return
        batch=torch.from_numpy(np.stack([r[0] for r in pending])).to("cuda:0")
        with torch.inference_mode():scores=model(batch).float().softmax(-1).cpu().numpy()
        for (_,info),score in zip(pending,scores,strict=True):
            records.append({**info,**guard_decision(score,["shadow_or_tint"]), "image_generations":0})
        pending.clear()
    for row in validation:
        clean=load_photo(row["target"])[0];original=load_photo(row["input"])[0]
        variant=int(row["id"].rsplit("-",1)[-1])
        recreated,_,_,overlay=render_profile_pair(clean,variant,random.Random(row["seed"]),a.font,
            profile=row["render"].get("profile","legacy"),return_overlay=True)
        if recreated.tobytes()!=original.tobytes():raise ValueError("Exact original overlay replay failed")
        box=tuple(row["render"].get("render",{}).get("panel") or row["render"].get("panel"))
        l,t,r,b=box;dx=.06*(r-l)+.008*clean.width;dy=.06*(b-t)+.008*clean.height
        padded=(max(0,l-dx),max(0,t-dy),min(clean.width,r+dx),min(clean.height,b+dy))
        for existing_shadow in (False,True):
            background=physical_shadow_background(clean,box,row["seed"]^0x7161) if existing_shadow else clean
            source=Image.alpha_composite(background.convert("RGBA"),overlay).convert("RGB")
            for kind,after,expected in controls(background,box):
                for scope,geometry in [("exact",box),("detector-padding-v1",padded)]:
                    started=time.monotonic()
                    value=guard_input(source,after,geometry,training["arguments"]["resolution"],profile=input_profile)
                    preprocess_seconds+=time.monotonic()-started
                    pending.append((value,{"id":row["id"],"source_original_sha256":row["source_original_sha256"],
                        "kind":kind,"scope":scope,"preexisting_shadow":existing_shadow,"expected":expected,
                        "quality_permission":False}))
                    if len(pending)>=32:flush()
    flush()
    if any(digest(path)!=sha for path,sha in files.items()):raise ValueError("Validation bytes changed")
    with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as response:
        if reserved!=helper["require_disjoint_server_gpus"](json.load(response),[a.gpu],True):
            raise ValueError("Server reservation changed")
    cohorts={}
    for scope in ("exact","detector-padding-v1"):
        group=[r for r in records if r["scope"]==scope]
        cohorts[scope]={"cases":len(group),"correct":sum(r["expected"]=="passed" for r in group),
            "false_vetoes":sum(r["expected"]=="passed" and r["would_veto"] for r in group),
            "defects":sum(r["expected"]=="review" for r in group),
            "defects_flagged":sum(r["expected"]=="review" and r["would_veto"] for r in group),
            "flagged_types":dict(Counter(r["kind"] for r in group if r["would_veto"]))}
    selection={r["source_original"]:r["source_original_sha256"] for r in validation}
    (work/"source-selection.json").write_text(json.dumps({"source_sha256":selection,
        "purpose":"Independent CNN synthetic shadow calibration; excluded from future native acceptance"}))
    report={"state":"completed","version":1,"input_profile":input_profile,"source_split":a.source_split,
        "source_selection_sha256":hashlib.sha256(json.dumps(selected).encode()).hexdigest(),"cases":len(records),"cohorts":cohorts,
        "validation_originals":len({r["source_original_sha256"] for r in validation}),
        "original_source_overlap":0,"seconds":round(time.monotonic()-began,3),
        "load_seconds":round(load_seconds,3),"preprocessing_seconds":round(preprocess_seconds,3),
        "checkpoint_sha256":a.checkpoint_sha256,"threshold":.995,"quality_permission":False,
        "image_generations":0,"production_changed":False,"scores_are_uncalibrated":True}
    (work/"results.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    (work/"report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=="__main__":main()
