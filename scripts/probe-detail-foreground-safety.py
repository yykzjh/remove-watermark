"""Known-target faint-ink/camera-body challenge; no new generation or native modification."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import runpy
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from watermark_studio.detail_foreground import STYLES, detail_support, foreground_challenge
from watermark_studio.images import load_photo
from watermark_studio.text_presence import TargetTextPresence
from watermark_studio.training_features import digest, validate_pairs
from watermark_studio.transmitted_detail import estimate_patch_transmission, recover_patch_detail


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--target",type=Path,required=True)
    p.add_argument("--split",choices=["train","validation"],default="validation")
    p.add_argument("--original-limit",type=int,default=24)
    p.add_argument("--seed",type=int,default=2026110211)
    p.add_argument("--font",type=Path,default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    p.add_argument("--mask-model-report",type=Path)
    p.add_argument("--mask-model-sha256")
    p.add_argument("--gpu",type=int,default=-1)
    a=p.parse_args()
    if not 1<=a.original_limit<=32 or not a.font.is_file():
        p.error("Bounded independently screened original sources and an explicit font required")
    if a.target.resolve().parent!=a.manifest.resolve().parent:
        p.error("Keep the diagnostic inside its source dataset root")
    data,_=validate_pairs(a.manifest);manifest_sha=digest(a.manifest)
    # Prefer the actual lower-frame crops when available; no quality/output labels.
    groups=collections.defaultdict(list)
    for row in data["pairs"]:
        if row["split"]==a.split:groups[row["source_original_sha256"]].append(row)
    identities=sorted(groups,key=lambda s:hashlib.sha256((str(a.seed)+s).encode()).hexdigest())[:a.original_limit]
    if len(identities)!=a.original_limit:p.error("Not enough original identities")
    selected=[min(groups[s],key=lambda r:("lower1024" not in r["id"],r["target_sha256"],r["id"]))
        for s in identities]
    selection=[{"id":r["id"],"source_original_sha256":r["source_original_sha256"],
        "source_original":r["source_original"],
        "target":r["target"],"target_sha256":r["target_sha256"]} for r in selected]
    model=None;model_identity=None
    if bool(a.mask_model_report)!=bool(a.mask_model_sha256):
        p.error("Both trained foreground report and exact checkpoint SHA are required")
    if a.mask_model_report:
        if a.gpu<0:p.error("Explicit disjoint classifier GPU required")
        from watermark_studio.detail_foreground_model import (
            DILATION,
            MODEL_TYPE,
            THRESHOLD,
            TILE,
            build_foreground_model,
            foreground_scores,
        )
        training=json.loads(a.mask_model_report.read_text())
        if (training.get("state")!="completed" or training["model_type"]!=MODEL_TYPE
                or training["inference_threshold"]!=THRESHOLD or training["inference_dilation"]!=DILATION
                or training["tile_size"]!=TILE or training["checkpoint_sha256"]!=a.mask_model_sha256
                or digest(training["checkpoint"])!=a.mask_model_sha256
                or digest(training["manifest"])!=training["manifest_sha256"]):
            raise ValueError("Immutable trained model and precommitted pixel protocol required")
        original_training,_=validate_pairs(Path(training["manifest"]))
        train_ids={r["source_original_sha256"] for r in original_training["pairs"] if r["split"]=="train"}
        if set(identities)&train_ids:raise ValueError("Foreground evaluation source overlaps training")
        helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
        with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as response:
            helper["require_disjoint_server_gpus"](json.load(response),[a.gpu],True)
        os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file
        torch.set_num_threads(2)
        with safe_open(training["checkpoint"],framework="pt") as weights:
            if weights.metadata().get("model_type")!=MODEL_TYPE:
                raise ValueError("Classifier architecture metadata differs")
        model=build_foreground_model().eval().requires_grad_(False).to("cuda:0")
        model.load_state_dict(load_file(training["checkpoint"]),strict=True)
        model_identity={"checkpoint_sha256":a.mask_model_sha256,"train_source_overlap":0,
            "threshold":THRESHOLD,"dilation":DILATION,"quality_permission":False}
    selection_raw=json.dumps({"version":1,"sources":[{"path":r["source_original"],
        "sha256":r["source_original_sha256"]} for r in selected],"derived_targets":selection,
        "scope":"Existing screened source-original roles, no new source selection by quality"},
        ensure_ascii=False,sort_keys=True)
    a.target.mkdir(exist_ok=False);detector=TargetTextPresence(upsample_small=True)
    began=time.monotonic();rows=[];model_total_seconds=0.
    for index,row in enumerate(selected):
        clean=load_photo(row["target"])[0]
        generated=clean.filter(ImageFilter.GaussianBlur(2))
        for style_i,style in enumerate(STYLES):
            for quality in (0,95):
                seed=(a.seed+index*1009+style_i*7919)%(2**32)
                source,neutral,glyph,metadata=foreground_challenge(clean,a.font,style,seed,quality=quality)
                l,t,r,b=metadata["body_box"];box=(l+4,t+4,r-4,b-4)
                target=[box[0]/clean.width,box[1]/clean.height,box[2]/clean.width,box[3]/clean.height]
                detected=detector.inspect(source,[target])
                foreground_veto=None;model_seconds=0.
                if model is not None:
                    started=time.monotonic()
                    scores=foreground_scores(model,source,generated,box,"cuda:0")
                    foreground_veto=np.asarray(Image.fromarray((scores>THRESHOLD).astype(np.uint8)*255)
                        .filter(ImageFilter.MaxFilter(2*DILATION+1)))>0
                    model_seconds=time.monotonic()-started
                    model_total_seconds+=model_seconds
                profiles=("pixel","ocr2","ocr8","ocr8_nn") if model is not None else ("pixel","ocr2","ocr8")
                for profile in profiles:
                    support,proof=detail_support(source,box,detected,
                        profile="ocr8" if profile=="ocr8_nn" else profile)
                    if profile=="ocr8_nn":
                        support &= ~foreground_veto
                        proof["foreground_model"]=model_identity
                    area=support[t:b,l:r];crop=support[box[1]:box[3],box[0]:box[2]].astype(float)
                    record={"original":row["source_original_sha256"],"target_sha256":row["target_sha256"],
                        "case":f"{index:03d}-{style}-{quality}-{profile}","profile":profile,
                        **metadata,"support":proof,"ocr_status":detected["status"],
                        "ocr_seconds":detected["seconds"],"foreground_model_seconds":model_seconds,
                        "support_fraction":float(area.mean()),"glyph_leak_pixels":int((support & glyph).sum()),
                        "foreground_pixels":int(glyph[t:b,l:r].sum()),"extra_generations":0}
                    rows.append(record)
                    if crop.mean()<.1:continue
                    fitted=estimate_patch_transmission(source.crop(box),generated.crop(box),crop,
                        attenuation_aware=True,low_transmission_experiment=True)
                    record["fit"]=fitted
                    if not fitted["eligible"]:continue
                    params={"transmission":fitted["transmission"],"radius":16,"max_delta":24,
                        "low_transmission_experiment":True}
                    actual,_=recover_patch_detail(source.crop(box),generated.crop(box),crop,**params)
                    counter,_=recover_patch_detail(neutral.crop(box),generated.crop(box),crop,**params)
                    impact=np.abs(np.asarray(actual).astype(np.int16)-np.asarray(counter).astype(np.int16))
                    record.update(recovery_eligible=True,foreground_impact_max=int(impact.max()),
                        foreground_impact_mean=float(impact.mean()))
        print(json.dumps({"completed_originals":index+1,"total":len(selected)}),flush=True)
    summaries={}
    for profile in (("pixel","ocr2","ocr8","ocr8_nn") if model is not None else ("pixel","ocr2","ocr8")):
        rr=[r for r in rows if r["profile"]==profile]
        summaries[profile]={"cases":len(rr),"leak_cases":sum(r["glyph_leak_pixels"]>0 for r in rr),
            "usable_support_cases":sum(r["support_fraction"]>=.1 for r in rr),
            "eligible_recovery_cases":sum(r.get("recovery_eligible",False) for r in rr),
            "eligible_with_foreground_impact_gt2":sum(r.get("foreground_impact_max",0)>2 for r in rr),
            "by_style":{style:{"cases":sum(r["style"]==style for r in rr),
                "leak_cases":sum(r["style"]==style and r["glyph_leak_pixels"]>0 for r in rr)}
                for style in STYLES}}
    if digest(a.manifest)!=manifest_sha:raise ValueError("Source manifest changed")
    validate_pairs(a.manifest)
    report={"state":"completed","originals":len(selected),"split":a.split,"source_selection_sha256":
        hashlib.sha256(selection_raw.encode()).hexdigest(),"manifest_sha256":manifest_sha,
        "font_sha256":digest(a.font),"foreground_model":model_identity,
        "foreground_model_total_seconds":round(model_total_seconds,3),"seconds":round(time.monotonic()-began,3),
        "profiles":summaries,"extra_generations":0,"native_images_modified":0,
        "quality_permission":False,"source_body_certified":False,
        "scope":"Deliberate synthetic foreground stress tests; truth only measures leakage, never changes support"}
    (a.target/"source-selection.json").write_text(selection_raw)
    (a.target/"results.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    (a.target/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
