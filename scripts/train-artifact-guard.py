"""Offline source-isolated paired CNN guard pilot; never approves images."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import runpy
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from watermark_studio.artifact_pairs import (
    CLASSES,
    build_guard,
    guard_input,
    physical_shadow_background,
    synthetic_case,
)
from watermark_studio.images import load_photo
from watermark_studio.training_features import digest, validate_pairs


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--target",type=Path,required=True)
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--steps",type=int,default=200)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--resolution",type=int,choices=[256,384],default=256)
    p.add_argument("--learning-rate",type=float,default=1e-4)
    p.add_argument("--validate-every",type=int,default=50)
    p.add_argument("--seed",type=int,default=88120)
    p.add_argument("--clean-fraction",type=float,default=.2)
    p.add_argument("--clean-weight",type=float,default=1.)
    p.add_argument("--source-shadow-negatives",action="store_true")
    p.add_argument("--input-profile",choices=["context","boundary-strips-v1"],default="context")
    p.add_argument("--shadow-profile",choices=["legacy","mixed-v1"],default="legacy")
    p.add_argument("--font",type=Path,default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    p.add_argument("--pretrained",action="store_true")
    p.add_argument("--tf32",action="store_true")
    p.add_argument("--preserved-markings",action="store_true",
        help="Augment paired unchanged scene-printing negatives; never authorizes watermark acceptance")
    a=p.parse_args()
    if (a.gpu<0 or not 50<=a.steps<=1000 or not 4<=a.batch_size<=64 or
            not 1e-6<=a.learning_rate<=1e-3 or not 1<=a.validate_every<=a.steps or
            not .2<=a.clean_fraction<=.7 or not 1<=a.clean_weight<=4):
        p.error("Explicit isolated GPU and bounded pilot settings required")
    data,_=validate_pairs(a.manifest)
    if a.source_shadow_negatives and (not a.font.is_file() or digest(a.font)!=data["font_sha256"]):
        p.error("Exact original overlay font required for causal shadow negatives")
    if a.target.resolve().parent!=a.manifest.resolve().parent:p.error("Stay inside pilot root")
    a.target.mkdir(exist_ok=False);work=a.target.resolve()
    helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    def isolation():
        with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as response:
            return helper["require_disjoint_server_gpus"](json.load(response),[a.gpu],True)
    reserved=isolation();manifest_sha=digest(a.manifest)
    revision=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
    import torch
    from safetensors.torch import save_file
    torch.set_num_threads(2);torch.manual_seed(a.seed);rng=random.Random(a.seed)
    torch.backends.cuda.matmul.allow_tf32=a.tf32
    torch.backends.cudnn.allow_tf32=a.tf32
    began=time.monotonic();model=build_guard(a.pretrained).to("cuda:0")
    pretrained_artifact=None
    if a.pretrained:
        path=Path(torch.hub.get_dir())/"checkpoints/resnet18-f37072fd.pth"
        sha=digest(path)
        if not sha.startswith("f37072fd"):raise ValueError("Official pretrained SHA prefix mismatch")
        pretrained_artifact={"path":str(path),"sha256":sha,
            "url":"https://download.pytorch.org/models/resnet18-f37072fd.pth"}
    # Keep ImageNet batch statistics fixed. Tiny heterogeneous photo batches
    # must not change inference behavior through accidental BN calibration.
    def train_mode():
        model.train()
        for module in model.modules():
            if isinstance(module,torch.nn.BatchNorm2d):module.eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=1e-4)
    load_seconds=time.monotonic()-began
    photos={}
    def photo(path):
        if path not in photos:photos[path]=load_photo(path)[0]
        return photos[path]
    masks={}
    def mask(path):
        if path not in masks:
            with Image.open(path) as image:masks[path]=image.convert("L")
        return masks[path]
    overlays={}
    def make(row,label,seed,force_shadow=None):
        original=photo(row["input"]);clean=photo(row["target"])
        if a.source_shadow_negatives:
            if row["id"] not in overlays:
                from watermark_studio.training_pairs import render_profile_pair
                variant=int(row["id"].rsplit("-",1)[-1])
                rendered,_,_,overlay=render_profile_pair(clean,variant,random.Random(row["seed"]),a.font,profile=row.get("render",{}).get("profile","legacy"),return_overlay=True)
                if rendered.tobytes()!=original.tobytes():
                    raise ValueError("Overlay re-render does not reproduce exact original training source")
                overlays[row["id"]]=overlay
            use_shadow=random.Random(seed^0x7161).random()<.35 if force_shadow is None else force_shadow
            if use_shadow:
                bounds=row["render"].get("render",{}).get("panel") or row["render"].get("panel") or mask(row["mask"]).getbbox()
                if bounds is None:bounds=(clean.width*.2,clean.height*.2,clean.width*.7,clean.height*.7)
                clean=physical_shadow_background(clean,bounds,seed^0x18E9)
                original=Image.alpha_composite(clean.convert("RGBA"),overlays[row["id"]]).convert("RGB")
        source,after,box,info=synthetic_case(original,clean,mask(row["mask"]),
            row["render"],label,seed,preserved_marks=a.preserved_markings,shadow_profile=a.shadow_profile)
        jitter=random.Random(seed^0x5319)
        l,t,r,b=box
        limit=.008 if a.input_profile=="boundary-strips-v1" else .025
        box=(max(0,l-jitter.uniform(0,limit)*source.width),
            max(0,t-jitter.uniform(0,limit)*source.height),
            min(source.width,r+jitter.uniform(0,limit)*source.width),
            min(source.height,b+jitter.uniform(0,limit)*source.height))
        return guard_input(source,after,box,a.resolution,profile=a.input_profile),info["label"]
    train=[r for r in data["pairs"] if r["split"]=="train"]
    validation=[r for r in data["pairs"] if r["split"]=="validation"]
    validation_cases=[]
    for row in validation:
        for label in ((0,1) if a.input_profile=="boundary-strips-v1" else range(len(CLASSES))):
            seed=(int(hashlib.sha256(row["id"].encode()).hexdigest()[:8],16)+label*7919+a.seed)%(2**32)
            value,actual_label=make(row,label,seed,False)
            validation_cases.append((value,actual_label,row["id"],label))
            if a.source_shadow_negatives:
                value,actual_label=make(row,label,seed,True)
                validation_cases.append((value,actual_label,row["id"]+"/source_shadow",label))
    def record(value):
        with (work/"metrics.jsonl").open("a") as stream:stream.write(json.dumps(value)+"\n")
        print(json.dumps(value),flush=True)
    def validate(step):
        model.eval();records=[];started=time.monotonic()
        with torch.inference_mode():
            for offset in range(0,len(validation_cases),a.batch_size):
                group=validation_cases[offset:offset+a.batch_size]
                batch=torch.from_numpy(np.stack([r[0] for r in group])).to("cuda:0")
                probabilities=model(batch).float().softmax(dim=-1).cpu().numpy()
                if not np.isfinite(probabilities).all():raise ValueError("Nonfinite guard validation scores")
                for (_,label,name,requested),scores in zip(group,probabilities,strict=True):
                    records.append({"id":name,"label":label,"requested":requested,
                        "scores":[float(v) for v in scores],"prediction":int(np.argmax(scores))})
        confusion=[[0]*len(CLASSES) for _ in CLASSES]
        for row in records:confusion[row["label"]][row["prediction"]]+=1
        clean=[r for r in records if r["label"]==0];bad=[r for r in records if r["label"]!=0]
        threshold=.995
        result={"stage":"validation","step":step,"cases":len(records),"confusion":confusion,
            "veto_threshold":threshold,
            "clean_vetoed":sum(max(r["scores"][1:])>=threshold for r in clean),"clean_cases":len(clean),
            "defects_flagged":sum(max(r["scores"][1:])>=threshold for r in bad),"defect_cases":len(bad),
            "seconds":round(time.monotonic()-started,3),"quality_permission":False,
            "scores_are_uncalibrated":True}
        (work/f"validation-step{step:04d}.json").write_text(json.dumps(records))
        record(result);train_mode()
    provenance={"state":"running","version":1,"model_type":"paired_resnet18_v1","classes":list(CLASSES),
        "dataset_manifest":str(a.manifest.resolve()),"dataset_manifest_sha256":manifest_sha,
        "input_profile":a.input_profile,"trained_defect_types":([CLASSES[1]] if a.input_profile=="boundary-strips-v1" else list(CLASSES[1:])),
        "code_revision":revision,"arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        "pretrained_artifact":pretrained_artifact,"train_originals":len({r["source_original_sha256"] for r in train}),
        "validation_originals":len({r["source_original_sha256"] for r in validation}),
        "train_source_overlap":0,"diffusion_generations":0,"production_changed":False,
        "load_seconds":round(load_seconds,3),"veto_threshold":.995}
    (work/"report.json").write_text(json.dumps(provenance,indent=2))
    validate(0);began=time.monotonic();checkpoints=[]
    for step in range(1,a.steps+1):
        started=time.monotonic();batch=[];labels=[]
        for index in range(a.batch_size):
            row=rng.choice(train)
            requested=0 if rng.random()<a.clean_fraction else (1 if a.input_profile=="boundary-strips-v1" else rng.randrange(1,len(CLASSES)))
            value,label=make(row,requested,rng.randrange(2**32))
            batch.append(value);labels.append(label)
        inputs=torch.from_numpy(np.stack(batch)).to("cuda:0")
        targets=torch.tensor(labels,device="cuda:0",dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits=model(inputs)
        weights=torch.tensor([a.clean_weight,1.,1.,1.,1.],device="cuda:0")
        loss=torch.nn.functional.cross_entropy(logits,targets,weight=weights)
        if not torch.isfinite(loss):raise ValueError("Nonfinite guard training loss")
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if step%10==0:record({"stage":"training","step":step,"loss":float(loss.detach()),
            "seconds":round(time.monotonic()-started,3),"peak_gpu_bytes":torch.cuda.max_memory_allocated()})
        if step%a.validate_every==0 or step==a.steps:validate(step)
        if step%100==0 or step==a.steps:
            path=work/f"guard-step{step:04d}.safetensors"
            save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(path),
                metadata={"model_type":"paired_resnet18_v1","dataset_manifest_sha256":manifest_sha,
                    "classes":json.dumps(CLASSES),"code_revision":revision,"resolution":str(a.resolution),"input_profile":a.input_profile})
            checkpoints.append({"path":str(path),"sha256":digest(path),"step":step})
        if step%50==0 and (reserved!=isolation() or revision!=subprocess.check_output(
                ["git","rev-parse","--short","HEAD"],text=True).strip()):
            raise ValueError("Code/isolation changed during training")
    if manifest_sha!=digest(a.manifest):raise ValueError("Training dataset changed")
    provenance.update(state="completed",seconds=round(time.monotonic()-began,3),
        checkpoints=checkpoints,peak_gpu_bytes=torch.cuda.max_memory_allocated())
    (work/"report.json").write_text(json.dumps(provenance,indent=2))
    print(json.dumps({k:provenance[k] for k in ("state","seconds","checkpoints")}),flush=True)


if __name__=="__main__":
    main()
