"""Offline tiny text-location sensitivity pilot; never a watermark quality classifier."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import runpy
import subprocess
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--target",type=Path,required=True)
    p.add_argument("--gpu",type=int,choices=[0,1,2,3],required=True)
    p.add_argument("--steps",type=int,default=300)
    p.add_argument("--learning-rate",type=float,default=5e-5)
    p.add_argument("--seed",type=int,default=2026110393)
    a=p.parse_args()
    assert not a.target.exists() and 1<=a.steps<=1000 and 0<a.learning_rate<=2e-4
    from watermark_studio.training_reservation import offline_reservation
    reservation=offline_reservation(a.gpu)
    helper=runpy.run_path(str(Path(__file__).with_name("train-verifier-presence-lora.py")))
    data,manifest_sha=helper["validate_manifest"](a.manifest)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
    import numpy as np
    import torch
    from torch.nn import functional as F

    from watermark_studio.images import load_photo
    from watermark_studio.native_text import NativeTextLocator
    from watermark_studio.training_features import digest

    torch.set_num_threads(1);torch.manual_seed(a.seed);rng=random.Random(a.seed)
    base=Path.home()/".cache/watermark-models/models/PaddlePaddle--PP-OCRv6_small_det_safetensors/snapshots/master"
    locator=NativeTextLocator(base,resolution=2048)
    teacher=locator.model;model=copy.deepcopy(teacher).requires_grad_(True).eval()
    by_id={r["id"]:r for r in data["rows"]}
    rows=[]
    for row in data["rows"]:
        if row["overlay_present"] and row["render"].get("kind")=="scattered_faint_text":
            clean=by_id[row["id"].removesuffix("-marked")+"-clean"]
            assert clean["source_original_sha256"]==row["source_original_sha256"] and clean["split"]==row["split"]
            rows.append((row,clean))
    train=[pair for pair in rows if pair[0]["split"]=="train"]
    validation=[pair for pair in rows if pair[0]["split"]=="validation"]
    assert train and validation
    cache={}
    def prepare(pair):
        row,clean=pair
        if row["id"] in cache:return cache[row["id"]]
        before=load_photo(clean["after"])[0].convert("RGB")
        marked=load_photo(row["after"])[0].convert("RGB")
        delta=np.abs(np.asarray(before,dtype=np.int16)-np.asarray(marked,dtype=np.int16)).max(axis=2)>1
        yy,xx=np.where(delta)
        if len(xx)<16:raise ValueError("Invisible or missing labeled text")
        left,top,right,bottom=int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)
        # DB text detection supervises a word region, not an isolated glyph edge.
        inset=max(1,round((bottom-top)*.15))
        mask=np.zeros((marked.height,marked.width),dtype=np.float32)
        mask[top+inset:bottom-inset,left+inset:right-inset]=1
        if mask.sum()<8:raise ValueError("Text supervision collapsed")
        inputs=locator.processor(images=[marked,before],return_tensors="pt",device="cuda:0",
            limit_side_len=1024,limit_type="max",max_side_limit=1024).to("cuda:0")
        pixels=inputs["pixel_values"]
        with torch.inference_mode():target=teacher(pixel_values=pixels).last_hidden_state
        target=target.detach().clone()
        mask=torch.from_numpy(mask)[None,None].to("cuda:0")
        mask=F.interpolate(mask,size=target.shape[-2:],mode="nearest")
        result=(pixels.detach(),target,mask)
        cache[row["id"]]=tuple(t.cpu() for t in result)
        return cache[row["id"]]

    a.target.mkdir(parents=True)
    revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    config={"revision":revision,"manifest_sha256":manifest_sha,"base_weights_sha256":locator.weights_sha256,
            "arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
            "train_pairs":len(train),"validation_pairs":len(validation),
            "purpose":"Text-like candidate locations only; cannot classify digital overlays or approve output",
            "label":"Exact paired synthetic differences; parent-disjoint original photo splits",
            "parameters":sum(v.numel() for v in model.parameters())}
    (a.target/"config.json").write_text(json.dumps(config,indent=2))
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=.001)
    began=time.monotonic()
    def evaluate(step):
        reports=[]
        with torch.inference_mode():
            for pair in validation:
                pixels,target,mask=[x.to("cuda:0") for x in prepare(pair)]
                output=model(pixel_values=pixels).last_hidden_state
                reports.append({"id":pair[0]["id"],"alpha":pair[0]["render"]["alpha"],
                    "base_word_probability":float(target[:1][mask.bool()].mean()),
                    "word_probability":float(output[:1][mask.bool()].mean()),
                    "clean_map_mae":float((output[1:]-target[1:]).abs().mean()),
                    "new_clean_hot_fraction":float(((output[1:]>.3)&(target[1:]<=.3)).float().mean())})
        (a.target/f"validation-{step:04d}.json").write_text(json.dumps(reports,indent=2))
        print(json.dumps({"validation_step":step,"pairs":len(reports),
            "mean_word_probability":sum(r["word_probability"] for r in reports)/len(reports),
            "clean_map_mae":sum(r["clean_map_mae"] for r in reports)/len(reports)}),flush=True)
    evaluate(0);order=[]
    for step in range(1,a.steps+1):
        if not order:order=list(range(len(train)));rng.shuffle(order)
        pixels,target,mask=[x.to("cuda:0") for x in prepare(train[order.pop()])]
        output=model(pixel_values=pixels).last_hidden_state.clamp(1e-5,1-1e-5)
        positive=-(output[:1].log()*mask).sum()/mask.sum().clamp_min(1)
        outside=F.binary_cross_entropy(output[:1],target[:1],reduction="none")*(1-mask)
        preservation=outside.sum()/(1-mask).sum().clamp_min(1)+F.binary_cross_entropy(output[1:],target[1:])
        loss=positive+2*preservation
        if not torch.isfinite(loss):raise ValueError("Nonfinite detector training loss")
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
        for group in optimizer.param_groups:
            group["lr"]=a.learning_rate*(.1+.9*.5*(1+math.cos(math.pi*step/a.steps)))
        optimizer.step()
        if step%25==0:
            print(json.dumps({"step":step,"loss":float(loss.detach()),
                "seconds":time.monotonic()-began}),flush=True)
        if step%100==0 or step==a.steps:
            folder=a.target/f"detector-step{step:04d}"
            model.save_pretrained(folder,safe_serialization=True);locator.processor.save_pretrained(folder)
            (folder/"sensitivity-task.json").write_text(json.dumps(config|{"step":step,
                "weights_sha256":digest(folder/"model.safetensors")},indent=2))
            evaluate(step)
    assert manifest_sha==digest(a.manifest)
    assert revision==subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    assert reservation==offline_reservation(a.gpu)
    report={"state":"completed","steps":a.steps,"seconds":time.monotonic()-began,
            "peak_cuda_bytes":torch.cuda.max_memory_allocated(),"revision":revision,
            "manifest_sha256":manifest_sha,"production_adopted":False,
            "scope":"Candidate sensitivity only, not native quality qualification"}
    (a.target/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
