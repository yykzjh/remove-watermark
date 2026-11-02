"""Bounded offline paired pixel foreground model; never changes native/production images."""
from __future__ import annotations

import argparse
import collections
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
from PIL import Image, ImageFilter

from watermark_studio.detail_foreground import STYLES, encoded_photo, foreground_challenge
from watermark_studio.detail_foreground_model import (
    DILATION,
    MODEL_TYPE,
    THRESHOLD,
    TILE,
    build_foreground_model,
)
from watermark_studio.images import load_photo
from watermark_studio.training_features import digest, validate_pairs


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--target",type=Path,required=True)
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--steps",type=int,default=1000)
    p.add_argument("--samples",type=int,choices=[256,512,1024],default=1024)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--learning-rate",type=float,default=.0002)
    p.add_argument("--seed",type=int,default=2026110212)
    p.add_argument("--font",type=Path,default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    p.add_argument("--tf32",action="store_true")
    a=p.parse_args()
    if (a.gpu<0 or not 100<=a.steps<=3000 or not 1<=a.batch_size<=32
            or not 1e-6<=a.learning_rate<=.001 or not a.font.is_file()):
        p.error("Bounded isolated training and exact font required")
    if a.target.resolve().parent!=a.manifest.resolve().parent:p.error("Stay inside the screened dataset")
    data,_=validate_pairs(a.manifest);manifest_sha=digest(a.manifest)
    helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as r:
        isolation=helper["require_disjoint_server_gpus"](json.load(r),[a.gpu],True)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
    import torch
    from safetensors.torch import save_file
    torch.set_num_threads(2);torch.manual_seed(a.seed);rng=random.Random(a.seed)
    torch.backends.cuda.matmul.allow_tf32=a.tf32;torch.backends.cudnn.allow_tf32=a.tf32
    groups=collections.defaultdict(list)
    for row in data["pairs"]:
        if row["split"]=="train":groups[row["source_original_sha256"]].append(row)
    selected=[min(groups[s],key=lambda r:("lower1024" not in r["id"],r["target_sha256"],r["id"]))
        for s in sorted(groups)]
    validation={r["source_original_sha256"] for r in data["pairs"] if r["split"]=="validation"}
    if not selected or set(groups)&validation:raise ValueError("Training original roles overlap validation")
    a.target.mkdir(exist_ok=False);revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    report={"state":"preparing","model_type":MODEL_TYPE,"manifest":str(a.manifest.resolve()),
        "manifest_sha256":manifest_sha,"font_sha256":digest(a.font),"code_revision":revision,
        "train_originals":len(selected),"validation_originals_excluded":len(validation),
        "inference_threshold":THRESHOLD,"inference_dilation":DILATION,"tile_size":TILE,
        "train_label_dilation":2,"input":"source_RGB_then_reference_RGB_uint8_to_minus1_plus1",
        "extra_generations":0,"quality_permission":False,"native_images_modified":0,
        "validation_policy":"No validation images used for training, checkpoint selection, or threshold selection",
        "arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        "source_isolation":isolation}
    (a.target/"report.json").write_text(json.dumps(report,indent=2))
    began=time.monotonic();photos={};examples=[];provenance=[]
    # Equal original rotation, then deterministic shuffle; truth controls training crops only.
    for i in range(a.samples):
        row=selected[i%len(selected)]
        if row["target"] not in photos:photos[row["target"]]=load_photo(row["target"])[0]
        clean=photos[row["target"]];seed=rng.randrange(2**32);style=rng.choice((*STYLES,"none","none"))
        source,_,foreground,meta=foreground_challenge(clean,a.font,style,seed,quality=rng.choice([0,95]))
        reference=clean.filter(ImageFilter.GaussianBlur(rng.choice([0,.35,.7,1.2,2.,3.])))
        values=np.asarray(reference).astype(np.float32)+np.array([rng.uniform(-2,2) for _ in range(3)])
        reference=encoded_photo(Image.fromarray(np.rint(np.clip(values,0,255)).astype(np.uint8)),rng.choice([0,95]))
        l,t,r,b=meta["body_box"]
        if r-l<TILE or b-t<TILE:raise ValueError("Selected native body cannot provide a whole classifier tile")
        points=np.argwhere(foreground[t:b,l:r])
        if len(points) and rng.random()<.5:
            y,x=points[rng.randrange(len(points))];x+=l;y+=t
            left=max(l,min(r-TILE,int(x)-TILE//2));top=max(t,min(b-TILE,int(y)-TILE//2))
        else:left=rng.randint(l,r-TILE);top=rng.randint(t,b-TILE)
        box=(left,top,left+TILE,top+TILE)
        mask=Image.fromarray(foreground.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(5)).crop(box)
        image=np.concatenate([np.asarray(im.crop(box)) for im in (source,reference)],axis=2).transpose(2,0,1).copy()
        label=(np.asarray(mask)>0).astype(np.uint8)[None].copy()
        examples.append((image,label));provenance.append({"source_original_sha256":row["source_original_sha256"],
            "target_sha256":row["target_sha256"],"seed":seed,"style":style,"encoding":meta["quality"],"box":box})
        if (i+1)%128==0:print(json.dumps({"stage":"preparing","samples":i+1,"total":a.samples}),flush=True)
    raw=json.dumps(provenance,sort_keys=True);(a.target/"training-samples.json").write_text(raw)
    report.update(state="training",preparation_seconds=round(time.monotonic()-began,3),
        samples_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        cached_numpy_bytes=sum(x.nbytes+y.nbytes for x,y in examples))
    (a.target/"report.json").write_text(json.dumps(report,indent=2))
    model=build_foreground_model().to("cuda:0").train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=.0001)
    positive=torch.tensor([20.],device="cuda:0").reshape(1,1,1)
    began=time.monotonic()
    for step in range(1,a.steps+1):
        batch=[]
        for _ in range(a.batch_size):
            image,label=examples[rng.randrange(len(examples))]
            rotation=rng.randrange(4)
            image=np.rot90(image,rotation,axes=(1,2));label=np.rot90(label,rotation,axes=(1,2))
            if rng.random()<.5:image=image[:,:,::-1];label=label[:,:,::-1]
            batch.append((image,label))
        x=torch.from_numpy(np.stack([v[0] for v in batch])).to("cuda:0",dtype=torch.float32).div(127.5).sub(1)
        y=torch.from_numpy(np.stack([v[1] for v in batch])).to("cuda:0",dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        logits=model(x);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,pos_weight=positive)
        if not torch.isfinite(loss):raise ValueError("Nonfinite foreground loss")
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if step%50==0:
            event={"stage":"training","step":step,"loss":float(loss.detach()),"seconds":round(time.monotonic()-began,3)}
            with (a.target/"metrics.jsonl").open("a") as f:f.write(json.dumps(event)+"\n")
            print(json.dumps(event),flush=True)
    path=a.target/f"foreground-step{a.steps:04d}.safetensors";tmp=path.with_suffix(".tmp")
    save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(tmp),
        metadata={"model_type":MODEL_TYPE,"inference_threshold":str(THRESHOLD),"tile_size":str(TILE)})
    tmp.replace(path)
    if digest(a.manifest)!=manifest_sha:raise ValueError("Training manifest changed")
    validate_pairs(a.manifest)
    report.update(state="completed",seconds=round(time.monotonic()-began,3),
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),checkpoint=str(path),checkpoint_sha256=digest(path),
        steps=a.steps,parameter_count=sum(p.numel() for p in model.parameters()))
    (a.target/"report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=="__main__":main()
