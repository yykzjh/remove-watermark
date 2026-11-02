"""One whole-frame generation per synthetic validation pair, with exact clean targets."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import runpy
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--target",type=Path,required=True)
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--limit",type=int,default=12)
    p.add_argument("--uniform-steps",type=int,choices=[0,12,16,20,24,40],default=0)
    p.add_argument("--lora-path",type=Path)
    p.add_argument("--lora-sha256",default="")
    p.add_argument("--lora-alpha",type=float,default=8.)
    p.add_argument("--lora-scale",type=float,default=1.)
    a=p.parse_args()
    if a.gpu<0 or not 4<=a.limit<=64:p.error("Disjoint explicit GPU and bounded validation subset required")
    if a.lora_path and not re.fullmatch(r"[a-f0-9]{64}",a.lora_sha256):
        p.error("Explicit exact adapter SHA required")
    from watermark_studio.training_features import digest, validate_pairs
    data,_=validate_pairs(a.manifest)
    # Stable source-diverse ordering; no selection on model output or verifier score.
    rows=[row for row in data["pairs"] if row["split"]=="validation"]
    groups={}
    for row in rows:groups.setdefault(row["source_original_sha256"],[]).append(row)
    ordered=[]
    for source_index,(_,group) in enumerate(sorted(groups.items())):
        group.sort(key=lambda row:int(row["id"].rsplit("-",1)[-1]))
        offset=source_index%len(group)
        ordered.append(group[offset:]+group[:offset])
    rows=[group[round_index] for round_index in range(4) for group in ordered
        if round_index<len(group)][:a.limit]
    if not rows:p.error("Held-out synthetic validation pairs required")
    if a.target.resolve().parent!=a.manifest.resolve().parent:p.error("Stay within pilot root")
    a.target.mkdir(exist_ok=False);work=a.target.resolve();(work/"output").mkdir()
    helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    def isolation():
        with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as response:
            return helper["require_disjoint_server_gpus"](json.load(response),[a.gpu],True)
    reserved=isolation()
    revision=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    manifest_sha=digest(a.manifest)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu);os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
    import torch

    from watermark_studio.batching import independent_batch
    from watermark_studio.config import DeploymentConfig
    from watermark_studio.images import load_photo
    from watermark_studio.probe_schedules import configure_turbo_uniform_probe
    from watermark_studio.prompts import build_full_prompt
    from watermark_studio.worker import load_pipeline
    torch.set_num_threads(1)
    cfg=DeploymentConfig(model=str(a.model),gpus=[a.gpu],quantization="float8",float8_granularity="tensor",
        prompt_encoder="lean",compile=True,compile_scope="projections",compile_dynamic=True,separate_vae_decode=True,
        lora_path=str(a.lora_path) if a.lora_path else "",lora_sha256=a.lora_sha256,
        lora_alpha=a.lora_alpha,lora_scale=a.lora_scale)
    began=time.monotonic();pipe=load_pipeline(cfg,str(a.model))
    uniform=configure_turbo_uniform_probe(pipe,a.uniform_steps) if a.uniform_steps else None
    load_seconds=time.monotonic()-began;began=time.monotonic();results=[]
    def metrics(actual,expected,mask):
        difference=np.mean((actual.astype(np.float32)-expected.astype(np.float32))**2,axis=2)
        selected=difference[mask]
        mse=float(np.mean(selected)) if selected.size else None
        return {"mse":mse,"psnr":(-10*math.log10(max(mse,1e-8)/255**2)) if mse is not None else None,
            "pixels":int(selected.size)}
    for row in rows:
        source=load_photo(row["input"])[0];clean=load_photo(row["target"])[0]
        seed=(row["seed"]+1000003)%(2**32)
        item={"image":source,"prompt":build_full_prompt("","neutral"),"seed":seed}
        if a.uniform_steps:item["_probe_num_inference_steps"]=a.uniform_steps
        started=time.monotonic()
        output=independent_batch(pipe,[item],512,True)[0]
        seconds=time.monotonic()-started
        if output.size!=source.size:raise ValueError("Pilot generation changed complete-frame shape")
        if a.uniform_steps and len(pipe.scheduler.timesteps)!=a.uniform_steps:
            raise ValueError("Requested step count not honored")
        with Image.open(row["mask"]) as image:mask=np.asarray(image.convert("L"))>0
        actual=np.asarray(output.convert("RGB"));expected=np.asarray(clean.convert("RGB"))
        before=np.asarray(source.convert("RGB"))
        record={"id":row["id"],"source_original_sha256":row["source_original_sha256"],
            "kind":row["render"]["kind"],"generation_count":1,"seed":seed,"steps":len(pipe.scheduler.timesteps),
            "seconds":round(seconds,3),"image_size":list(output.size),
            "before_overlay":metrics(before,expected,mask),"after_overlay":metrics(actual,expected,mask),
            "unchanged_content":metrics(actual,expected,~mask),"whole_frame":metrics(actual,expected,np.ones_like(mask)),
            "mask_used_as_model_input":False,"quality_accepted":None}
        destination=work/"output"/(row["id"]+".png");output.save(destination)
        record.update(output=str(destination),output_sha256=digest(destination))
        results.append(record)
        with (work/"results.jsonl").open("a") as stream:stream.write(json.dumps(record)+"\n")
        print(json.dumps({"completed":len(results),"total":len(rows),"seconds":record["seconds"]}),flush=True)
    if manifest_sha!=digest(a.manifest) or revision!=subprocess.check_output(
            ["git","rev-parse","--short","HEAD"],text=True).strip() or reserved!=isolation():
        raise ValueError("Frozen dataset/code/isolation changed")
    report={"state":"completed","code_revision":revision,"dataset_manifest_sha256":manifest_sha,
        "total":len(results),"seconds":round(time.monotonic()-began,3),"load_seconds":round(load_seconds,3),
        "uniform_turbo":uniform,"image_adapter":getattr(pipe,"_image_adapter",None),
        "scope":"Synthetic held-out pilot; exact restoration metrics are not native dataset acceptance"}
    (work/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
