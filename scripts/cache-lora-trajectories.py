"""Freeze bounded synthetic-pair sampling trajectories; never inspect the blind set."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import subprocess
import time
import urllib.request
from pathlib import Path


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--target",type=Path,required=True)
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--seed",type=int,default=20261102)
    p.add_argument("--batch-size",type=int,choices=[1,2,4,8],default=1)
    p.add_argument("--kv-cache",action="store_true")
    p.add_argument("--uniform-steps",type=int,choices=[12,16,20,24,40])
    p.add_argument("--source-flow-context",action="store_true")
    p.add_argument("--context-margin-cells",type=int,choices=[0,1,2,3,4,6,8,12,16],default=1)
    a=p.parse_args()
    if a.gpu<0 or (a.context_margin_cells!=1 and not a.source_flow_context):
        p.error("Explicit disjoint GPU and context margin scope required")
    from watermark_studio.training_features import (
        digest,
        editing_kwargs,
        model_signature,
        validate_pairs,
    )
    cached=json.loads(a.cache.read_text())
    if cached.get("state")!="completed" or cached.get("version")!=1:
        p.error("Completed independently screened feature cache required")
    dataset=Path(cached["dataset_manifest"]);validate_pairs(dataset)
    if digest(dataset)!=cached["dataset_manifest_sha256"]:
        p.error("Dataset changed")
    if a.target.resolve().parent!=dataset.resolve().parent:
        p.error("Trajectory folder must remain under the pilot root")
    rows=cached["rows"]
    if (not rows or len(rows)>640 or any(
            not Path(r["path"]).resolve().is_relative_to(a.cache.resolve().parent)
            or digest(r["path"])!=r["sha256"] for r in rows)):
        p.error("Frozen feature bytes changed or escaped root")
    signature=model_signature(cached["model"])
    if signature!=cached["model_signature"]:p.error("Backbone changed")
    helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    def isolation():
        with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as response:
            return helper["require_disjoint_server_gpus"](json.load(response),[a.gpu],True)
    reserved=isolation()
    revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    a.target.mkdir(exist_ok=False);work=a.target.resolve()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu);os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
    import numpy as np
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Transformer2DModel
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    from watermark_studio.attention import IndependentSDPAProcessor
    from watermark_studio.probe_schedules import tail_sigma_grid
    from watermark_studio.trajectory_training import context_support, cpu_packed_noise
    torch.set_num_threads(1);began=time.monotonic()
    model=QwenImage21Transformer2DModel.from_pretrained(cached["model"],subfolder="transformer",
        dtype=torch.bfloat16,local_files_only=True).eval().requires_grad_(False).to("cuda:0")
    for block in model.transformer_blocks:block.attn.set_processor(IndependentSDPAProcessor("native"))
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(cached["model"],
        subfolder="scheduler",local_files_only=True)
    if scheduler.config.use_dynamic_shifting or scheduler.config.shift!=1 or scheduler.config.stochastic_sampling:
        raise ValueError("This bounded cache requires the verified deterministic unshifted Qwen21 schedule")
    index=json.loads((Path(cached["model"])/"model_index.json").read_text())
    sigmas=tail_sigma_grid(index["sample_sigmas"],"native") if a.uniform_steps is None else np.linspace(
        1.,1./a.uniform_steps,a.uniform_steps).tolist()
    load_seconds=time.monotonic()-began;began=time.monotonic();results=[]
    report={"version":1,"state":"running","code_revision":revision,
        "feature_cache":str(a.cache.resolve()),"feature_cache_sha256":digest(a.cache),
        "dataset_manifest_sha256":cached["dataset_manifest_sha256"],"model_signature":signature,
        "arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        "scope":"Offline synthetic training trajectories; exact clean targets remain real original crops",
        "quality_permission":False,"load_seconds":round(load_seconds,3),"rows":results}
    (work/"manifest.json").write_text(json.dumps(report,indent=2))
    buckets={}
    for ordinal,row in enumerate(rows):
        with safe_open(row["path"],framework="pt",device="cpu") as frozen:
            key=(row["height"],row["width"],tuple(frozen.get_slice("prompt_embeds").get_shape()),
                hashlib.sha256(frozen.get_tensor("image_pad_mask").numpy().tobytes()).hexdigest())
        buckets.setdefault(key,[]).append((ordinal,row))
    batches=[group[offset:offset+a.batch_size] for group in buckets.values()
        for offset in range(0,len(group),a.batch_size)]
    with torch.inference_mode():
        for batch_index,group in enumerate(batches):
            row=group[0][1]
            frozen=[load_file(r["path"]) for _,r in group]
            values={key:torch.cat([v[key] for v in frozen],0).to("cuda:0") for key in frozen[0]}
            clean=values["clean_latents"]
            seeds=[a.seed+ordinal for ordinal,_ in group]
            noise=torch.cat([cpu_packed_noise(row["height"],row["width"],clean.shape[-1],seed,clean.dtype)
                for seed in seeds],0).to("cuda:0")
            support=context_support(values,row["height"],row["width"],a.context_margin_cells)
            scheduler.set_timesteps(sigmas=sigmas,device="cuda:0")
            runtime=scheduler.sigmas.float().cpu().tolist()
            if runtime[-1]!=0 or len(runtime)!=len(sigmas)+1:raise ValueError("Native schedule changed")
            def blend(current,sigma,values=values,noise=noise,support=support):
                if not a.source_flow_context:return current
                reference=(1-sigma)*values["source_latents"]+sigma*noise
                return support*current+(1-support)*reference
            current=blend(noise.clone(),runtime[0]);states=[];velocities=[]
            kwargs=editing_kwargs(values,row["height"],row["width"])
            kwargs["kv_cache"]=QwenImage21KVCache(len(model.transformer_blocks)) if a.kv_cache else None
            for i,timestep in enumerate(scheduler.timesteps):
                states.append(current.cpu().contiguous())
                kwargs["kv_cache_mode"]=("extract" if i==0 else "cached") if a.kv_cache else None
                with model.cache_context("cond"):
                    prediction=model(hidden_states=torch.cat([values["source_latents"],current],1),
                        timestep=timestep.expand(len(group)).to(torch.bfloat16)/1000,
                        **kwargs)[0][:,-clean.shape[1]:]
                velocities.append(prediction.cpu().contiguous())
                current=scheduler.step(prediction,timestep,current,return_dict=False)[0]
                current=blend(current,runtime[i+1])
            states=torch.stack(states);velocities=torch.stack(velocities)
            for index,((_,original),seed) in enumerate(zip(group,seeds)):
                path=work/(original["id"]+".safetensors")
                save_file({"states":states[:,index:index+1].contiguous(),
                    "velocities":velocities[:,index:index+1].contiguous(),
                    "sigmas":torch.tensor(runtime,dtype=torch.float32),
                    "noise":noise[index:index+1].cpu().contiguous(),
                    "final_latents":current[index:index+1].cpu().contiguous()},str(path),
                    metadata={"feature_sha256":original["sha256"],"seed":str(seed)})
                results.append({"id":original["id"],"path":str(path),"sha256":digest(path),
                    "feature_sha256":original["sha256"],"seed":seed,"steps":len(sigmas),
                    "teacher_batch_size":len(group)})
            print(json.dumps({"completed":len(results),"total":len(rows),
                "seconds":round(time.monotonic()-began,3)}),flush=True)
            if batch_index%20==0 and (isolation()!=reserved or revision!=subprocess.check_output(
                    ["git","rev-parse","HEAD"],text=True).strip()):
                raise ValueError("Code or GPU reservation changed")
    if signature!=model_signature(cached["model"]) or digest(a.cache)!=report["feature_cache_sha256"]:
        raise ValueError("Frozen model/features changed")
    report.update(state="completed",seconds=round(time.monotonic()-began,3),rows=results,
        peak_gpu_bytes=torch.cuda.max_memory_allocated())
    (work/"manifest.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ("state","seconds","peak_gpu_bytes")}),flush=True)


if __name__=="__main__":main()
