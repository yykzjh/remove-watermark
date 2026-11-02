"""Bounded single-GPU conditional-flow LoRA pilot; never changes deployment."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
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
    p.add_argument("--steps",type=int,default=200)
    p.add_argument("--rank",type=int,choices=[4,8,16],default=8)
    p.add_argument("--alpha",type=float,default=8.)
    p.add_argument("--lora-targets",choices=["attention","attention_mlp"],default="attention")
    p.add_argument("--learning-rate",type=float,default=5e-5)
    p.add_argument("--warmup",type=int,default=20)
    p.add_argument("--save-every",type=int,default=100)
    p.add_argument("--validate-every",type=int,default=50)
    p.add_argument("--validation-limit",type=int,default=16)
    p.add_argument("--seed",type=int,default=9417)
    p.add_argument("--gradient-checkpointing",action="store_true")
    p.add_argument("--source-flow-context",action="store_true")
    a=p.parse_args()
    if (a.gpu<0 or not 1<=a.steps<=2000 or not 0<=a.warmup<a.steps or
            not 1<=a.save_every<=a.steps or not 1<=a.validate_every<=a.steps or
            not 4<=a.validation_limit<=64 or not 0<a.learning_rate<=2e-4 or
            not math.isfinite(a.alpha) or not 0<a.alpha<=32):
        p.error("Explicit disjoint GPU and bounded pilot settings required")
    from watermark_studio.training_features import (
        digest,
        editing_kwargs,
        model_signature,
        validate_pairs,
    )
    cached=json.loads(a.cache.read_text())
    if cached.get("state")!="completed" or cached.get("version")!=1:
        p.error("Completed frozen feature cache required")
    dataset=Path(cached["dataset_manifest"]);validate_pairs(dataset)
    if digest(dataset)!=cached["dataset_manifest_sha256"]:
        p.error("Dataset manifest changed")
    if a.target.resolve().parent!=dataset.parent.resolve():
        p.error("Training folder must remain under pilot root")
    model_path=Path(cached["model"])
    signature=model_signature(model_path)
    if signature!=cached["model_signature"]:p.error("Feature cache backbone changed")
    rows=cached["rows"]
    if (not rows or len(rows)>640 or
            any(not Path(r["path"]).resolve().is_relative_to(a.cache.resolve().parent) or
                digest(r["path"])!=r["sha256"] for r in rows)):
        p.error("Frozen feature bytes changed or escaped cache root")
    train=[r for r in rows if r["split"]=="train"]
    validation=[r for r in rows if r["split"]=="validation"]
    if (not train or not validation or
            {r["source_original_sha256"] for r in train}&{r["source_original_sha256"] for r in validation}):
        p.error("Source-level disjoint train and validation features required")
    helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    def isolation():
        with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as response:
            return helper["require_disjoint_server_gpus"](json.load(response),[a.gpu],True)
    reserved=isolation()
    revision=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    a.target.mkdir(exist_ok=False);work=a.target.resolve()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
    import torch
    from diffusers import QwenImage21Transformer2DModel
    from safetensors.torch import load_file, save_file
    from torch.nn import functional

    from watermark_studio.attention import IndependentSDPAProcessor
    from watermark_studio.lora_training import (
        adapter_tensors,
        flow_objective,
        install_attention_lora,
    )
    torch.set_num_threads(1);torch.manual_seed(a.seed);random.seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    began=time.monotonic()
    model=QwenImage21Transformer2DModel.from_pretrained(model_path,subfolder="transformer",
        dtype=torch.bfloat16,local_files_only=True).to("cuda:0")
    for block in model.transformer_blocks:
        if hasattr(block.attn,"to_qkv"):raise ValueError("Separate unquantized QKV required")
        block.attn.set_processor(IndependentSDPAProcessor("native"))
    spec=install_attention_lora(model,a.rank,a.alpha,include_mlp=a.lora_targets=="attention_mlp")
    if a.gradient_checkpointing:model.enable_gradient_checkpointing()
    parameters=[v for v in model.parameters() if v.requires_grad]
    if any(v.dtype!=torch.float32 for v in parameters):
        raise ValueError("Trainable low-rank matrices must remain FP32")
    optimizer=torch.optim.AdamW(parameters,lr=a.learning_rate,betas=(.9,.999),weight_decay=.01)
    if {id(v) for group in optimizer.param_groups for v in group["params"]}!={id(v) for v in parameters}:
        raise ValueError("Optimizer escaped adapter-only training")
    load_seconds=time.monotonic()-began
    tensor_cache={}
    def tensors(row):
        if row["id"] not in tensor_cache:tensor_cache[row["id"]]=load_file(row["path"],device="cpu")
        return {k:v.to("cuda:0") for k,v in tensor_cache[row["id"]].items()}
    def loss(row,generator):
        values=tensors(row);clean=values["clean_latents"]
        noise=torch.randn(clean.shape,device="cuda:0",dtype=torch.float32,generator=generator).to(clean.dtype)
        choices=cached["sigma_tables"][row["shape"]]
        position=int(torch.randint(len(choices),(1,),device="cuda:0",generator=generator).item())
        sigma=float(choices[position])
        mixed=(1-sigma)*clean+sigma*noise
        if a.source_flow_context:
            mask=values["edit_mask"].reshape(1,1,row["height"]//16,row["width"]//16)
            mask=functional.max_pool2d((mask>0).float(),3,stride=1,padding=1)
            mask=mask.flatten(2).transpose(1,2).to(clean.dtype)
            reference=(1-sigma)*values["source_latents"]+sigma*noise
            mixed=mask*mixed+(1-mask)*reference
        hidden=torch.cat([values["source_latents"],mixed],dim=1)
        timestep=torch.tensor([sigma*1000],device="cuda:0",dtype=torch.bfloat16)/1000
        kwargs=editing_kwargs(values,row["height"],row["width"])
        with model.cache_context("cond"):
            prediction=model(hidden_states=hidden,timestep=timestep,**kwargs)[0][:,-clean.shape[1]:]
        return flow_objective(prediction,noise,clean,values["edit_mask"])
    import hashlib
    validation=sorted(validation,key=lambda r:hashlib.sha256(r["id"].encode()).hexdigest())[:a.validation_limit]
    def validate(step):
        model.eval()
        gen=torch.Generator("cuda:0").manual_seed(a.seed+8191)
        started=time.monotonic()
        with torch.no_grad():values=[float(loss(row,gen)) for row in validation]
        model.train()
        if any(not math.isfinite(value) for value in values):raise ValueError("Nonfinite validation loss")
        return {"stage":"validation","step":step,"mean_loss":sum(values)/len(values),
            "cases":len(values),"seconds":round(time.monotonic()-started,3),
            "not_image_quality_acceptance":True}
    def record(value):
        with (work/"metrics.jsonl").open("a") as stream:stream.write(json.dumps(value)+"\n")
        print(json.dumps(value),flush=True)
    provenance={"version":1,"state":"running","code_revision":revision,
        "cache_manifest":str(a.cache.resolve()),"cache_manifest_sha256":digest(a.cache),
        "dataset_manifest":str(dataset),"dataset_manifest_sha256":digest(dataset),
        "model":str(model_path),"model_signature":signature,"adapter":spec,
        "arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        "objective":"conditional_flow_noise_minus_clean; overlay_weight4; preservation_weight1",
        "training_generations":0,"source_overlap":0,"production_changed":False,
        "train_pairs":len(train),"validation_pairs":len([r for r in rows if r["split"]=="validation"]),
        "load_seconds":round(load_seconds,3)}
    (work/"report.json").write_text(json.dumps(provenance,indent=2))
    model.train();record(validate(0));began=time.monotonic()
    gen=torch.Generator("cuda:0").manual_seed(a.seed)
    order=list(train);random.shuffle(order);saved=[]
    for step in range(1,a.steps+1):
        if (step-1)%len(order)==0 and step>1:random.shuffle(order)
        row=order[(step-1)%len(order)]
        lr=a.learning_rate*min(1.,step/max(1,a.warmup))
        for group in optimizer.param_groups:group["lr"]=lr
        optimizer.zero_grad(set_to_none=True)
        started=time.monotonic();value=loss(row,gen)
        if not torch.isfinite(value):raise ValueError("Nonfinite conditional-flow loss")
        value.backward()
        if any(v.grad is not None for v in model.parameters() if not v.requires_grad):
            raise ValueError("Frozen backbone unexpectedly received gradients")
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        if not any(v.grad is not None for v in parameters):raise ValueError("Disconnected adapter gradients")
        optimizer.step()
        record({"stage":"training","step":step,"loss":float(value.detach()),"gradient_norm":float(norm),
            "learning_rate":lr,"pair":row["id"],"seconds":round(time.monotonic()-started,3),
            "peak_gpu_bytes":torch.cuda.max_memory_allocated()})
        if step%a.validate_every==0 or step==a.steps:record(validate(step))
        if step%a.save_every==0 or step==a.steps:
            path=work/f"adapter-step{step:04d}.safetensors"
            save_file(adapter_tensors(model),str(path),metadata={"alpha":str(a.alpha),"rank":str(a.rank),
                "model_index_sha256":signature["model_index.json"],
                "dataset_manifest_sha256":cached["dataset_manifest_sha256"],
                "code_revision":revision,"steps":str(step),"objective":"conditional_flow_matching"})
            saved.append({"path":str(path),"sha256":digest(path),"step":step,"alpha":a.alpha})
        if step%25==0 and (reserved!=isolation() or revision!=subprocess.check_output(
                ["git","rev-parse","--short","HEAD"],text=True).strip()):
            raise ValueError("Isolation or code changed during training")
    if signature!=model_signature(model_path) or digest(dataset)!=cached["dataset_manifest_sha256"]:
        raise ValueError("Backbone/dataset files changed")
    if reserved!=isolation():raise ValueError("GPU reservation changed")
    provenance.update(state="completed",seconds=round(time.monotonic()-began,3),checkpoints=saved,
        peak_gpu_bytes=torch.cuda.max_memory_allocated())
    (work/"report.json").write_text(json.dumps(provenance,indent=2))
    print(json.dumps({"state":"completed","seconds":provenance["seconds"],"checkpoints":saved}),flush=True)


if __name__=="__main__":
    main()
