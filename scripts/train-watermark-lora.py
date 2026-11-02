"""Bounded single-GPU flow or native-terminal LoRA pilot; never changes deployment."""
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
    p.add_argument("--trajectory-cache",type=Path)
    p.add_argument("--initial-lora-path",type=Path)
    p.add_argument("--initial-lora-sha256",default="")
    p.add_argument("--prefix-weight",type=float,default=1.)
    p.add_argument("--pixel-supervision",action="store_true")
    p.add_argument("--pixel-weight",type=float,default=1.)
    a=p.parse_args()
    if (a.pixel_supervision and not a.trajectory_cache) or (a.pixel_weight!=1 and not a.pixel_supervision):
        p.error("Pixel supervision requires a frozen terminal trajectory; weight requires explicit supervision")
    if not math.isfinite(a.pixel_weight) or not .1<=a.pixel_weight<=4:
        p.error("Bounded positive pixel supervision weight required")
    if bool(a.initial_lora_path)!=bool(a.initial_lora_sha256):
        p.error("Both pinned initial adapter path and SHA required")
    if a.prefix_weight!=1 and not a.trajectory_cache:
        p.error("Prefix preservation weight requires an explicit trajectory cache")
    if (a.gpu<0 or not 1<=a.steps<=2000 or not 0<=a.warmup<a.steps or
            not 1<=a.save_every<=a.steps or not 1<=a.validate_every<=a.steps or
            not 4<=a.validation_limit<=64 or not 0<a.learning_rate<=2e-4 or
            not math.isfinite(a.alpha) or not 0<a.alpha<=32 or
            not math.isfinite(a.prefix_weight) or not .1<=a.prefix_weight<=4):
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
    dataset=Path(cached["dataset_manifest"]);pair_data,_=validate_pairs(dataset)
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
    trajectory=None;trajectory_rows={}
    if a.trajectory_cache:
        trajectory=json.loads(a.trajectory_cache.read_text())
        if (trajectory.get("state")!="completed" or trajectory.get("version")!=1
                or trajectory.get("feature_cache_sha256")!=digest(a.cache)
                or trajectory.get("model_signature")!=signature
                or trajectory.get("dataset_manifest_sha256")!=cached["dataset_manifest_sha256"]
                or trajectory.get("arguments",{}).get("source_flow_context")!=a.source_flow_context):
            p.error("Completed matching trajectory/cache/model/context required")
        trajectory_rows={r["id"]:r for r in trajectory["rows"]}
        if (set(trajectory_rows)!={r["id"] for r in rows}
                or len(trajectory_rows)!=len(trajectory["rows"])
                or any(trajectory_rows[r["id"]]["feature_sha256"]!=r["sha256"] for r in rows)
                or any(not Path(r["path"]).resolve().is_relative_to(a.trajectory_cache.resolve().parent)
                    or digest(r["path"])!=r["sha256"] for r in trajectory_rows.values())):
            p.error("Exact one-to-one immutable trajectory features required")
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
        initialize_adapter,
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
    initialization=(initialize_adapter(model,a.initial_lora_path,a.initial_lora_sha256,
        signature["model_index.json"]) if a.initial_lora_path else None)
    if a.gradient_checkpointing:model.enable_gradient_checkpointing()
    parameters=[v for v in model.parameters() if v.requires_grad]
    if any(v.dtype!=torch.float32 for v in parameters):
        raise ValueError("Trainable low-rank matrices must remain FP32")
    optimizer=torch.optim.AdamW(parameters,lr=a.learning_rate,betas=(.9,.999),weight_decay=.01)
    if {id(v) for group in optimizer.param_groups for v in group["params"]}!={id(v) for v in parameters}:
        raise ValueError("Optimizer escaped adapter-only training")
    vae=None;pixel_cache={}
    pair_rows={r["id"]:r for r in pair_data["pairs"]}
    if a.pixel_supervision:
        from diffusers import AutoencoderKLQwenImage21
        vae=AutoencoderKLQwenImage21.from_pretrained(model_path,subfolder="vae",
            dtype=torch.bfloat16,local_files_only=True).eval().requires_grad_(False).to("cuda:0")
    load_seconds=time.monotonic()-began
    tensor_cache={};trajectory_tensors={};loss_components={}
    def pixels(row):
        if row["id"] not in pixel_cache:
            import numpy as np
            from PIL import Image

            from watermark_studio.images import load_photo
            original=pair_rows[row["id"]]
            if (digest(original["target"])!=original["target_sha256"]
                    or digest(original["mask"])!=original["mask_sha256"]):
                raise ValueError("Exact pixel supervision files changed")
            target=load_photo(original["target"])[0].convert("RGB")
            with Image.open(original["mask"]) as support:
                support=np.asarray(support.convert("L"),dtype=np.float32).copy()/255
            if target.size!=(row["width"],row["height"]) or support.shape!=(row["height"],row["width"]):
                raise ValueError("Exact pixel/latent grid mismatch")
            rgb=torch.from_numpy(np.asarray(target,dtype=np.float32).copy()/127.5-1).permute(2,0,1).unsqueeze(0)
            pixel_cache[row["id"]]=(rgb,torch.from_numpy(support).unsqueeze(0).unsqueeze(0))
        return tuple(v.to("cuda:0") for v in pixel_cache[row["id"]])

    def trajectory_loss(row,generator):
        from watermark_studio.trajectory_training import prefix_objective, terminal_objective
        values=tensors(row);clean=values["clean_latents"]
        if row["id"] not in trajectory_tensors:
            trajectory_tensors[row["id"]]=load_file(trajectory_rows[row["id"]]["path"],device="cpu")
        trace=trajectory_tensors[row["id"]]
        states=trace["states"];velocities=trace["velocities"];sigmas=trace["sigmas"]
        if (states.shape[0]<2 or states.shape[1:]!=clean.shape or velocities.shape!=states.shape
                or sigmas.shape!=(states.shape[0]+1,) or float(sigmas[-1])!=0):
            raise ValueError("Complete aligned frozen sampling trajectory required")
        kwargs=editing_kwargs(values,row["height"],row["width"])
        def predict(index):
            index=index%states.shape[0]
            mixed=states[index].to("cuda:0")
            time_value=float(sigmas[index])
            with model.cache_context("cond"):
                return model(hidden_states=torch.cat([values["source_latents"],mixed],dim=1),
                    timestep=torch.tensor([time_value*1000],device="cuda:0",dtype=torch.bfloat16)/1000,
                    **kwargs)[0][:,-clean.shape[1]:]
        terminal_prediction=predict(-1)
        terminal=terminal_objective(terminal_prediction,states[-1].to("cuda:0"),clean,
            float(sigmas[-2]),values["edit_mask"])
        pixel_loss=None
        if vae is not None:
            from watermark_studio.pixel_training import decode_exact_pixels, exact_pixel_objective
            from watermark_studio.trajectory_training import context_support
            endpoint=(states[-1].to("cuda:0").float()-float(sigmas[-2])*terminal_prediction.float()).to(clean.dtype)
            if a.source_flow_context:
                support=context_support(values,row["height"],row["width"],
                    trajectory["arguments"]["context_margin_cells"])
                endpoint=support*endpoint+(1-support)*values["source_latents"]
            target,mask=pixels(row)
            decoded=decode_exact_pixels(vae,endpoint,row["height"],row["width"],
                checkpoint=a.gradient_checkpointing and torch.is_grad_enabled())
            pixel_loss=exact_pixel_objective(decoded,target,mask)
        prefix=int(torch.randint(states.shape[0]-1,(1,),device="cuda:0",generator=generator).item())
        preservation=prefix_objective(predict(prefix),velocities[prefix].to("cuda:0"))
        loss_components.update(terminal_loss=float(terminal.detach()),prefix_loss=float(preservation.detach()))
        if pixel_loss is not None:loss_components["pixel_loss"]=float(pixel_loss.detach())
        return terminal+a.prefix_weight*preservation+(a.pixel_weight*pixel_loss if pixel_loss is not None else 0)

    def tensors(row):
        if row["id"] not in tensor_cache:tensor_cache[row["id"]]=load_file(row["path"],device="cpu")
        return {k:v.to("cuda:0") for k,v in tensor_cache[row["id"]].items()}
    def loss(row,generator):
        if trajectory is not None:return trajectory_loss(row,generator)
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
        values=[];parts=[]
        with torch.no_grad():
            for row in validation:
                values.append(float(loss(row,gen)));parts.append(dict(loss_components))
        model.train()
        if any(not math.isfinite(value) for value in values):raise ValueError("Nonfinite validation loss")
        result={"stage":"validation","step":step,"mean_loss":sum(values)/len(values),
            "cases":len(values),"seconds":round(time.monotonic()-started,3),
            "not_image_quality_acceptance":True}
        if trajectory:
            result.update({key:sum(part[key] for part in parts)/len(parts)
                for key in loss_components})
        return result
    def record(value):
        with (work/"metrics.jsonl").open("a") as stream:stream.write(json.dumps(value)+"\n")
        print(json.dumps(value),flush=True)
    provenance={"version":1,"state":"running","code_revision":revision,
        "cache_manifest":str(a.cache.resolve()),"cache_manifest_sha256":digest(a.cache),
        "dataset_manifest":str(dataset),"dataset_manifest_sha256":digest(dataset),
        "model":str(model_path),"model_signature":signature,"adapter":spec,
        "arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        "objective":("native_terminal_exact_clean_with_frozen_prefix_preservation" if trajectory else
            "conditional_flow_noise_minus_clean; overlay_weight4; preservation_weight1"),
        "trajectory_cache_sha256":digest(a.trajectory_cache) if trajectory else None,
        "initial_adapter":initialization,
        "pixel_supervision":"exact_RGB_L1_plus_0.25_local_texture_amplitude" if a.pixel_supervision else None,
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
        if vae is not None and any(v.requires_grad or v.grad is not None for v in vae.parameters()):
            raise ValueError("Frozen decoder must not receive parameter gradients")
        if any(v.grad is not None for v in model.parameters() if not v.requires_grad):
            raise ValueError("Frozen backbone unexpectedly received gradients")
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        if not any(v.grad is not None for v in parameters):raise ValueError("Disconnected adapter gradients")
        optimizer.step()
        record({"stage":"training","step":step,"loss":float(value.detach()),"gradient_norm":float(norm),
            "learning_rate":lr,"pair":row["id"],**loss_components,"seconds":round(time.monotonic()-started,3),
            "peak_gpu_bytes":torch.cuda.max_memory_allocated()})
        if step%a.validate_every==0 or step==a.steps:record(validate(step))
        if step%a.save_every==0 or step==a.steps:
            path=work/f"adapter-step{step:04d}.safetensors"
            save_file(adapter_tensors(model),str(path),metadata={"alpha":str(a.alpha),"rank":str(a.rank),
                "model_index_sha256":signature["model_index.json"],
                "dataset_manifest_sha256":cached["dataset_manifest_sha256"],
                "code_revision":revision,"steps":str(step),"objective":provenance["objective"]})
            saved.append({"path":str(path),"sha256":digest(path),"step":step,"alpha":a.alpha})
        if step%25==0 and (reserved!=isolation() or revision!=subprocess.check_output(
                ["git","rev-parse","--short","HEAD"],text=True).strip()):
            raise ValueError("Isolation or code changed during training")
    if signature!=model_signature(model_path) or digest(dataset)!=cached["dataset_manifest_sha256"]:
        raise ValueError("Backbone/dataset files changed")
    if trajectory and digest(a.trajectory_cache)!=provenance["trajectory_cache_sha256"]:
        raise ValueError("Trajectory manifest changed")
    if reserved!=isolation():raise ValueError("GPU reservation changed")
    provenance.update(state="completed",seconds=round(time.monotonic()-began,3),checkpoints=saved,
        peak_gpu_bytes=torch.cuda.max_memory_allocated())
    (work/"report.json").write_text(json.dumps(provenance,indent=2))
    print(json.dumps({"state":"completed","seconds":provenance["seconds"],"checkpoints":saved}),flush=True)


if __name__=="__main__":
    main()
