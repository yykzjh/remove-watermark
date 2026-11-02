"""Freeze prompt/whole-image VAE features, without diffusion generation."""
from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--target",type=Path,required=True)
    p.add_argument("--gpu",type=int,required=True)
    a=p.parse_args()
    if a.gpu<0:p.error("Explicit disjoint physical GPU required")
    from watermark_studio.training_features import digest, model_signature, validate_pairs
    data,input_files=validate_pairs(a.manifest)
    if a.target.resolve().parent!=a.manifest.resolve().parent:
        p.error("Feature folder must stay under its original pilot root")
    a.target.mkdir(exist_ok=False);work=a.target.resolve()
    helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as response:
        isolation=helper["require_disjoint_server_gpus"](json.load(response),[a.gpu],True)
    revision=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    manifest_sha=digest(a.manifest);signature=model_signature(a.model)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu);os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
    import torch
    from diffusers import QwenImage21Pipeline
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift
    from PIL import Image
    from safetensors.torch import save_file

    from watermark_studio.images import load_photo
    from watermark_studio.optimizations import lean_encode
    from watermark_studio.prompts import build_full_prompt
    torch.set_num_threads(1);began=time.monotonic()
    pipe=QwenImage21Pipeline.from_pretrained(a.model,transformer=None,dtype=torch.bfloat16,local_files_only=True)
    pipe.set_progress_bar_config(disable=True)
    pipe.text_encoder.eval().requires_grad_(False).to("cuda:0")
    pipe.vae.eval().requires_grad_(False).to("cuda:0")
    load_seconds=time.monotonic()-began;began=time.monotonic()
    rows=[];clean_cache={};sigma_tables={};prompt=build_full_prompt("","neutral")
    def encode(photo,width,height):
        value=pipe.image_processor.preprocess(photo.convert("RGBA"),width=width,height=height).unsqueeze(2)
        value=value.to("cuda:0",dtype=torch.bfloat16)
        encoded=pipe._encode_vae_image(value,torch.Generator("cpu").manual_seed(42))
        return pipe._pack_latents(encoded,1,pipe.latent_channels,encoded.shape[3],encoded.shape[4])
    for row in data["pairs"]:
        original=load_photo(row["input"])[0];target=load_photo(row["target"])[0]
        width,height=original.size
        if target.size!=original.size or width%32 or height%32:
            raise ValueError("Exact common 32-aligned full-frame pair required")
        with torch.inference_mode():
            resized=pipe.image_processor.resize(original.convert("RGBA"),width=width,height=height)
            embeds,attention,img_mask=lean_encode(pipe,[{"prompt":prompt,"image":resized}])[0]
            reference=encode(original,width,height)
            if row["target_sha256"] not in clean_cache:
                clean_cache[row["target_sha256"]]=encode(target,width,height).cpu()
            clean=clean_cache[row["target_sha256"]]
            if reference.shape!=clean.shape or int(img_mask.sum())!=reference.shape[1]//4:
                raise ValueError("Prompt vision tokens do not match complete reference latent grid")
            image_mask=torch.cat([img_mask,img_mask.new_ones((1,clean.shape[1]//4))],dim=1)
            with Image.open(row["mask"]) as mask:
                support=np.asarray(mask.convert("L").resize((width//16,height//16),Image.Resampling.BOX),dtype=np.float32)/255.
            support=torch.from_numpy(support.copy()).reshape(1,-1)
            tensors={"prompt_embeds":embeds.cpu().contiguous(),"prompt_attention_mask":attention.cpu().contiguous(),
                "image_pad_mask":image_mask.cpu().contiguous(),"source_latents":reference.cpu().contiguous(),
                "clean_latents":clean.contiguous(),"edit_mask":support.contiguous()}
            if any(t.is_floating_point() and not torch.isfinite(t).all() for t in tensors.values()):
                raise ValueError("Nonfinite frozen feature")
            if row["render"]["kind"]=="identity" and not torch.equal(reference.cpu(),clean):
                raise ValueError("Identity source and clean VAE latents differ")
        shape=f"{width}x{height}"
        if shape not in sigma_tables:
            mu=calculate_shift(clean.shape[1],pipe.scheduler.config.get("base_image_seq_len",256),
                pipe.scheduler.config.get("max_image_seq_len",4096),pipe.scheduler.config.get("base_shift",.5),
                pipe.scheduler.config.get("max_shift",1.15))
            pipe.scheduler.set_timesteps(1000,device="cpu",sigmas=np.linspace(1.,.001,1000),mu=mu)
            sigma_tables[shape]=pipe.scheduler.sigmas[:-1].float().tolist()
        destination=work/(row["id"]+".safetensors")
        save_file(tensors,str(destination),metadata={"source_original_sha256":row["source_original_sha256"],
            "dataset_manifest_sha256":manifest_sha})
        rows.append({"id":row["id"],"split":row["split"],"source_original_sha256":row["source_original_sha256"],
            "path":str(destination),"sha256":digest(destination),"width":width,"height":height,"shape":shape})
        print(json.dumps({"completed":len(rows),"total":len(data["pairs"])}),flush=True)
    if manifest_sha!=digest(a.manifest) or any(digest(path)!=sha for path,sha in input_files.items()):
        raise ValueError("Dataset changed during feature caching")
    if revision!=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip():
        raise ValueError("Code changed during feature caching")
    with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as response:
        if isolation!=helper["require_disjoint_server_gpus"](json.load(response),[a.gpu],True):
            raise ValueError("Server reservation changed")
    report={"state":"completed","version":1,"dataset_manifest":str(a.manifest.resolve()),
        "dataset_manifest_sha256":manifest_sha,"model":str(a.model.resolve()),"model_signature":signature,
        "code_revision":revision,"sigma_tables":sigma_tables,"rows":rows,
        "scope":"Frozen prompt/reference/clean features; zero diffusion image generations",
        "seconds":round(time.monotonic()-began,3),"load_seconds":round(load_seconds,3)}
    (work/"manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:report[k] for k in ("state","seconds","load_seconds")}))


if __name__=="__main__":
    main()
