"""Versioned frozen image-edit features for small conditional-flow LoRA experiments."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def validate_pairs(path):
    path=Path(path).resolve(strict=True);root=path.parent
    data=json.loads(path.read_text());rows=data.get("pairs",[])
    if data.get("version")!=1 or data.get("state")!="completed" or not rows or len(rows)>640:
        raise ValueError("Completed bounded independently screened pilot required")
    if len({r["id"] for r in rows})!=len(rows):
        raise ValueError("Unique pair identifiers required")
    if any(r.get("split") not in ("train","validation") for r in rows):
        raise ValueError("Explicit pilot split required")
    train={r["source_original_sha256"] for r in rows if r["split"]=="train"}
    validation={r["source_original_sha256"] for r in rows if r["split"]=="validation"}
    forbidden=set(json.loads((root/"forbidden-sha256.json").read_text()))
    if not train or not validation or train&validation or (train|validation)&forbidden:
        raise ValueError("Original-SHA training, validation and previous-exposure separation required")
    files={}
    for row in rows:
        if row["split"] not in ("train","validation"):
            raise ValueError("Explicit pilot split required")
        for key in ("input","target","mask"):
            file=Path(row[key]).resolve(strict=True)
            if not file.is_relative_to(root):
                raise ValueError("Derived pilot image escaped its own root")
            previous=files.setdefault(str(file),row[key+"_sha256"])
            if previous!=row[key+"_sha256"]:raise ValueError("Conflicting pilot file identity")
    with ThreadPoolExecutor(max_workers=4) as pool:
        values=list(pool.map(digest,files))
    if any(actual!=expected for actual,expected in zip(values,files.values())):
        raise ValueError("Pilot source/target/mask bytes changed")
    return data,files


def model_signature(model):
    model=Path(model).resolve(strict=True)
    paths=[model/"model_index.json"]
    for component in ("transformer","vae","text_encoder","processor","scheduler"):
        paths.extend(p for p in (model/component).iterdir()
            if p.is_file() and p.suffix in (".json",".jinja",".safetensors"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        values=list(pool.map(digest,paths))
    return {str(path.relative_to(model)):value for path,value in zip(paths,values)}


def editing_kwargs(tensors,height,width,vae_scale_factor=16):
    """Match the pinned Qwen21 production forward layout, source before target."""
    import torch
    n=tensors["clean_latents"].shape[1]
    if (tensors["source_latents"].shape!=tensors["clean_latents"].shape or
            n!=(height//vae_scale_factor)*(width//vae_scale_factor) or n%4 or
            tensors["image_pad_mask"].shape[1]!=tensors["prompt_embeds"].shape[1]+n//4):
        raise ValueError("Frozen full-photo Qwen21 layout mismatch")
    mask=tensors["prompt_attention_mask"]
    return {"encoder_hidden_states":tensors["prompt_embeds"],
        "encoder_hidden_states_mask":None if torch.all(mask) else mask,
        "img_shapes":[[(1,height//vae_scale_factor,width//vae_scale_factor)]*2],
        "img_mask":tensors["image_pad_mask"],"attention_kwargs":None,
        "kv_cache":None,"kv_cache_mode":None,"return_dict":False}
