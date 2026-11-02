"""Whole-frame SAM2 mask experiments; no image generation or edit permissions."""
from __future__ import annotations

import hashlib
import re

import numpy as np


def load_segmenter(checkpoint,sha256,device):
    import torch
    from transformers import Sam2ImageProcessor, Sam2Model, Sam2Processor

    from ._sam2_conversion import CONVERTER_SOURCE_SHA256, get_config, replace_keys
    if not re.fullmatch(r"[a-f0-9]{64}",sha256):
        raise ValueError("Pinned checkpoint SHA256 required")
    with checkpoint.open("rb") as stream:
        if hashlib.file_digest(stream,"sha256").hexdigest()!=sha256:
            raise ValueError("Segmentation checkpoint changed")
    raw=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if not isinstance(raw,dict) or not isinstance(raw.get("model"),dict):
        raise TypeError("Official SAM2 model state dictionary required")
    converted=replace_keys(raw["model"]);model=Sam2Model(get_config("sam2.1_hiera_small"))
    expected=set(model.state_dict());extra=set(converted)-expected;missing=expected-set(converted)
    allowed=getattr(model,"_keys_to_ignore_on_load_unexpected",[]) or []
    unknown=[key for key in extra if not any(re.search(pattern,key) for pattern in allowed)]
    if missing or unknown:
        raise ValueError(f"Unaccounted SAM2 weights: missing={sorted(missing)[:4]}, extra={unknown[:4]}")
    model.load_state_dict({k:v for k,v in converted.items() if k in expected},strict=True)
    model.eval().to(device)
    processor=Sam2Processor(image_processor=Sam2ImageProcessor())
    audit={"checkpoint_sha256":sha256,"converter_source_sha256":CONVERTER_SOURCE_SHA256,
        "loaded_tensors":len(expected),"declared_unused_video_tensors":sorted(extra),
        "processor":type(processor.image_processor).__name__,"quality_permission":False}
    return model,processor,audit


def prompt_boxes(box,size,jitter=0.):
    if not 0<=jitter<=.025:
        raise ValueError("Bounded whole-frame prompt jitter required")
    l,t,r,b=box;w,h=size
    if not 0<=l<r<=1 or not 0<=t<b<=1:
        raise ValueError("Valid full-photo normalized source box required")
    boxes=[]
    for delta in ([0.,-jitter,jitter] if jitter else [0.]):
        a=max(0,l-delta);c=min(1,r+delta);d=max(0,t-delta);e=min(1,b+delta)
        if not a<c or not d<e:
            raise ValueError("Jitter cannot collapse a prompt box")
        boxes.append([a*w,d*h,c*w,e*h])
    return boxes


def segment_whole_photo(photo,box,model,processor,device,jitter=0.):
    import torch
    inputs=processor(images=photo,input_boxes=[prompt_boxes(box,photo.size,jitter)],
        return_tensors="pt").to(device)
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        output=model(**inputs,multimask_output=True)
    masks=processor.post_process_masks(output.pred_masks.cpu(),inputs["original_sizes"].cpu())[0].numpy()
    scores=output.iou_scores[0].float().cpu().numpy()
    if (masks.ndim!=4 or masks.shape[:2]!=scores.shape or masks.shape[-2:]!=(photo.height,photo.width)
            or not np.isfinite(scores).all()):
        raise ValueError("Complete native full-photo segmentation masks required")
    best=np.stack([row[int(np.argmax(score))] for row,score in zip(masks,scores)]).astype(bool)
    return {"best":best[0],"union_best":best.any(axis=0),
        "union_all":masks.astype(bool).any(axis=(0,1))},{"scores":scores.tolist(),
        "whole_frame":True,"source_size":list(photo.size),"prompt_boxes":prompt_boxes(box,photo.size,jitter),
        "extra_image_generations":0,"quality_permission":False}


def mask_truth(mask,truth):
    if mask.dtype!=bool or truth.dtype!=bool or mask.shape!=truth.shape or mask.ndim!=2:
        raise ValueError("Aligned native Boolean mask and exact source-change truth required")
    required=int(truth.sum());covered=int((truth&mask).sum());pixels=int(mask.sum())
    if not required:
        raise ValueError("Nonempty known digital overlay truth required")
    return {"required_pixels":required,"covered_pixels":covered,"missed_pixels":required-covered,
        "full_coverage":covered==required,"coverage":round(covered/required,8),
        "mask_pixels":pixels,"outside_pixels":pixels-covered,
        "iou":round(covered/(required+pixels-covered),8)}
