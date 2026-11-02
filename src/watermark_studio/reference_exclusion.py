"""Offline reference-key ablations; complete source pixels always enter the image encoders."""
from __future__ import annotations

import inspect
import math
from contextlib import contextmanager

from .batching import override_methods


def target_key_keep(transformer,items,img_mask,img_shapes,device,margin_cells=0):
    """Use the pinned transformer's exact image-token metadata, never guessed joint offsets."""
    import torch
    from diffusers.models.transformers.transformer_qwenimage21 import _IMG_TOKENS_PER_SLOT

    from .denoising_context import pixel_edit_masks

    if type(margin_cells) is not int or not 0<=margin_cells<=8:
        raise ValueError("Bounded integer reference-key margin required")
    batch=len(items)
    if (img_mask.ndim!=2 or img_mask.shape[0]!=batch or len(img_shapes)!=batch
            or not torch.equal(img_mask,img_mask[:1].expand_as(img_mask))):
        raise ValueError("One common image-token layout for an independent batch required")
    shapes=img_shapes[0]
    if (len(shapes)!=2 or any(list(row)!=list(shapes) for row in img_shapes)
            or shapes[0]!=shapes[1] or len(shapes[0])!=3 or shapes[0][0]!=1):
        raise ValueError("Exactly one matching complete source and target frame required")
    _,height,width=shapes[0]
    if not all(type(v) is int and v>0 for v in (height,width)):
        raise ValueError("Concrete spatial image token dimensions required")
    # The original pixel image is never rewritten or cropped.
    masks=pixel_edit_masks([{"_denoising_context_regions":item["_reference_exclusion_regions"]}
        for item in items],(width,height))
    excluded=torch.from_numpy(masks).to(device=device,dtype=torch.float32).unsqueeze(1)
    if margin_cells:
        excluded=torch.nn.functional.max_pool2d(excluded,2*margin_cells+1,
            stride=1,padding=margin_cells)
    excluded=excluded.bool().flatten(1)
    if any(not row.any().item() or row.all().item() for row in excluded):
        raise ValueError("A nonempty, partial source reference exclusion is required")
    repeats=torch.where(img_mask,_IMG_TOKENS_PER_SLOT,1)[0]
    image_pad_mask=torch.repeat_interleave(img_mask[0],repeats)
    image_ids,target_mask=transformer.build_token_metadata(image_pad_mask,shapes)
    source_positions=(image_ids==0).nonzero(as_tuple=True)[0]
    if source_positions.numel()!=width*height or target_mask.sum().item()!=math.prod(shapes[-1]):
        raise ValueError("Source or target spatial metadata differs")
    keep=torch.ones(batch,image_pad_mask.numel(),device=device,dtype=torch.bool)
    keep[:,source_positions]=~excluded
    if not keep[:,target_mask].all().item() or not keep[:,image_ids<0].all().item():
        raise ValueError("Text and target keys must stay unchanged")
    return keep,math.prod(shapes[-1]),excluded.float().mean(1).cpu().tolist()


@contextmanager
def reference_exclusion(pipe,items,*,enabled=False,exclude_prefix=False,margin_cells=0,
                        compact_cache=False,max_sigma=1.):
    """Apply only to an explicit offline probe and restore every processor/method on exit."""
    audit={"enabled":enabled,"whole_source_pixels_unchanged":True,"prefix_computation_unchanged":not exclude_prefix,
        "extra_model_forwards":0,"forward_calls":0,"quality_permission":False,
        "reference_margin_cells":margin_cells,"compact_reference_cache":compact_cache,
        "reference_max_sigma":max_sigma,"active_forward_calls":0}
    if type(margin_cells) is not int or not 0<=margin_cells<=8:
        raise ValueError("Bounded integer reference-key margin required")
    if not isinstance(max_sigma,(float,int)) or not math.isfinite(max_sigma) or not 0<max_sigma<=1:
        raise ValueError("Reference schedule cutoff must be finite in0..1")
    if max_sigma<1 and (not enabled or exclude_prefix or compact_cache):
        raise ValueError("Scheduled exclusion requires unchanged full prefix computation and cache")
    if (exclude_prefix or margin_cells or compact_cache) and not enabled:
        raise ValueError("Prefix exclusion and margins require explicit reference exclusion")
    if not enabled:
        yield audit
        return
    import torch

    from .attention import IndependentSDPAProcessor

    if not items or any("_reference_exclusion_regions" not in item for item in items):
        raise ValueError("Explicit per-image source regions are required for the offline probe")
    transformer=pipe.transformer
    processors=list({id(block.attn.processor):block.attn.processor
        for block in transformer.transformer_blocks}.values())
    if not processors or any(not isinstance(proc,IndependentSDPAProcessor) for proc in processors):
        raise ValueError("Only the verified independent SDPA processor supports this probe")
    names=("_watermark_target_key_keep","_watermark_target_query_count",
        "_watermark_exclude_prefix_keys","_watermark_prefix_cache_plan")
    previous=[{name:(hasattr(proc,name),getattr(proc,name,None)) for name in names} for proc in processors]
    original=transformer.forward;signature=inspect.signature(original);layout={}
    def forward(*args,**kwargs):
        values=signature.bind(*args,**kwargs).arguments
        img_mask=values["img_mask"];shapes=values["img_shapes"];hidden=values["hidden_states"]
        if not layout:
            keep,count,fractions=target_key_keep(transformer,items,img_mask,shapes,hidden.device,margin_cells)
            layout.update(mask=img_mask.detach().clone(),shapes=shapes,keep=keep,count=count)
            if compact_cache:
                from .reference_cache import ReferencePrefixCache
                layout["cache_plan"]=ReferencePrefixCache(keep,count)
                audit.update(original_prefix_tokens=layout["cache_plan"].prefix_tokens,
                    compact_prefix_tokens=layout["cache_plan"].compact_tokens,
                    prefix_cache_fraction=layout["cache_plan"].compact_tokens/layout["cache_plan"].prefix_tokens)
            audit.update(method=("all_query_reference_key_exclusion_v1" if exclude_prefix
                else "target_only_reference_key_exclusion_v1"),
                excluded_source_fractions=fractions,target_tokens=count,joint_key_tokens=keep.shape[1])
        elif (shapes!=layout["shapes"] or img_mask.device!=layout["mask"].device
                or not torch.equal(img_mask,layout["mask"])):
            raise ValueError("Reference exclusion token layout changed within a generation")
        active=True
        if max_sigma<1:
            timestep=values.get("timestep")
            if not isinstance(timestep,torch.Tensor) or timestep.numel() not in (1,len(items)):
                raise ValueError("Actual normalized model timesteps are required for scheduled exclusion")
            sigmas=timestep.detach().float().flatten()
            sigma=float(sigmas[0].item())
            if not 0<=sigma<=1 or not torch.all(sigmas==sigmas[0]).item():
                raise ValueError("Scheduled reference exclusion requires homogeneous flow sigmas")
            active=sigma<=max_sigma+1e-7
            audit.setdefault("forward_sigmas",[]).append(sigma)
        for proc in processors:
            proc._watermark_target_key_keep=layout["keep"] if active else None
            proc._watermark_target_query_count=layout["count"]
            proc._watermark_exclude_prefix_keys=exclude_prefix
            proc._watermark_prefix_cache_plan=layout.get("cache_plan")
        audit["forward_calls"]+=1
        audit["active_forward_calls"]+=int(active)
        return original(*args,**kwargs)
    try:
        with override_methods(transformer,forward=forward):
            yield audit
    finally:
        for proc,old in zip(processors,previous):
            for name,(owned,value) in old.items():
                if owned:setattr(proc,name,value)
                elif hasattr(proc,name):delattr(proc,name)
