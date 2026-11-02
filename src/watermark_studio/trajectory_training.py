"""Offline exact-target supervision on frozen native denoising trajectories."""
from __future__ import annotations

import math
from itertools import pairwise


def context_support(values, height, width, margin=1):
    import torch
    from torch.nn import functional
    if type(margin) is not int or not 0<=margin<=16:
        raise ValueError("Bounded integer latent margin required")
    mask=values["edit_mask"]
    if (mask.ndim!=2 or not 1<=mask.shape[0]<=8
            or mask.shape!=(values["source_latents"].shape[0],(height//16)*(width//16))
            or height%32 or width%32):
        raise ValueError("Single complete native latent grid required")
    if not torch.isfinite(mask).all() or not ((mask>=0)&(mask<=1)).all():
        raise ValueError("Finite exact overlay support required")
    mask=(mask>0).float().reshape(mask.shape[0],1,height//16,width//16)
    mask=functional.max_pool2d(mask,2*margin+1,1,margin)
    return mask.flatten(2).transpose(1,2).to(values["source_latents"].dtype)


def cpu_packed_noise(height,width,channels,seed,dtype):
    """Match pipeline CPU generator and channel-first VAE packing."""
    import torch
    value=torch.randn((1,channels,1,height//16,width//16),
        generator=torch.Generator("cpu").manual_seed(seed),device="cpu",dtype=dtype)
    return value.squeeze(2).flatten(2).transpose(1,2).contiguous()


def terminal_objective(prediction,terminal,clean,sigma,mask,*,normalize_velocity=False):
    """Supervise final Euler x0 against exact real clean VAE target, never teacher x0."""
    import torch

    from .lora_training import flow_objective
    if type(normalize_velocity) is not bool:
        raise ValueError("Explicit terminal velocity normalization flag required")
    if isinstance(sigma,bool) or not math.isfinite(sigma) or not 0<sigma<=1:
        raise ValueError("Positive final native sigma required")
    # Flow objective's algebra supplies common shape/support/weight validation.
    predicted=terminal.float()-sigma*prediction.float()
    value=flow_objective(predicted,clean.float(),torch.zeros_like(clean),mask)
    # Prefix regularization is measured in velocity units. Optional division
    # removes the sigma-squared attenuation of an endpoint-space objective.
    return value/(sigma*sigma) if normalize_velocity else value


def prefix_objective(prediction,teacher):
    import torch
    if prediction.shape!=teacher.shape or prediction.ndim!=3:
        raise ValueError("Aligned full-frame teacher velocity required")
    if not torch.isfinite(teacher).all():
        raise ValueError("Finite frozen teacher velocity required")
    return (prediction.float()-teacher.float()).square().mean()


def current_prefix(predict,noise,reference,sigmas,support=None):
    """Recompute current student's prefix, retaining gradients only at the final node."""
    import torch
    values=[float(v) for v in sigmas]
    if (noise.shape!=reference.shape or noise.ndim!=3 or len(values)<3 or values[-1]!=0
            or any(not math.isfinite(v) or not 0<v<=1 for v in values[:-1])
            or any(a<=b for a,b in pairwise(values))):
        raise ValueError("Complete descending deterministic trajectory required")
    if support is not None and (support.shape!=(*noise.shape[:2],1) or
            not torch.isfinite(support).all() or not ((support==0)|(support==1)).all()):
        raise ValueError("Matching binary whole-frame context support required")
    def blend(value,sigma):
        return value if support is None else support*value+(1-support)*((1-sigma)*reference+sigma*noise)
    with torch.no_grad():
        current=blend(noise.clone(),values[0])
        for index in range(len(values)-2):
            velocity=predict(current,values[index],index)
            if velocity.shape!=current.shape or velocity.dtype!=current.dtype:
                raise ValueError("Current model changed the packed prefix layout")
            current=(current.float()+(values[index+1]-values[index])*velocity.float()).to(velocity.dtype)
            current=blend(current,values[index+1])
    return current.detach()
