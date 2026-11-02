"""Offline exact-target supervision on frozen native denoising trajectories."""
from __future__ import annotations

import math


def context_support(values, height, width, margin=1):
    import torch
    from torch.nn import functional
    if type(margin) is not int or not 0<=margin<=16:
        raise ValueError("Bounded integer latent margin required")
    mask=values["edit_mask"]
    if mask.shape!=(1,(height//16)*(width//16)) or height%32 or width%32:
        raise ValueError("Single complete native latent grid required")
    if not torch.isfinite(mask).all() or not ((mask>=0)&(mask<=1)).all():
        raise ValueError("Finite exact overlay support required")
    mask=(mask>0).float().reshape(1,1,height//16,width//16)
    mask=functional.max_pool2d(mask,2*margin+1,1,margin)
    return mask.flatten(2).transpose(1,2).to(values["source_latents"].dtype)


def cpu_packed_noise(height,width,channels,seed,dtype):
    """Match pipeline CPU generator and channel-first VAE packing."""
    import torch
    value=torch.randn((1,channels,1,height//16,width//16),
        generator=torch.Generator("cpu").manual_seed(seed),device="cpu",dtype=dtype)
    return value.squeeze(2).flatten(2).transpose(1,2).contiguous()


def terminal_objective(prediction,terminal,clean,sigma,mask):
    """Supervise final Euler x0 against exact real clean VAE target, never teacher x0."""
    import torch

    from .lora_training import flow_objective
    if isinstance(sigma,bool) or not math.isfinite(sigma) or not 0<sigma<=1:
        raise ValueError("Positive final native sigma required")
    # Flow objective's algebra supplies common shape/support/weight validation.
    predicted=terminal.float()-sigma*prediction.float()
    return flow_objective(predicted,clean.float(),torch.zeros_like(clean),mask)


def prefix_objective(prediction,teacher):
    import torch
    if prediction.shape!=teacher.shape or prediction.ndim!=3:
        raise ValueError("Aligned full-frame teacher velocity required")
    if not torch.isfinite(teacher).all():
        raise ValueError("Finite frozen teacher velocity required")
    return (prediction.float()-teacher.float()).square().mean()
