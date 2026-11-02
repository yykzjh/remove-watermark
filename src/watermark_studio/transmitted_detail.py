"""Offline transmitted-detail experiments; never a watermark-free certificate."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image


def masked_mean(values,weights,radius):
    """Linear masked box average: a constant neutral foreground cancels exactly."""
    if values.ndim==2:values=values[...,None]
    def box_sum(array):
        padded=np.pad(array,((radius,radius),(radius,radius),(0,0)),mode="reflect")
        integral=np.pad(padded,((1,0),(1,0),(0,0))).cumsum(0,dtype=np.float64).cumsum(1,dtype=np.float64)
        n=2*radius+1
        return integral[n:,n:]-integral[:-n,n:]-integral[n:,:-n]+integral[:-n,:-n]
    weight=np.asarray(weights,dtype=np.float64)[...,None]
    count=box_sum(weight)
    return box_sum(values.astype(np.float64)*weight)/np.maximum(count,1e-12)


def recover_patch_detail(source,generated,support,*,transmission=.5,radius=12,max_delta=24.):
    """Replace detail only inside explicit support; no extra diffusion inference.

    For an assumed constant-field overlay S=t*B+(1-t)*F,
    S-masked_mean(S)=t*(B-masked_mean(B)). Foreground color is irrelevant.
    Text/opaque/uncertain pixels MUST have zero support; this function cannot
    identify them, certify the assumption, or authorize QA acceptance.
    """
    if (source.size!=generated.size or not math.isfinite(transmission) or
            not .25<=transmission<=.85 or type(radius) is not int or not 2<=radius<=64 or
            not math.isfinite(max_delta) or not 0<max_delta<=32):
        raise ValueError("Common size and bounded explicit detail experiment required")
    src=np.asarray(source.convert("RGB"),dtype=np.float64)
    gen=np.asarray(generated.convert("RGB"),dtype=np.float64)
    weight=np.asarray(support,dtype=np.float64)
    if weight.shape!=src.shape[:2] or not np.isfinite(weight).all() or not ((weight>=0)&(weight<=1)).all():
        raise ValueError("Explicit finite source-text-excluded support in0..1 required")
    if not np.any(weight):return generated.copy(),{"applied":False,"quality_permission":False}
    src_hf=src-masked_mean(src,weight,radius)
    gen_hf=gen-masked_mean(gen,weight,radius)
    delta=np.clip(src_hf/transmission-gen_hf,-max_delta,max_delta)
    result=np.rint(np.clip(gen+weight[...,None]*delta,0,255)).astype(np.uint8)
    if not np.array_equal(result[weight==0],gen.astype(np.uint8)[weight==0]):
        raise ValueError("Detail experiment changed pixels outside explicit support")
    return Image.fromarray(result),{"applied":True,"method":"masked_transmitted_detail_v1",
        "transmission_assumption":transmission,"radius":radius,"max_delta":max_delta,
        "support_pixels":int((weight>0).sum()),"mean_abs_delta":float(np.abs(result.astype(float)-gen).mean()),
        "quality_permission":False,"extra_diffusion_inferences":0}
