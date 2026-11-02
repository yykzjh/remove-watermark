"""Offline transmitted-detail experiments; never a watermark-free certificate."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageFilter


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

def neutral_detail_candidates(source):
    """Conservative pixel exclusions; these are candidates, never source permissions."""
    rgb=np.asarray(source.convert("RGB"));luminance=np.asarray(source.convert("L"))
    dark=np.asarray(Image.fromarray((luminance<140).astype(np.uint8)*255).filter(ImageFilter.MaxFilter(9)))>0
    bright=np.asarray(Image.fromarray((luminance>225).astype(np.uint8)*255).filter(ImageFilter.MaxFilter(5)))>0
    chromatic=np.asarray(Image.fromarray((np.ptp(rgb.astype(np.int16),axis=2)>24).astype(np.uint8)*255).filter(
        ImageFilter.MaxFilter(3)))>0
    return ~(dark|bright|chromatic)


def estimate_patch_transmission(source,generated,support,*,radius=32):
    """Offline coarse affine hypothesis; no GT opacity or approval is used."""
    if source.size!=generated.size or type(radius) is not int or not 8<=radius<=64:
        raise ValueError("Aligned bounded coarse-transmission diagnostic required")
    src=np.asarray(source.convert("RGB"),dtype=np.float64)
    gen=np.asarray(generated.convert("RGB"),dtype=np.float64);weight=np.asarray(support,dtype=float)
    if weight.shape!=src.shape[:2] or not np.isfinite(weight).all() or not ((weight>=0)&(weight<=1)).all():
        raise ValueError("Explicit text-excluded support required")
    record={"method":"masked_coarse_transmission_v1","eligible":False,"quality_permission":False}
    selected=weight>.5
    if selected.sum()<256:return record|{"reason":"insufficient_support"}
    low_src=masked_mean(src,weight,radius);low_gen=masked_mean(gen,weight,radius)
    def fit(selection):
        y,x=low_src[selection],low_gen[selection]
        if len(x)<64:return None
        stride=max(1,len(x)//4096);x=x[::stride];y=y[::stride]
        centred=x-x.mean(0);reference=y-y.mean(0);variance=float(np.square(centred).mean())
        if variance<4:return None
        t=float((centred*reference).sum()/np.square(centred).sum())
        error=reference-t*centred
        r2=1-float(np.square(error).sum())/max(float(np.square(reference).sum()),1e-8)
        return {"transmission":t,"r2":r2,"coarse_variance":variance}
    full=fit(selected)
    if full is None:return record|{"reason":"coarse_field_unobservable"}
    record.update(full)
    if not .25<=full["transmission"]<=.85 or full["r2"]<.9:
        return record|{"reason":"coarse_affine_hypothesis_not_established"}
    height,width=selected.shape;parts=[]
    for top,bottom,left,right in [(0,height//2,0,width//2),(0,height//2,width//2,width),
            (height//2,height,0,width//2),(height//2,height,width//2,width)]:
        part=np.zeros_like(selected);part[top:bottom,left:right]=selected[top:bottom,left:right]
        fitted=fit(part)
        if fitted is not None and fitted["r2"]>=.85:parts.append(fitted["transmission"])
    record["quadrant_transmissions"]=parts
    if len(parts)<2 or max(abs(t-full["transmission"]) for t in parts)>.2*full["transmission"]:
        return record|{"reason":"spatial_transmission_inconsistent"}
    src_hf=src-masked_mean(src,weight,12);gen_hf=gen-masked_mean(gen,weight,12)
    src_rms=float(np.sqrt(np.square(src_hf[selected]).mean()))
    gen_rms=float(np.sqrt(np.square(gen_hf[selected]).mean()))
    record.update(source_detail_rms=src_rms,generated_detail_rms=gen_rms)
    if src_rms<=1.15*gen_rms:
        return record|{"reason":"no_observable_detail_deficit"}
    return record|{"eligible":True,"reason":"coarse_fit_and_source_detail_deficit"}
