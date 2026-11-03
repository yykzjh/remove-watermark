"""Bounded color-step removal; no new texture, geometry or QA permissions."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .images import pixel_box
from .rectangle_phase import rectangle_phase_tight

SIDES={"left":(True,1,0),"right":(True,-1,2),"top":(False,1,1),"bottom":(False,-1,3)}


def _profile(rgb,core,side,pos):
    vertical,inward,_=SIDES[side];height,width=rgb.shape[:2]
    extent=width if vertical else height;along=height if vertical else width
    first,last=(core[1],core[3]) if vertical else (core[0],core[2])
    lo,hi=round((first+.18*(last-first))*along),round((last-.18*(last-first))*along)
    if hi-lo<48:return None
    offsets=[];differences=[]
    for h in (1,2,3,5,8,13,21):
        inside=pos+h-1 if inward==1 else pos-h
        outside=pos-h if inward==1 else pos+h-1
        if not 0<=inside<extent or not 0<=outside<extent:continue
        a=rgb[lo:hi,inside] if vertical else rgb[inside,lo:hi]
        b=rgb[lo:hi,outside] if vertical else rgb[outside,lo:hi]
        differences.append(np.median(a-b,axis=0));offsets.append(h-.5)
    if len(offsets)<3:return None
    return np.asarray(offsets),np.asarray(differences)


def _fit(profile,sigma,delta=None):
    h,values=profile
    step=np.ones(len(h)) if sigma==0 else np.array([math.erf(v/(math.sqrt(2)*sigma)) for v in h])
    gradient=2*h
    if delta is None:
        design=np.stack([step,gradient],axis=1)
        normal=design/np.maximum(np.linalg.norm(design,axis=0,keepdims=True),1e-8)
        singular=np.linalg.svd(normal,compute_uv=False)
        if singular[-1]/singular[0]<.05:return None
        weights=np.linalg.lstsq(design,values,rcond=None)[0];delta=weights[0]
        residual=values-design@weights
    else:
        remaining=values-step[:,None]*delta
        slope=(gradient[:,None]*remaining).sum(0)/np.square(gradient).sum()
        residual=remaining-gradient[:,None]*slope
    return np.asarray(delta),float(np.sqrt(np.square(residual).mean()))


def repair_tone_steps(source,candidate,regions,cores=(),*,resolution=1024,max_delta=32):
    """Fit one shared RGB step only after the unchanged conservative phase cue."""
    if source.size!=candidate.size or resolution not in (512,768,1024) or not 4<=max_delta<=32:
        raise ValueError("Aligned full frames and bounded tone repair required")
    initial=rectangle_phase_tight(source,candidate,regions,cores,resolution)
    audit={"method":"bounded_shared_rgb_step_v1","applied":False,"quality_permission":False,
        "extra_diffusion_inferences":0,"initial_phase_count":initial["count"],"repairs":[]}
    if not initial["count"]:return candidate.copy(),audit
    scale=min(1,resolution/max(candidate.size));size=(round(candidate.width*scale),round(candidate.height*scale))
    rgb=np.asarray(candidate.convert("RGB").resize(size,Image.Resampling.LANCZOS),dtype=np.float64)
    original=np.asarray(candidate.convert("RGB"),dtype=np.uint8);output=original.astype(np.float64)
    scope=Image.new("L",source.size,0);draw=ImageDraw.Draw(scope)
    for region in regions:
        l,t,r,b=pixel_box(region.box,source.size);draw.rectangle((l,t,r-1,b-1),fill=255)
    scope_array=np.asarray(scope)>0
    for finding in initial["findings"][:2]:
        mask=tuple(finding["box"]);core=next((tuple(row["core"]) for row in cores
            if isinstance(row,dict) and tuple(row.get("mask",()))==mask),mask)
        evidence=finding["side_evidence"];rich={}
        for side,record in evidence.items():
            vertical,_,_=SIDES[side];extent=size[0] if vertical else size[1]
            centre=round(record["position"]*extent)
            rich[side]=[(pos,_profile(rgb,core,side,pos)) for pos in range(max(1,centre-6),min(extent,centre+7))]
            rich[side]=[(pos,profile) for pos,profile in rich[side] if profile is not None]
        if len(rich)<2 or any(not values for values in rich.values()):continue
        choices=[]
        for sigma in (0.,.5,1.,2.,3.,4.,6.,8.):
            fitted={}
            for side,profiles in rich.items():
                best=None
                for pos,profile in profiles:
                    fit=_fit(profile,sigma)
                    if fit is None:continue
                    delta,error=fit;amplitude=float(np.abs(delta).mean())
                    if not 4<=amplitude or np.abs(delta).max()>max_delta:continue
                    if float(delta@np.array([.2126,.7152,.0722]))*evidence[side]["log_tone_step"]<=0:continue
                    score=error/amplitude+1e-4*abs(pos-evidence[side]["position"]*(size[0] if SIDES[side][0] else size[1]))
                    if best is None or score<best[0]:best=(score,pos,delta,error)
                if best is not None:fitted[side]=best
            if set(fitted)!=set(rich):continue
            delta=np.median([value[2] for value in fitted.values()],axis=0)
            amplitude=float(np.abs(delta).mean())
            deviation=max(float(np.sqrt(np.square(v[2]-delta).mean())) for v in fitted.values())
            if deviation>max(2.,.2*amplitude) or max(v[3] for v in fitted.values())>.2*amplitude:continue
            choices.append((sum(v[0] for v in fitted.values())+deviation/amplitude+.0001*sigma,sigma,delta,fitted))
        if not choices:continue
        _,sigma,delta,fitted=min(choices,key=lambda v:v[0]);bounds={}
        possible=True
        for side,(vertical,_,coordinate) in SIDES.items():
            extent=size[0] if vertical else size[1]
            lo,hi=math.floor(mask[0 if vertical else 1]*extent),math.ceil(mask[2 if vertical else 3]*extent)
            if side in fitted:position=fitted[side][1]
            elif core[coordinate] in (0,1):position=round(core[coordinate]*extent)
            else:
                centre=round(core[coordinate]*extent);radius=min(32,max(8,round(.025*extent)));best=None
                for pos in range(max(1,lo,centre-radius),min(extent,hi+1,centre+radius+1)):
                    profile=_profile(rgb,core,side,pos)
                    if profile is None:continue
                    fit=_fit(profile,sigma,delta)
                    if fit is None:continue
                    score=fit[1]+.001*abs(pos-centre)
                    if best is None or score<best[0]:best=(score,pos,fit[1])
                if best is None or best[2]>.2*float(np.abs(delta).mean()):possible=False;break
                position=best[1]
            if not lo<=position<=hi:possible=False;break
            bounds[side]=position
        if not possible:continue
        if bounds["right"]-bounds["left"]<48 or bounds["bottom"]-bounds["top"]<48:continue
        box=(bounds["left"],bounds["top"],bounds["right"],bounds["bottom"])
        small=Image.new("L",size,0);ImageDraw.Draw(small).rectangle((box[0],box[1],box[2]-1,box[3]-1),fill=255)
        if sigma:small=small.filter(ImageFilter.GaussianBlur(sigma))
        weight=np.asarray(small.resize(source.size,Image.Resampling.BILINEAR),dtype=np.float64)/255
        weight*=scope_array
        output-=weight[...,None]*delta
        audit["repairs"].append({"source_box":list(mask),"comparison_bounds":list(box),
            "rgb_step":delta.tolist(),"edge_sigma":sigma,"common_step_fit":True})
    if not audit["repairs"]:return candidate.copy(),audit
    output=np.clip(np.rint(output),0,255).astype(np.uint8)
    if not np.array_equal(output[~scope_array],original[~scope_array]):
        raise ValueError("Tone repair changed pixels outside the immutable edit scope")
    result=Image.fromarray(output)
    if "A" in candidate.getbands():result.putalpha(candidate.getchannel("A"))
    after=rectangle_phase_tight(source,result,regions,cores,resolution)
    audit["final_phase_count"]=after["count"]
    if after["count"]>=initial["count"]:
        return candidate.copy(),audit|{"reason":"no_phase_reduction","repairs":audit["repairs"]}
    audit.update(applied=True,reason="one_shared_color_step_repair")
    return result,audit

def protected_tone_repair(source, candidate, regions, cores=(), *, scene="outdoor",
                          source_rims=(), source_edges=()):
    """One bounded photometric correction before QA, with immutable source exclusions."""
    from .images import protect_edge_completion_context
    from .source_panel_rims import effective_rim_scope

    if scene != "outdoor" or not regions:
        return candidate, {"applied": False, "reason": "no_outdoor_edit_scope",
                           "quality_permission": False, "extra_diffusion_inferences": 0}
    regions, cores, invalid = effective_rim_scope(
        source, regions, cores, source_rims, header_core=True)
    if invalid:
        return candidate, {"applied": False, "reason": "invalid_source_rim_binding",
                           "quality_permission": False, "extra_diffusion_inferences": 0}
    fixed, audit = repair_tone_steps(source, candidate, regions, cores)
    if audit["applied"]:
        fixed, _ = protect_edge_completion_context(candidate, fixed, source_edges)
        for rim in source_rims:
            if rim.get("approved") is True:
                native = tuple(rim["restore_pixels"])
                fixed.paste(candidate.crop(native), native)
        # Immutable exclusions may invalidate a proposal. Recheck the same cue.
        after = rectangle_phase_tight(source, fixed, regions, cores, 1024)
        if after["count"] >= audit["initial_phase_count"]:
            return candidate, audit | {"applied": False, "reason": "protected_phase_not_reduced"}
        audit["final_phase_count"] = after["count"]
    return fixed, audit
