"""Experimental source-only alpha recovery for translucent neutral card bodies.

Never used by the live worker. It can recover actual transmitted detail, rather
than inventing texture. Opaque lettering remains excluded and uses the existing
single model candidate. No recovered region constitutes a quality approval.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def fit_transmission(outside, inside):
    x,y=np.asarray(outside,dtype=np.float64),np.asarray(inside,dtype=np.float64)
    if x.shape!=y.shape or x.ndim!=2 or x.shape[1]!=3 or len(x)<64:
        return {"ok":False,"reason":"insufficient_paired_edge_evidence"}
    weights=np.ones(len(x))
    for _ in range(8):
        meanx=np.average(x,axis=0,weights=weights)
        meany=np.average(y,axis=0,weights=weights)
        dx=x-meanx
        denominator=float(np.sum(weights[:,None]*dx*dx))
        if denominator<1:
            return {"ok":False,"reason":"unidentifiable_transmission"}
        beta=float(np.sum(weights[:,None]*dx*(y-meany))/denominator)
        intercept=np.median(y-beta*x,axis=0)
        residual=np.sqrt(np.mean((y-beta*x-intercept)**2,axis=1))
        scale=max(2.,float(np.median(residual))*1.4826)
        weights=np.minimum(1.,(1.5*scale)/np.maximum(residual,1e-6))
    accepted=residual<max(8.,2.5*scale)
    if accepted.sum()<64 or accepted.mean()<.65:
        return {"ok":False,"reason":"inconsistent_edge_samples"}
    error=float(np.sqrt(np.mean((y[accepted]-beta*x[accepted]-intercept)**2)))
    variation=float(np.sqrt(np.mean((y[accepted]-y[accepted].mean(axis=0))**2)))
    r2=1-error**2/max(variation**2,1)
    foreground=intercept/max(1-beta,1e-6)
    report={"ok":False,"transmission":beta,"intercept":intercept.tolist(),
            "foreground":foreground.tolist(),"rmse":error,"r2":r2,"variation":variation,
            "support":int(accepted.sum()),"samples":len(x)}
    if not .35<=beta<=.85:
        report["reason"]="opaque_or_unstable_transmission"
    elif not (foreground.min()>=190 and foreground.max()<=270 and np.ptp(foreground)<=25):
        report["reason"]="not_a_neutral_light_card"
    elif error>7 or r2<.8 or variation<8:
        report["reason"]="weak_physical_fit"
    else:
        report.update(ok=True,reason="consistent_translucent_neutral_edges")
    return report


def recover_translucent_body(before, candidate, core):
    """Conservative exploratory correction inside one source-proven digital card."""
    from .surface_reference import _box
    box=_box(core)
    log={"experimental":True,"applied":False,"method":"source_transmission_edges_v1"}
    if box is None or before.size!=candidate.size:
        return candidate,log|{"reason":"invalid_core_or_size"}
    w,h=before.size
    l,t,r,b=[round(v*s) for v,s in zip(box,(w,h,w,h))]
    if min(r-l,b-t)<100 or (r-l)*(b-t)/(w*h)>.4:
        return candidate,log|{"reason":"unsupported_card_extent"}
    rgb=np.asarray(before.convert("RGB"),dtype=np.float32)
    blurred=np.asarray(before.convert("RGB").filter(ImageFilter.GaussianBlur(2)),dtype=np.float32)
    # Locate the neutral body from source colours only. Exclude blue headers.
    inset=max(12,round((r-l)*.1))
    row=np.median(rgb[t:b,l+inset:r-inset],axis=1)
    neutral=(np.max(row,axis=1)-np.min(row,axis=1)<45)&(np.mean(row,axis=1)>130)
    starts=[i for i in range(len(row)) if neutral[i:min(len(row),i+20)].size==20
            and neutral[i:i+20].mean()>=.9]
    if not starts:
        return candidate,log|{"reason":"no_neutral_body"}
    top=t+starts[0]+8
    if b-top<80:
        return candidate,log|{"reason":"short_neutral_body"}
    # Both vertical sides and the lower outer edge are independent background
    # references. Do not use the blue-header/body boundary as scene evidence.
    edges=[]
    for side,expected in [("left",l),("right",r),("bottom",b)]:
        vertical=side!="bottom"
        extent=w if vertical else h
        if not 20<expected<extent-20:
            continue
        inward=1 if side=="left" else -1
        lo,hi=(top+12,b-20) if vertical else (l+inset,r-inset)
        if hi-lo<64:
            continue
        candidates=[]
        for at in range(max(20,expected-20),min(extent-20,expected+21)):
            def stripe(offset,at=at,inward=inward,lo=lo,hi=hi,vertical=vertical):
                pos=at+inward*offset
                return blurred[lo:hi,pos-1:pos+2].mean(axis=1) if vertical else blurred[pos-1:pos+2,lo:hi].mean(axis=0)
            x,y=stripe(-6),stripe(6)
            contrast=float(np.median(np.linalg.norm(y-x,axis=1)))
            candidates.append((contrast,at,x,y))
        _,at,x,y=max(candidates,key=lambda record:record[0])
        fit=fit_transmission(x,y)
        edges.append({"side":side,"position":at,"fit":fit,"outside":x,"inside":y})
    valid=[edge for edge in edges if edge["fit"]["ok"]]
    log["edges"]=[{k:v for k,v in edge.items() if k not in {"outside","inside"}} for edge in edges]
    if len(valid)<2:
        return candidate,log|{"reason":"need_two_independent_supported_edges"}
    betas=[edge["fit"]["transmission"] for edge in valid]
    offsets=np.asarray([edge["fit"]["intercept"] for edge in valid])
    if max(betas)-min(betas)>.10 or np.ptp(offsets,axis=0).max()>12:
        return candidate,log|{"reason":"edges_disagree"}
    fit=fit_transmission(np.concatenate([e["outside"] for e in valid]),
                         np.concatenate([e["inside"] for e in valid]))
    log["combined_fit"]=fit
    if not fit["ok"]:
        return candidate,log|{"reason":"combined_fit_failed"}
    positions={edge["side"]:edge["position"] for edge in valid}
    l,r,b=positions.get("left",l),positions.get("right",r),positions.get("bottom",b)
    patch=rgb[top:b,l:r]
    beta=fit["transmission"];offset=np.array(fit["intercept"],dtype=np.float32)
    # The foreground envelope excludes opaque/antialiased glyphs. Dilate the
    # exclusion before compositing; never reintroduce source text as texture.
    invalid=np.any((patch<offset-4)|(patch>offset+255*beta+4),axis=2)
    rejected=Image.fromarray((invalid*255).astype(np.uint8)).filter(ImageFilter.MaxFilter(9))
    trusted=1-np.asarray(rejected,dtype=np.float32)/255
    yy,xx=np.mgrid[:b-top,:r-l]
    distance=np.minimum.reduce([xx,yy,r-l-1-xx,b-top-1-yy])
    trusted*=np.clip(distance/12,0,1)
    if float((trusted>.99).mean())<.4:
        return candidate,log|{"reason":"too_little_unobscured_transmission"}
    recovered=(patch-offset)/beta
    target=np.asarray(candidate.convert("RGB"),dtype=np.float32).copy()
    target[top:b,l:r]=trusted[...,None]*np.clip(recovered,0,255)+(1-trusted[...,None])*target[top:b,l:r]
    log.update(applied=True,reason="source_transmission_only",body_pixels=[l,top,r,b],
               recovered_fraction=float((trusted>.99).mean()))
    return Image.fromarray(np.rint(target).astype(np.uint8)),log
