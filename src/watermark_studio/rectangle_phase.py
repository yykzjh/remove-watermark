"""Conservative aligned-boundary cue; findings require review, never approval."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def _prepare_phase(before, after, resolution, blur_radius=1.2):
    scale=min(1,resolution/max(after.size))
    size=(max(32,round(after.width*scale)),max(32,round(after.height*scale)))
    photos=[image.convert("RGB").resize(size,Image.Resampling.LANCZOS).filter(
        ImageFilter.GaussianBlur(blur_radius)) for image in (before,after)]
    source,rgb=[np.asarray(image,dtype=np.float32) for image in photos]
    y=np.log(16+rgb@np.array([.2126,.7152,.0722],dtype=np.float32))
    return source,rgb,y,size


def _single_phase(before, after, regions, resolution=1024, forced_sign=None, core_mapping=None, prepared=None, core_radius=5):
    """Require a sharp, spatially coherent tone step on every available panel side."""
    panels=[r for r in regions if r.kind=="panel" and (r.box[2]-r.box[0])*(r.box[3]-r.box[1])>=.06]
    if not panels:
        return {"method":"source_panel_phase_v1","count":0,"findings":[],"comparison_size":None}
    source,rgb,y,size=prepared if prepared is not None else _prepare_phase(before,after,resolution)
    findings=[]
    for region in panels:
        l,t,r,b=region.box
        if (r-l)*(b-t)<.06:
            continue
        core=(core_mapping or {}).get(tuple(region.box))
        if core_mapping is not None and core is None:
            continue
        px=.008+.06*max(0,(r-l-.016)/1.12)
        py=.008+.06*max(0,(b-t-.016)/1.12)
        evidence={};available=[];contaminated=False
        for side in ("top","bottom","left","right"):
            vertical=side in ("left","right")
            coordinate={"left":l,"right":r,"top":t,"bottom":b}[side]
            if core is not None:
                coordinate=core[{"left":0,"top":1,"right":2,"bottom":3}[side]]
            if coordinate in (0,1):
                continue
            inward=1 if side in ("left","top") else -1
            extent=size[0] if vertical else size[1]
            along=size[1] if vertical else size[0]
            start,end=((core[1],core[3]) if vertical else (core[0],core[2])) if core else ((t,b) if vertical else (l,r))
            margin=px if vertical else py
            lo,hi=round((start+(end-start)*.18)*along),round((end-(end-start)*.18)*along)
            positions=[p for p in range(round(coordinate*extent)-2,
                                       round(coordinate*extent)+inward*round(margin*extent*1.8)+inward,
                                       inward) if 17<p<extent-18]
            if core is not None:
                centre=round(coordinate*extent)
                positions=[pos for pos in range(centre-core_radius,centre+core_radius+1) if 17<pos<extent-18]
            if hi-lo<64 or len(positions)<5:
                continue
            band=round(margin*extent*1.8)+18
            centre=coordinate*extent
            zone=([max(0,centre-band)/size[0],lo/size[1],min(size[0],centre+band)/size[0],hi/size[1]]
                  if vertical else [lo/size[0],max(0,centre-band)/size[1],hi/size[0],min(size[1],centre+band)/size[1]])
            if any(other is not region and max(zone[0],other.box[0])<min(zone[2],other.box[2])
                   and max(zone[1],other.box[1])<min(zone[3],other.box[3]) for other in regions):
                contaminated=True
                break
            available.append(side)
            profiles=[]
            for first in range(lo,hi-23,24):
                last=first+24
                values=[]
                for pos in positions:
                    def stripe(image,offset,pos=pos,inward=inward,first=first,last=last,vertical=vertical):
                        at=pos+inward*offset
                        return image[first:last,at-1:at+2] if vertical else image[at-1:at+2,first:last]
                    means=[float(np.median(stripe(y,offset))) for offset in (5,-5,13,-13)]
                    values.append((means[0]-means[1])-(5/13)*(means[2]-means[3]))
                profiles.append(values)
            if len(profiles)<3:
                continue
            profiles=np.asarray(profiles,dtype=np.float32)
            average=np.mean(np.sort(profiles,axis=0)[1:-1],axis=0) if len(profiles)>=5 else np.mean(profiles,axis=0)
            candidate=int(np.argmax(np.abs(average) if forced_sign is None else average*forced_sign))
            step=float(average[candidate])
            sign=1 if step>=0 else -1
            peaks=np.argmax(profiles*sign,axis=1)
            phase=float(np.mean(np.abs(peaks-candidate)<=3))
            coherence=float(np.mean(profiles[:,candidate]*sign>=.012))
            if (forced_sign is not None and sign!=forced_sign) or abs(step)<.022 or phase<.5 or coherence<.65:
                continue
            pos=positions[candidate]
            changes=[]
            for first in range(lo,hi-23,24):
                last=first+24
                at=pos+inward*5
                a=source[first:last,at-1:at+2] if vertical else source[at-1:at+2,first:last]
                z=rgb[first:last,at-1:at+2] if vertical else rgb[at-1:at+2,first:last]
                changes.append(float(np.abs(a-z).mean()))
            if float(np.median(changes))<3:
                continue
            evidence[side]={"position":pos/extent,"log_tone_step":round(step,5),
                            "phase_support":phase,"sign_support":coherence}
        # All usable sides, at least one horizontal and one vertical, same sign.
        if (contaminated or len(available)<2 or set(evidence)!=set(available) or
                not set(evidence)&{"top","bottom"} or not set(evidence)&{"left","right"}):
            continue
        steps=[v["log_tone_step"] for v in evidence.values()]
        if min(steps)*max(steps)<=0 or max(abs(x) for x in steps)>3*min(abs(x) for x in steps):
            continue
        findings.append({"box":list(region.box),"side_evidence":evidence})
    return {"method":"source_panel_phase_v1","count":len(findings),"findings":findings,
            "comparison_size":list(size)}


def rectangle_phase(before, after, regions, resolution=1024, *, _prepared=None):
    """Joint bright/dark hypotheses keep texture distractors from hiding a box.

    All side, amplitude, coherence, contamination and source-change conditions
    are identical to the original absolute-peak search. This only adds vetoes.
    """
    findings={}
    size=None
    has_panel=any(r.kind=="panel" and (r.box[2]-r.box[0])*(r.box[3]-r.box[1])>=.06 for r in regions)
    prepared=(_prepared if _prepared is not None else _prepare_phase(before,after,resolution)) if has_panel else None
    for sign in (-1,1):
        result=_single_phase(before,after,regions,resolution,sign,prepared=prepared)
        size=result["comparison_size"]
        for item in result["findings"]:
            key=tuple(item["box"])
            # One record per panel, even if it contains both bright and dark bands.
            findings.setdefault(key,item|{"joint_sign":sign})
    return {"method":"source_panel_phase_v2_joint","count":len(findings),
            "findings":list(findings.values()),"comparison_size":size}


def rectangle_phase_core(before, after, regions, core_boxes, resolution=1024, *, _core_prepared=None):
    """Experimental added vetoes near validated raw detector edges; no approvals.

    Keep every original warning and all amplitude/coherence/source conditions.
    A natural texture peak elsewhere in the padded margin cannot hide a weaker
    but coherent step at the actual source panel edge.
    """
    from .surface_reference import _box
    mapping = {}
    for item in core_boxes or []:
        if not isinstance(item, dict):
            continue
        mask, core = _box(item.get("mask")), _box(item.get("core"))
        if mask is None or core is None:
            continue
        match = next((tuple(r.box) for r in regions if
                      all(abs(a-b)<1e-6 for a,b in zip(r.box,mask))), None)
        if (match is not None and mask[0]<=core[0]<core[2]<=mask[2]
                and mask[1]<=core[1]<core[3]<=mask[3]):
            mapping[match] = core
    has_panel=any(r.kind=="panel" and (r.box[2]-r.box[0])*(r.box[3]-r.box[1])>=.06 for r in regions)
    prepared=_prepare_phase(before,after,resolution) if has_panel else None
    baseline = rectangle_phase(before,after,regions,resolution,_prepared=prepared)
    findings = {tuple(item["box"]): item for item in baseline["findings"]}
    if mapping:
        # Additional evidence needs stronger side-to-side consistency. The
        # original v2 vetoes retain their exact thresholds and original pixels.
        # The safety-expanded core includes .008 frame padding: at a 1024px
        # comparison the actual photographed panel edge can be >5px inward.
        # Probe a bounded 12px neighborhood; still require every usable side.
        core_prepared=(_core_prepared if _core_prepared is not None
                       else _prepare_phase(before,after,resolution,blur_radius=3.))
        for sign in (-1,1):
            result = _single_phase(before,after,regions,resolution,sign,mapping,core_prepared,core_radius=12)
            for item in result["findings"]:
                steps=[abs(v["log_tone_step"]) for v in item["side_evidence"].values()]
                if max(steps)>2*min(steps):
                    continue
                findings.setdefault(tuple(item["box"]),item|{"joint_sign":sign,"source_core_search":True,
                    "core_blur_radius":3.,"maximum_side_ratio":2.,"core_search_radius":12})
    return baseline|{"method":"source_panel_phase_v4_core_experiment",
                     "findings":list(findings.values()),"count":len(findings)}


def rectangle_phase_tight(before, after, regions, core_boxes, resolution=1024):
    """Experimental added veto with bounded core search; never an approval.

    Preserve v4 and every original v2 warning. Only the search neighborhood changes;
    amplitude, coherence, source-change, all-side and common-sign requirements stay.
    Raw detector cores are hypotheses, not permission to change an original image.
    """
    from .surface_reference import _box

    mapping={}
    for item in core_boxes or []:
        if not isinstance(item,dict):
            continue
        mask,core=_box(item.get("mask")),_box(item.get("core"))
        if mask is None or core is None:
            continue
        match=next((tuple(r.box) for r in regions
                    if all(abs(a-b)<1e-6 for a,b in zip(r.box,mask))),None)
        if (match is not None and mask[0]<=core[0]<core[2]<=mask[2]
                and mask[1]<=core[1]<core[3]<=mask[3]):
            mapping[match]=core
    prepared=_prepare_phase(before,after,resolution,blur_radius=3.) if mapping else None
    base=rectangle_phase_core(before,after,regions,core_boxes,resolution,_core_prepared=prepared)
    findings={tuple(item["box"]):item for item in base["findings"]}
    if mapping:
        for sign in (-1,1):
            result=_single_phase(before,after,regions,resolution,sign,mapping,prepared,core_radius=2)
            for item in result["findings"]:
                findings.setdefault(tuple(item["box"]),item|{"joint_sign":sign,"tight_source_core":True,
                    "core_blur_radius":3.,"core_search_radius":2})
    return base|{"method":"source_panel_phase_v5_tight_experiment",
                 "findings":list(findings.values()),"count":len(findings)}
