"""Conservative aligned-boundary cue; findings require review, never approval."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def _single_phase(before, after, regions, resolution=1024, forced_sign=None):
    """Require a sharp, spatially coherent tone step on every available panel side."""
    panels=[r for r in regions if r.kind=="panel" and (r.box[2]-r.box[0])*(r.box[3]-r.box[1])>=.06]
    if not panels:
        return {"method":"source_panel_phase_v1","count":0,"findings":[],"comparison_size":None}
    scale=min(1,resolution/max(after.size))
    size=(max(32,round(after.width*scale)),max(32,round(after.height*scale)))
    photos=[image.convert("RGB").resize(size,Image.Resampling.LANCZOS).filter(
        ImageFilter.GaussianBlur(1.2)) for image in (before,after)]
    source,rgb=[np.asarray(image,dtype=np.float32) for image in photos]
    y=np.log(16+rgb@np.array([.2126,.7152,.0722],dtype=np.float32))
    findings=[]
    for region in panels:
        l,t,r,b=region.box
        if (r-l)*(b-t)<.06:
            continue
        px=.008+.06*max(0,(r-l-.016)/1.12)
        py=.008+.06*max(0,(b-t-.016)/1.12)
        evidence={};available=[];contaminated=False
        for side in ("top","bottom","left","right"):
            vertical=side in ("left","right")
            coordinate={"left":l,"right":r,"top":t,"bottom":b}[side]
            if coordinate in (0,1):
                continue
            inward=1 if side in ("left","top") else -1
            extent=size[0] if vertical else size[1]
            along=size[1] if vertical else size[0]
            start,end=(t,b) if vertical else (l,r)
            margin=px if vertical else py
            lo,hi=round((start+(end-start)*.18)*along),round((end-(end-start)*.18)*along)
            positions=[p for p in range(round(coordinate*extent)-2,
                                       round(coordinate*extent)+inward*round(margin*extent*1.8)+inward,
                                       inward) if 17<p<extent-18]
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


def rectangle_phase(before, after, regions, resolution=1024):
    """Joint bright/dark hypotheses keep texture distractors from hiding a box.

    All side, amplitude, coherence, contamination and source-change conditions
    are identical to the original absolute-peak search. This only adds vetoes.
    """
    findings={}
    size=None
    for sign in (-1,1):
        result=_single_phase(before,after,regions,resolution,sign)
        size=result["comparison_size"]
        for item in result["findings"]:
            key=tuple(item["box"])
            # One record per panel, even if it contains both bright and dark bands.
            findings.setdefault(key,item|{"joint_sign":sign})
    return {"method":"source_panel_phase_v2_joint","count":len(findings),
            "findings":list(findings.values()),"comparison_size":size}
