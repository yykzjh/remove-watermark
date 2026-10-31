"""Conservative rectangle-aligned tone/texture cues with source-pixel safeguards."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def rectangular_repairs(before, after, regions, resolution=1024, minimum_step=4,
                        coverage=.70, ratio_tolerance=.055):
    scale=min(1,resolution/max(after.size))
    size=(max(1,round(after.width*scale)),max(1,round(after.height*scale)))
    photo=after.convert("RGB").resize(size,Image.Resampling.LANCZOS)
    rgb=np.asarray(photo.filter(ImageFilter.GaussianBlur(3)),dtype=np.float32)
    weights=np.array([.2126,.7152,.0722],dtype=np.float32)
    gray=np.asarray(photo.convert("L"),dtype=np.float32)
    high=np.abs(gray-np.asarray(photo.convert("L").filter(ImageFilter.GaussianBlur(1)),dtype=np.float32))
    source=np.asarray(before.convert("RGB").resize(size,Image.Resampling.LANCZOS)
                      .filter(ImageFilter.GaussianBlur(3)),dtype=np.float32)
    records=[]
    for region in regions:
        l,t,r,b=region.box if hasattr(region,"box") else region["box"]
        if (r-l)*(b-t)<.06 or min((r-l)*size[0],(b-t)*size[1])<80:
            continue
        px=.008+.06*max(0,(r-l-.016)/1.12)
        py=.008+.06*max(0,(b-t-.016)/1.12)
        side_records={};mandatory=[]
        for side in ("left","right","top","bottom"):
            vertical=side in {"left","right"}
            outward=1 if side in {"left","top"} else -1
            coordinate={"left":l,"right":r,"top":t,"bottom":b}[side]
            if 0<coordinate<1: mandatory.append(side)
            extent=size[0] if vertical else size[1]
            zextent=size[1] if vertical else size[0]
            start,end=(t,b) if vertical else (l,r)
            start=round((start+(end-start)*.18)*zextent)
            end=round((end-(end-start)*.18)*zextent)
            margin=px if vertical else py
            candidates=[]
            for fraction in (0,.25,.5,.75,1,1.25,1.5,1.75):
                pos=round((coordinate+outward*fraction*margin)*extent)
                if not 19<pos<extent-19 or end-start<40: continue
                def stripe(array,offset,lo=start,hi=end,pos=pos,outward=outward,vertical=vertical):
                    at=pos+outward*offset
                    return array[lo:hi,at-2:at+3] if vertical else array[at-2:at+3,lo:hi]
                inner=np.median(stripe(rgb,7).reshape(-1,3),axis=0)
                near=np.median(stripe(rgb,-7).reshape(-1,3),axis=0)
                far=np.median(stripe(rgb,-16).reshape(-1,3),axis=0)
                slope=(near-far)/9
                if abs(float(slope@weights))>.40: continue
                expected=np.clip(near+slope*14,0,255)
                ratios=(inner+12)/(expected+12)
                if np.max(ratios)-np.min(ratios)>.10: continue
                step=float((inner-expected)@weights)
                local=[];texture_ratios=[]
                for lo in range(start,end-15,16):
                    hi=min(end,lo+24)
                    a=np.median(stripe(rgb,7,lo,hi).reshape(-1,3),axis=0)
                    o=np.median(stripe(rgb,-7,lo,hi).reshape(-1,3),axis=0)
                    local.append(float((a-o-slope*14)@weights))
                    oi=float(stripe(high,-7,lo,hi).mean())
                    ii=float(stripe(high,7,lo,hi).mean())
                    if oi>=1.2: texture_ratios.append(ii/oi)
                uniform=float(np.mean(np.asarray(local)*np.sign(step)>minimum_step*.45)) if local else 0
                # Ignore edges which are already present in the original image.
                changed=float(np.abs(stripe(source,7)-stripe(rgb,7)).mean())
                if changed<3: continue
                strength=abs(step)*uniform
                candidates.append({"position":pos/extent,"step":step,"ratio":float(np.median(ratios)),
                    "uniform":uniform,"strength":strength,
                    "texture_ratio":float(np.median(texture_ratios)) if len(texture_ratios)>=3 else 1.0})
            if candidates:
                side_records[side]=candidates
        # A closed shape must agree on every side with usable exterior context.
        # Clipped image edges are optional, never invented as positive evidence.
        if len(mandatory)<2: continue
        for sign in (-1,1):
            selected={}
            for side,values in side_records.items():
                eligible=[v for v in values if v["step"]*sign>=minimum_step and v["uniform"]>=coverage]
                if eligible: selected[side]=max(eligible,key=lambda v:v["strength"])
            if not all(side in selected for side in mandatory): continue
            if len(selected)<3 and len(mandatory)!=2: continue
            scales=[v["ratio"] for v in selected.values()]
            if max(scales)-min(scales)>ratio_tolerance: continue
            records.append({"issue":"rectangular_shadow","box":[l,t,r,b],
                "side_evidence":selected,"mandatory_sides":mandatory,
                "tone_ratio_range":[min(scales),max(scales)]})
    return {"method":"closed_rectangle_tone_v1","count":len(records),"findings":records[:4],
            "comparison_size":list(size)}
