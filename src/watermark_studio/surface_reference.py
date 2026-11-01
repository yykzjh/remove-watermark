"""Visible, source-only smooth-surface references for texture measurement scope."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageFilter


def _box(value):
    if (not isinstance(value,(list,tuple)) or len(value)!=4 or
            any(type(x) not in (int,float) or not math.isfinite(x) for x in value)):
        return None
    a,b,c,d=value
    return tuple(value) if 0<=a<c<=1 and 0<=b<d<=1 else None


def source_rims(rgb,regions,cores):
    """Only raw detector cores with matching padded masks define visible margins."""
    height,width=rgb.shape[:2]
    masks=[tuple(r.box) for r in regions]
    mapping={}
    for item in cores or []:
        if not isinstance(item,dict):
            continue
        mask,core=_box(item.get("mask")),_box(item.get("core"))
        if mask is None or core is None:
            continue
        match=next((m for m in masks if all(abs(a-b)<1e-6 for a,b in zip(m,mask))),None)
        if match and mask[0]<=core[0]<core[2]<=mask[2] and mask[1]<=core[1]<core[3]<=mask[3]:
            mapping[match]=core
    if not mapping:
        return {}
    unknown=[mapping.get(mask,mask) for mask in masks]
    gray=Image.fromarray(rgb.astype(np.uint8)).convert("L")
    high=np.abs(np.asarray(gray,dtype=np.float32)-
                np.asarray(gray.filter(ImageFilter.GaussianBlur(1)),dtype=np.float32))
    result={}
    for index,mask in enumerate(masks):
        if mask not in mapping:
            continue
        core=mapping[mask]
        bounds=tuple(round(v*s) for v,s in zip(mask,(width,height,width,height)))
        l,t,r,b=(round(v*s) for v,s in zip(core,(width,height,width,height)))
        sides=[]
        for side,coord,outer in [("left",l,bounds[0]),("right",r,bounds[2]),
                                 ("top",t,bounds[1]),("bottom",b,bounds[3])]:
            vertical=side in {"left","right"}
            lower=side in {"left","top"}
            # A four-pixel gap separates the reference from the UNKNOWN core.
            low,high_at=(coord-8,coord-4) if lower else (coord+4,coord+8)
            if low<0 or high_at>(width if vertical else height):
                continue
            if (lower and low<outer+1) or (not lower and high_at>outer-1):
                continue
            start,end=(t,b) if vertical else (l,r)
            tiles=[]
            for at in range(start+8,end-39,32):
                native=(low,at,high_at,at+32) if vertical else (at,low,at+32,high_at)
                box=[v/s for v,s in zip(native,(width,height,width,height))]
                blocked=any(max(box[0],q[0])<min(box[2],q[2]) and
                            max(box[1],q[1])<min(box[3],q[3]) for q in unknown)
                if blocked:
                    tiles.append(None);continue
                x0,y0,x1,y1=native
                sample=rgb[y0:y1,x0:x1]
                tiles.append({"box":box,"detail":float(high[y0:y1,x0:x1].mean()),
                              "spread":float(sample.std(axis=(0,1)).mean()),
                              "rgb":np.median(sample.reshape(-1,3),axis=0).tolist()})
            sides.append({"side":side,"tiles":tiles})
        result[index]=sides
    return result


def matching_smooth_run(sides,colour):
    """Require 96 consecutive source pixels of visible, similarly coloured surface.

    This replaces an inappropriate cross-material reference, never a VLM veto.
    Existing 0.8 detail floor is retained; core pixels cannot support this proof.
    """
    for side in sides:
        run=[]
        for tile in side["tiles"]:
            delta=(float(np.abs(np.asarray(tile["rgb"])-colour).mean())
                   if tile is not None else math.inf)
            if tile is None or tile["detail"]>.8 or tile["spread"]>8 or delta>12:
                run=[];continue
            run.append(tile|{"colour_delta":round(delta,3)})
            if len(run)>=3:
                return {"side":side["side"],"reference_boxes":[r["box"] for r in run],
                        "reference_detail_max":round(max(r["detail"] for r in run),4),
                        "colour_delta_max":round(max(r["colour_delta"] for r in run),3),
                        "visible_source_pixels":96}
    return None
