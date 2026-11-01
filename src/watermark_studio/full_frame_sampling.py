"""Offline invertible full-frame sampling; no cropped diffusion inputs."""
from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
from PIL import Image

from .config import Region


def _axis(intervals,mass):
    knots=sorted({0.,1.,*(v for interval in intervals for v in interval)})
    focus=[any(a<=(lo+hi)/2<=b for a,b in intervals) for lo,hi in pairwise(knots)]
    span=sum(b-a for a,b,on in zip(knots,knots[1:],focus) if on)
    if not 0<span<1:
        return {"source":[0.,1.],"model":[0.,1.],"effective_mass":0.}
    amount=min(mass,span/(1-span))
    density=[1-amount+amount/span if on else 1-amount for on in focus]
    model=[0.]
    for a,b,value in zip(knots,knots[1:],density):
        model.append(model[-1]+(b-a)*value)
    model[-1]=1.
    return {"source":knots,"model":model,"effective_mass":amount}


def full_frame_mapping(regions,size,mass=0.,padding=.025):
    if not math.isfinite(mass) or not 0<=mass<=.35 or not 0<=padding<=.05:
        raise ValueError("Bounded full-frame sampling mass and padding required")
    if len(size)!=2 or any(type(v) is not int or not 16<=v<=65536 for v in size) or size[0]*size[1]>200_000_000:
        raise ValueError("Bounded native frame dimensions required")
    if not mass:
        return None
    chosen=sorted([r for r in regions if r.kind=="panel" and
        (r.box[2]-r.box[0])*(r.box[3]-r.box[1])>=.04],
        key=lambda r:(r.box[2]-r.box[0])*(r.box[3]-r.box[1]),reverse=True)[:2]
    if not chosen:
        return None
    boxes=[list(r.box) for r in chosen]
    axes=[]
    for start,end in ((0,2),(1,3)):
        axes.append(_axis([(max(0,b[start]-padding),min(1,b[end]+padding)) for b in boxes],mass))
    if all(a["effective_mass"]==0 for a in axes):
        return None
    return {"method":"whole_frame_separable_cdf_v1","original_size":list(size),
        "mass":mass,"padding":padding,"focus_boxes":boxes,"x":axes[0],"y":axes[1],
        "all_frame_content_retained":True,"cropped_diffusion_input":False}


def validate_mapping(mapping):
    if mapping is None:
        return
    if not isinstance(mapping,dict) or mapping.get("method")!="whole_frame_separable_cdf_v1":
        raise ValueError("Unknown full-frame sampling protocol")
    try:
        boxes=mapping["focus_boxes"]
        if not isinstance(boxes,list) or not 1<=len(boxes)<=2:
            raise ValueError("Bounded automatic focus regions required")
        regenerated=full_frame_mapping([Region(kind="panel",box=b) for b in boxes],
            mapping["original_size"],mapping["mass"],mapping["padding"])
        if mapping!=regenerated:
            raise ValueError("Mapping is not derived from the recorded full frame and automatic regions")
    except (KeyError,TypeError) as error:
        raise ValueError("Incomplete full-frame mapping") from error


def map_region(region,mapping):
    validate_mapping(mapping)
    if mapping is None:
        return region
    x,y=mapping["x"],mapping["y"]
    a,b,c,d=region.box
    box=(float(np.interp(a,x["source"],x["model"])),
         float(np.interp(b,y["source"],y["model"])),
         float(np.interp(c,x["source"],x["model"])),
         float(np.interp(d,y["source"],y["model"])))
    return region.model_copy(update={"box":box})


def _mesh(image,size,mapping,inverse):
    axes=[]
    for axis,extent,source_extent in zip((mapping["x"],mapping["y"]),size,image.size):
        output=axis["source" if inverse else "model"]
        source=axis["model" if inverse else "source"]
        nodes=sorted({round(v*extent) for v in output})
        # Evaluate the continuous inverse map at actual raster boundaries.
        # Rounding nearby knots never leaves holes or discards a frame border.
        at=np.interp(np.asarray(nodes)/extent,output,source)*source_extent
        axes.append(list(zip(nodes,at.tolist())))
    x,y=axes;mesh=[]
    for (left,sleft),(right,sright) in pairwise(x):
        for (top,stop),(bottom,sbottom) in pairwise(y):
            mesh.append(((left,top,right,bottom),
                (sleft,stop,sleft,sbottom,sright,sbottom,sright,stop)))
    return image.convert("RGB").transform(tuple(size),Image.Transform.MESH,mesh,
        resample=Image.Resampling.BICUBIC)


def sample_full_frame(source,mapping):
    validate_mapping(mapping)
    if mapping is None:
        return source
    if list(source.size)!=mapping["original_size"]:
        raise ValueError("Sampling source frame size differs from its mapping")
    return _mesh(source,source.size,mapping,False)


def restore_full_frame(generated,size,mapping=None):
    validate_mapping(mapping)
    if mapping is None:
        return generated.convert("RGB").resize(size,Image.Resampling.LANCZOS)
    if list(size)!=mapping["original_size"]:
        raise ValueError("Restoration must preserve the original full frame")
    return _mesh(generated,size,mapping,True)
