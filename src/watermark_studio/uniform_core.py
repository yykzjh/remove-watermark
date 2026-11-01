"""Conservative new neutral paint-block cue; never a removal approval."""
import numpy as np
from PIL import Image

from .surface_reference import _box


def uniform_core_fill(after, regions, cores, resolution=1024):
    size=(max(1,round(after.width*min(1,resolution/max(after.size)))),
          max(1,round(after.height*min(1,resolution/max(after.size)))))
    rgb=np.asarray(after.convert("RGB").resize(size,Image.Resampling.LANCZOS),dtype=np.float32)
    findings=[]
    for item in cores or []:
        if not isinstance(item,dict):
            continue
        mask,core=_box(item.get("mask")),_box(item.get("core"))
        if (mask is None or core is None or not any(r.kind=="panel" and
                all(abs(a-b)<1e-6 for a,b in zip(r.box,mask)) for r in regions)
                or not (mask[0]<=core[0]<core[2]<=mask[2] and mask[1]<=core[1]<core[3]<=mask[3])):
            continue
        l,t,r,b=[round(v*s) for v,s in zip(core,(*size,*size))]
        if min(r-l,b-t)<48 or (r-l)*(b-t)<.006*size[0]*size[1]:
            continue
        inner=rgb[t+8:b-8,l+8:r-8]
        pixels=inner.reshape(-1,3)
        colour=np.median(pixels,axis=0)
        spread=np.quantile(pixels,.95,axis=0)-np.quantile(pixels,.05,axis=0)
        uniform=float(np.mean(np.max(np.abs(pixels-colour),axis=1)<=2))
        # Ignore saturated whites/blacks and coloured physical/digital surfaces.
        if colour.min()<=7 or colour.max()>=248 or np.ptp(colour)>8 or spread.max()>2 or uniform<.97:
            continue
        sides=[]
        for side,at in [("left",l),("right",r),("top",t),("bottom",b)]:
            vertical=side in {"left","right"}
            low=side in {"left","top"}
            lo,hi=(t+(b-t)//5,b-(b-t)//5) if vertical else (l+(r-l)//5,r-(r-l)//5)
            first,last=(at-10,at-4) if low else (at+4,at+10)
            if first<0 or last>(size[0] if vertical else size[1]) or hi-lo<32:
                continue
            outside=(rgb[lo:hi,first:last] if vertical else rgb[first:last,lo:hi]).reshape(-1,3)
            texture=float(np.mean(np.std(outside,axis=0)))
            difference=float(np.mean(np.abs(outside-colour)))
            if texture>=8 and difference>=8:
                sides.append({"side":side,"outside_spread":round(texture,3),
                              "outside_colour_difference":round(difference,3)})
        names={v["side"] for v in sides}
        if names&{"left","right"} and names&{"top","bottom"}:
            findings.append({"box":list(core),"colour":colour.tolist(),"uniform_fraction":uniform,
                             "spread":spread.tolist(),"adjacent_exterior_evidence":sides})
    return {"method":"new_uniform_neutral_core_v1","count":len(findings),"findings":findings}
