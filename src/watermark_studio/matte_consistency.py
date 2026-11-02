"""Offline physical-consistency cue for correlated translucent panels; never approval."""
from __future__ import annotations

import numpy as np
from PIL import Image


def fit_constant_matte(source,after):
    """Robust S=t*A+(1-t)*F fit; reject unobservable or nonconstant mattes."""
    x=np.asarray(after,dtype=np.float64).reshape(-1,3)
    y=np.asarray(source,dtype=np.float64).reshape(-1,3)
    if x.shape!=y.shape or len(x)<1024:
        return None
    indices=np.linspace(0,len(x)-1,min(len(x),4096)).astype(int)
    x,y=x[indices],y[indices];keep=np.ones(len(x),dtype=bool)
    for _ in range(4):
        ax,ay=x[keep].mean(0),y[keep].mean(0)
        centred=x[keep]-ax;variance=np.square(centred).mean()
        if variance<64:return None
        transmission=float((centred*(y[keep]-ay)).sum()/np.square(centred).sum())
        offset=ay-transmission*ax
        residual=np.square(y-(transmission*x+offset)).mean(1)
        keep=residual<=np.quantile(residual,.6)
    residual=np.square(y-(transmission*x+offset)).mean(1)
    keep=residual<=2.5**2
    if keep.mean()<.6 or not .15<=transmission<=.9:
        return None
    ax,ay=x[keep].mean(0),y[keep].mean(0);centred=x[keep]-ax
    if np.square(centred).mean()<64:return None
    transmission=float((centred*(y[keep]-ay)).sum()/np.square(centred).sum())
    if not .15<=transmission<=.9:return None
    offset=ay-transmission*ax
    error=y[keep]-(transmission*x[keep]+offset)
    rmse=float(np.sqrt(np.square(error).mean()))
    signal=float(np.square(y[keep]-ay).mean())
    r2=1-float(np.square(error).mean())/max(signal,1e-8)
    if rmse>1.5 or r2<.98:return None
    foreground=offset/(1-transmission)
    return {"transmission":transmission,"foreground":foreground.tolist(),
        "rmse":rmse,"r2":r2,"inlier_fraction":float(keep.mean()),
        "impossible_channels":[int(i) for i in range(3) if foreground[i]<-16 or foreground[i]>271]}


def matte_consistency(before,after,regions,resolution=1024):
    """Two separated impossible tiles add a review cue, never remove any veto."""
    if before.size!=after.size:raise ValueError("Aligned complete frames required")
    if resolution not in (512,768,1024):raise ValueError("Bounded comparison budget required")
    scale=min(1,resolution/max(before.size))
    size=(max(1,round(before.width*scale)),max(1,round(before.height*scale)))
    source=np.asarray(before.convert("RGB").resize(size,Image.Resampling.LANCZOS))
    candidate=np.asarray(after.convert("RGB").resize(size,Image.Resampling.LANCZOS))
    panels=sorted([r for r in regions if r.kind=="panel" and
        (r.box[2]-r.box[0])*(r.box[3]-r.box[1])>=.02],
        key=lambda r:(r.box[2]-r.box[0])*(r.box[3]-r.box[1]),reverse=True)
    findings=[];tested=observable=0
    for region in panels:
        l,t,r,b=region.box
        left,top,right,bottom=round(l*size[0])+8,round(t*size[1])+8,round(r*size[0])-8,round(b*size[1])-8
        side=min(128,(right-left)//2,(bottom-top)//2)
        if side<48:continue
        fits=[]
        for y in np.linspace(top,bottom-side,3).astype(int):
            for x in np.linspace(left,right-side,3).astype(int):
                if tested>=18:break
                tested+=1
                fit=fit_constant_matte(source[y:y+side,x:x+side],candidate[y:y+side,x:x+side])
                if fit is None:continue
                observable+=1
                if fit["impossible_channels"]:fits.append(fit|{"tile":[int(x),int(y),int(x+side),int(y+side)]})
        valid=[]
        for first in fits:
            for second in fits:
                if first is second:continue
                a,b=first["tile"],second["tile"]
                separated=max(abs(a[0]-b[0]),abs(a[1]-b[1]))>=side*.5
                common=set(first["impossible_channels"])&set(second["impossible_channels"])
                consistent=any((first["foreground"][channel]>271 and second["foreground"][channel]>271)
                    or (first["foreground"][channel]<-16 and second["foreground"][channel]<-16) for channel in common)
                if separated and consistent:valid=[first,second];break
            if valid:break
        if valid:findings.append({"box":list(region.box),"tiles":valid,
            "reason":"correlated_panel_requires_out_of_gamut_foreground"})
        if tested>=18:break
    return {"method":"constant_translucent_matte_physics_v1","count":len(findings),"findings":findings,
        "tested_tiles":tested,"observable_tiles":observable,"comparison_size":list(size),"quality_permission":False}
