"""Source-only side-band proposals; CV findings never authorize pixel restoration."""
from __future__ import annotations

import numpy as np
from PIL import ImageFilter

from .images import pixel_box
from .source_panel_rims import restore_proven_rims
from .surface_reference import _box


def _runs(values):
    edges=np.diff(np.r_[False,values,False].astype(np.int8))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))


def _header_edges(rgb,core,header_top):
    height,width=rgb.shape[:2]
    left,top,right,bottom=pixel_box(core,(width,height))
    inset=max(8,round((right-left)*.10))
    centre=rgb[header_top:bottom,left+inset:right-inset]
    if min(centre.shape[:2],default=0)<8:
        return None
    blue=((centre[...,2]>90)&(centre[...,2]>centre[...,0]*1.15)&
          (centre[...,2]>centre[...,1]*1.05)).mean(axis=1)
    neutral=((np.ptp(centre,axis=2)<45)&(centre.mean(axis=2)>100)).mean(axis=1)
    limit=min(len(blue)-8,max(16,round((bottom-top)*.65)))
    starts=[i for i in range(8,limit) if blue[i:i+8].max()<.25 and neutral[i:i+8].min()>.65]
    if not starts:
        return None
    end=header_top+starts[0]
    y0,y1=header_top+6,end-6
    if y1-y0<8:
        return None
    margin=max(16,round(width*.03))
    x0,x1=max(0,left-margin),min(width,right+margin)
    rows=rgb[y0:y1,x0:x1]
    blue=(rows[...,2]>90)&(rows[...,2]>rows[...,0]*1.15)&(rows[...,2]>rows[...,1]*1.05)
    positions=[]
    for row in blue:
        row=row.copy()
        # Fill internal glyph holes, never extrapolate beyond observed blue.
        for a,b in _runs(~row):
            if a>0 and b<len(row) and b-a<=min(64,max(3,round((y1-y0)*.75))):
                row[a:b]=True
        runs=[(a,b) for a,b in _runs(row) if .65*(right-left)<=b-a<=1.5*(right-left)
              and a<=(left+right)/2-x0<=b]
        if runs:
            a,b=max(runs,key=lambda q:q[1]-q[0]);positions.append((a+x0,b+x0))
    if len(positions)<.75*len(blue):
        return None
    native=np.asarray(positions);median=np.rint(np.median(native,axis=0)).astype(int)
    support=np.sum(np.abs(native-median)<=3,axis=0)/len(blue)
    if min(support)<.75:
        return None
    return {"left":int(median[0]),"right":int(median[1]),"body_start":end,
            "header_rows":y1-y0,"blue_edge_support":support.tolist()}


def _body_bottom(blurred,header,expected):
    height,_width=blurred.shape[:2]
    left,right=header["left"],header["right"]
    inset=max(12,round((right-left)*.10));left+=inset;right-=inset
    radius=max(20,round(.008*height)+8)
    positions=range(max(header["body_start"]+32,expected-radius,8),min(height-8,expected+radius+1))
    candidates=[]
    for at in positions:
        inside=blurred[at-7:at-4,left:right].mean(axis=0)
        outside=blurred[at+4:at+7,left:right].mean(axis=0)
        if len(inside)<96 or ((np.ptp(inside,axis=1)<45)&(inside.mean(axis=1)>120)).mean()<.65:
            continue
        changes=[];signed=[]
        for lo in range(0,len(inside)-31,32):
            a=np.median(inside[lo:lo+32],axis=0);b=np.median(outside[lo:lo+32],axis=0)
            changes.append(float(np.abs(a-b).mean()));signed.append(float((a-b).mean()))
        contrast=float(np.median(changes))
        if contrast<25 or np.mean(np.asarray(changes)>=20)<.75:
            continue
        sign=1 if np.median(signed)>=0 else -1
        support=float(np.mean(np.asarray(signed)*sign>=15))
        if support<.75:
            continue
        candidates.append({"pixel":at,"contrast":contrast,"sign_support":support,"_segments":changes})
    best=max(candidates,key=lambda x:x["contrast"],default=None)
    if best is None:
        return None
    values=np.asarray([c["_segments"] for c in candidates])
    peaks=np.asarray([c["pixel"] for c in candidates])[np.argmax(values,axis=0)]
    support=float(np.mean(np.abs(peaks-best["pixel"])<=3))
    if support<.75:
        return None
    return {k:v for k,v in best.items() if k!="_segments"}|{"edge_phase_support":support}



def _body_side(blurred,header,bottom,side):
    """Separate body edge evidence: a blue header alone cannot bound a gray card."""
    height,width=blurred.shape[:2]
    start=header["body_start"]+8;end=min(height,bottom)-8
    if end-start<128:
        return None
    view=blurred.transpose(1,0,2)
    expected=header[side]
    if side=="left":
        view=view[::-1];expected=width-expected
    body={"left":start,"right":end,"body_start":0}
    edge=_body_bottom(view,body,expected)
    if edge is None:
        return None
    if side=="left":
        edge=edge|{"pixel":width-edge["pixel"]}
    return edge


def propose_card_side_rims(photo,regions,top_records,feather=12,*,verify_body_sides=False):
    """Find conservative candidate clear strips left/right/below known blue cards.

    Existing top certificates are geometry anchors only. Every returned strip
    still requires a NEW independent digital-card/clear-band proof, OCR veto,
    exact band SHA and complete-source RGB binding before any use.
    """
    if not 0<=feather<=64:
        raise ValueError("Bounded native feather required")
    width,height=photo.size
    rgb=np.asarray(photo.convert("RGB"),dtype=np.float32)
    blurred=np.asarray(photo.convert("RGB").filter(ImageFilter.GaussianBlur(2)),dtype=np.float32)
    proposals=[];seen=set()
    for record in top_records or []:
        if not isinstance(record,dict) or record.get("approved") is not True:
            continue
        mask=_box(record.get("mask"));core=_box(record.get("core"))
        region=next((r for r in regions if r.kind=="panel" and tuple(r.box)==mask),None)
        if (region is None or core is None or not mask[0]<=core[0]<core[2]<=mask[2]
                or not mask[1]<=core[1]<core[3]<=mask[3] or type(record.get("header_top_pixel")) is not int):
            continue
        _restored,count=restore_proven_rims(photo,photo,[record])
        if count!=1:
            continue
        header=_header_edges(rgb,core,record["header_top_pixel"])
        if header is None:
            continue
        ml,mt,mr,mb=pixel_box(mask,photo.size)
        _left,_top,_right,bottom=pixel_box(core,photo.size)
        lower=_body_bottom(blurred,header,bottom)
        side_bodies=({side:_body_side(blurred,header,bottom,side) for side in ("left","right")}
                     if verify_body_sides else {})
        candidates=[]
        for side in ("left","right"):
            body=side_bodies.get(side)
            if verify_body_sides and body is None:
                continue
            edge=(min(header[side],body["pixel"]) if side=="left" else max(header[side],body["pixel"])) if body else header[side]
            native=((max(0,ml-feather),max(0,mt-feather),edge-8,min(height,mb+feather))
                    if side=="left" else
                    (edge+8,max(0,mt-feather),min(width,mr+feather),min(height,mb+feather)))
            candidates.append((side,edge,native))
        if lower:
            candidates.append(("bottom",lower["pixel"],(max(0,ml-feather),lower["pixel"]+8,min(width,mr+feather),min(height,mb+feather))))
        for side,edge,native in candidates:
            l,t,r,b=native
            if not (0<=l<r<=width and 0<=t<b<=height) or min(r-l,b-t)<8:
                continue
            box=[v/s for v,s in zip(native,(width,height,width,height))]
            if any(other is not region and max(box[0],other.box[0])<min(box[2],other.box[2])
                   and max(box[1],other.box[1])<min(box[3],other.box[3]) for other in regions):
                continue
            key=(mask,side)
            if key in seen:
                continue
            seen.add(key)
            proposals.append({"protocol":"source_card_side_rim_v1","side":side,
                "mask":list(mask),"core":list(core),"header_top_pixel":record["header_top_pixel"],
                "card_edge_pixel":edge,"uncertainty_gap_pixels":8,"source_size":list(photo.size),
                "restore_pixels":list(native),"restore_box":box,"approved":False,
                "source_geometry":{"blue_header":header,"body_sides":side_bodies,"bottom":lower},
                "anchor_band_sha256":record["source_rim_sha256"]})
    return proposals[:12]
