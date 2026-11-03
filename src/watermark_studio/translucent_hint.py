"""Experimental full-frame VAE hints, never a certified recovery or QA exemption."""
from __future__ import annotations

import hashlib
import math

import numpy as np
from PIL import Image, ImageFilter

from .config import Region
from .images import pixel_box
from .source_panel_rims import effective_rim_scope, full_source_binding


def translucent_body_hint(source, records, transmission=.55, *, foreground=255.,
                          _dense_transmission_experiment=False):
    """Boost transmitted structure in a small interior of proven digital cards.

    Neutral foreground and transmission are ASSUMPTIONS. Keep the original full
    photo for semantic encoding/QA; the hint must never become final pixels.
    No glyph synthesis/removal, no extra encoder call, no edit outside the body.
    """
    minimum=.10 if _dense_transmission_experiment else .35
    if not math.isfinite(transmission) or not minimum <= transmission <= .85:
        raise ValueError(f"Experimental transmission must be finite in{minimum}..0.85")
    if not math.isfinite(foreground) or not 190<=foreground<=255:
        raise ValueError("Experimental neutral foreground must be finite in190..255")
    if not any(isinstance(r,dict) and r.get("approved") is True for r in records or []):
        return source, []
    fingerprint=hashlib.sha256(source.convert("RGB").tobytes()).hexdigest()
    original = np.asarray(source.convert("RGB"), dtype=np.float32)
    result = original.copy()
    occupied = np.zeros(original.shape[:2], dtype=bool)
    audits = []
    for record in records or []:
        if not isinstance(record, dict) or record.get("approved") is not True:
            continue
        audit = {"applied": False, "method": ("assumed_translucent_condition_v2_dense"
                    if _dense_transmission_experiment or foreground!=255 else "assumed_translucent_condition_v1"),
                 "transmission_assumption": transmission,
                 "foreground_assumption": [255,255,255] if foreground==255 else [foreground]*3,
                 "quality_permission": False}
        try:
            if not full_source_binding(source,record,fingerprint=fingerprint):
                raise ValueError("Body conditioning requires the full original RGB proof")
            region = Region(kind="panel", box=record["mask"])
            cores = [{"mask": record["mask"], "core": record["core"]}]
            _, _, invalid = effective_rim_scope(source, [region], cores, [record])
            if invalid:
                raise ValueError("Original digital-card proof mismatch")
            l, _, r, b = pixel_box(record["core"], source.size)
            top = record["header_top_pixel"]
            inset = max(12, round((r-l)*.26))
            l, r = l+inset, r-inset  # Only a central interior, never the card sides.
            if r-l < 64 or b-top < 48:
                raise ValueError("Body too small")
            patch = original[top:b, l:r]
            blue = ((patch[...,2] > 90) & (patch[...,2] > patch[...,0]*1.15)
                    & (patch[...,2] > patch[...,1]*1.05)).mean(axis=1)
            neutral = ((np.ptp(patch, axis=2) < 45) & (patch.mean(axis=2) > 110)).mean(axis=1)
            starts = [i for i in range(12, min(len(blue)-12, round((b-top)*.7)))
                      if blue[i:i+12].max() < .25 and neutral[i:i+12].min() > .65]
            if not starts:
                raise ValueError("No stable neutral card body")
            start = top+starts[0]
            t, end = start+8, start+round((b-start)*.60)
            if end-t < 32 or occupied[t:end, l:r].any():
                raise ValueError("Short or overlapping hint interior")
            patch = original[t:end, l:r]
            # Text remains for the model to remove. Exclude ink and its edges
            # ONLY from the test for real transmitted variation.
            ink = Image.fromarray((patch.mean(axis=2) < 100).astype(np.uint8)*255)
            background = np.asarray(ink.filter(ImageFilter.MaxFilter(9))) == 0
            if background.mean() < .40 or float(patch.mean(axis=2)[background].std()) < 4:
                raise ValueError("No reliable transmitted variation")
            recovered = (patch-foreground*(1-transmission))/transmission
            clipped = np.any((recovered < 0) | (recovered > 255), axis=2)
            fraction = float(clipped[background].mean())
            if fraction > .10:
                raise ValueError("Contrast assumption clips too much visible background")
            yy, xx = np.indices((end-t, r-l))
            distance = np.minimum.reduce([xx, yy, r-l-1-xx, end-t-1-yy])
            weight = np.clip(distance/12, 0, 1).astype(np.float32)
            weight = (weight*weight*(3-2*weight))[...,None]
            result[t:end,l:r] = patch*(1-weight)+np.clip(recovered,0,255)*weight
            occupied[t:end,l:r] = True
            audit.update(applied=True, body_pixels=[l,t,r,end], clipped_background_fraction=round(fraction,6))
        except (KeyError, ValueError, TypeError) as error:
            audit["skip_reason"] = str(error)[:160]
        audits.append(audit)
    return (Image.fromarray(np.rint(result).astype(np.uint8)) if any(r["applied"] for r in audits)
            else source), audits


def roi_body_conditioning(source, box, transmission=.45):
    """Offline VAE-only assumption, never final recovered pixels or a QA proof."""
    if not math.isfinite(transmission) or not .35 <= transmission <= .85:
        raise ValueError("Finite bounded conditioning transmission required")
    if len(box) != 4 or any(type(v) is not int for v in box):
        raise ValueError("Native integer source ROI required")
    l,t,r,b=box
    if not 0 <= l < r <= source.width or not 0 <= t < b <= source.height:
        raise ValueError("Conditioning ROI escaped source")
    audit={"method":"assumed_neutral_roi_vae_only_v1","applied":False,
           "transmission_assumption":transmission,"foreground_assumption":255,
           "quality_permission":False,"final_pixels":False}
    if min(r-l,b-t) < 64 or (r-l)*(b-t) > .4*source.width*source.height:
        return source,audit|{"reason":"unsupported_extent"}
    rgb=np.asarray(source.convert("RGB"),dtype=np.float32)
    inset=max(8,round((r-l)*.08))
    patch=rgb[t:b,l+inset:r-inset]
    median=np.median(patch,axis=1)
    neutral=(np.ptp(median,axis=1)<40)&(median.mean(axis=1)>150)
    runs=[]
    start=None
    for index,good in enumerate(neutral.tolist()+[False]):
        if good and start is None:
            start=index
        elif not good and start is not None:
            runs.append((start,index))
            start=None
    if not runs:
        return source,audit|{"reason":"no_neutral_body"}
    top,end=max(runs,key=lambda span:span[1]-span[0])
    if end-top < max(48,round((b-t)*.22)):
        return source,audit|{"reason":"short_neutral_body"}
    left,right=l+inset,r-inset
    top,end=t+top+4,t+end-4
    patch=rgb[top:end,left:right]
    eligible=(patch.min(axis=2)>150)&(np.ptp(patch,axis=2)<55)
    if eligible.mean()<.45 or patch.mean(axis=2)[eligible].std()<4:
        return source,audit|{"reason":"no_transmitted_variation"}
    recovered=(patch-255*(1-transmission))/transmission
    clipped=np.any((recovered<0)|(recovered>255),axis=2)
    if clipped[eligible].mean()>.15:
        return source,audit|{"reason":"assumption_clips_background"}
    yy,xx=np.indices((end-top,right-left))
    distance=np.minimum.reduce([xx,yy,right-left-1-xx,end-top-1-yy])
    weight=np.clip(distance/12,0,1)
    weight=(weight*weight*(3-2*weight))[...,None]
    output=rgb.copy()
    output[top:end,left:right]=patch*(1-weight)+np.clip(recovered,0,255)*weight
    audit.update(applied=True,body_pixels=[left,top,right,end],
                 reason="unverified_conditioning_assumption_only")
    return Image.fromarray(np.rint(output).astype(np.uint8)),audit
