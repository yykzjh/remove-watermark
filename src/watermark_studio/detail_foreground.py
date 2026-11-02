"""Synthetic foreground challenges and source-only detail supports; never quality approval."""
from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from .images import pixel_box
from .transmitted_detail import neutral_detail_candidates

STYLES=("black","gray","faint","tiny","mixed","grid")


def encoded_photo(image,quality):
    if quality==0:return image.convert("RGB")
    stream=io.BytesIO();image.convert("RGB").save(stream,format="JPEG",quality=quality)
    stream.seek(0)
    with Image.open(stream) as photo:return photo.convert("RGB")


def foreground_challenge(clean,font_path,style,seed,*,quality=0):
    from PIL import ImageFont

    if style not in (*STYLES,"none") or quality not in (0,95):
        raise ValueError("Explicit bounded foreground challenge style/encoding required")
    rng=random.Random(seed);clean=clean.convert("RGB");w,h=clean.size
    box=(round(.08*w),round(.42*h),round(.78*w),round(.95*h))
    l,t,r,b=box
    if r-l<120 or b-t<100:raise ValueError("Native challenge body is too small")
    alpha=rng.choice([140,184,216]);color=rng.choice([220,230,238])
    plate=Image.new("RGBA",clean.size,(0,0,0,0))
    ImageDraw.Draw(plate).rectangle((l,t,r-1,b-1),fill=(color,color,color,alpha))
    neutral=Image.alpha_composite(clean.convert("RGBA"),plate)
    ink=Image.new("RGBA",clean.size,(0,0,0,0));draw=ImageDraw.Draw(ink)
    normal=max(14,min(36,(b-t)//7))
    def font(size):
        return (ImageFont.truetype(str(font_path),size) if font_path
            else ImageFont.load_default(size=size))
    if style!="none":
        draw.text((l+10,t+8),"施工区域: 金溪镇",font=font(normal),fill=(0,0,0,255))
        lines=["拍摄时间: 2025.07.28 09:10","今日水印  相机  真实时间","防伪 WE130 T4R9MB WT"]
        for i,text in enumerate(lines):
            size=max(9,normal//2) if style=="tiny" or (style=="mixed" and i==2) else normal
            tone,opacity={"black":(0,255),"gray":(178,255),"faint":(0,24),
                "tiny":(190,96),"mixed":(165,48),"grid":(0,20)}[style]
            draw.text((l+10,t+8+(i+1)*normal*1.4),text,font=font(size),
                fill=(tone,tone,tone,opacity))
        if style=="grid":
            for y in range(t+normal*2,b-6,13):
                draw.line((l+6,y,r-7,y),fill=(170,170,170,28),width=1)
            for x in range(l+6,r-6,17):
                draw.line((x,t+normal*2,x,b-7),fill=(175,175,175,24),width=1)
    clipping=Image.new("L",clean.size,0)
    ImageDraw.Draw(clipping).rectangle((l,t,r-1,b-1),fill=255)
    ink.putalpha(ImageChops.multiply(ink.getchannel("A"),clipping))
    source=encoded_photo(Image.alpha_composite(neutral,ink),quality)
    counterfactual=encoded_photo(neutral,quality)
    # Ground truth is only an independent diagnostic, never a correction of support.
    difference=np.max(np.abs(np.asarray(source).astype(np.int16)-
        np.asarray(counterfactual).astype(np.int16)),axis=2)
    foreground=difference>1
    metadata={"style":style,"seed":seed,"quality":quality,"body_box":list(box),
        "true_transmission":1-alpha/255,"neutral_foreground":color,"font_size":normal,
        "truth":"encoded_source_vs_identically_encoded_constant_body_counterfactual",
        "quality_permission":False}
    return source,counterfactual,foreground,metadata


def detail_support(source,body_box,detected,*,profile="ocr2"):
    """Neither pixels nor an empty OCR result certifies a neutral foreground."""
    if profile not in ("pixel","ocr2","ocr8"):raise ValueError("Unknown support profile")
    w,h=source.size;l,t,r,b=body_box
    if not (0<=l<r<=w and 0<=t<b<=h):raise ValueError("Body box escaped its complete source")
    support=np.zeros((h,w),dtype=bool);support[t:b,l:r]=True
    if detected.get("status")!="text_present":
        return support & False,{"reason":"text_anchor_not_localized","quality_permission":False}
    boxes=[v for row in detected.get("regions",[]) for v in row.get("textlike_boxes",[]) if v]
    if not boxes:return support & False,{"reason":"missing_text_geometry","quality_permission":False}
    support &= neutral_detail_candidates(source)
    if profile!="pixel":
        margin={"ocr2":2,"ocr8":8}[profile]
        for value in boxes:
            x0,y0,x1,y1=pixel_box(value,source.size)
            support[max(0,y0-margin):min(h,y1+margin),max(0,x0-margin):min(w,x1+margin)]=False
    return support,{"profile":profile,"ocr_boxes":len(boxes),"quality_permission":False,
        "reason":"source_only_candidate_not_certificate"}
