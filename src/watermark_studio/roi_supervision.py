"""Crop-relative synthetic targets preserve every non-active overlay in context."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from .batching import dimensions
from .images import context_box
from .prompts import build_focused_prompt, build_small_overlay_prompt, scene_instruction


def make_roi_supervision(edited,clean,box,budget,kind,*,box_is_padded=False):
    if edited.size!=clean.size or kind not in {"panel","logo"} or budget not in (512,768,1024):
        raise ValueError("Matching native images and a supported ROI training budget required")
    x0,y0,x1,y1=box
    if not 0<=x0<x1<=edited.width or not 0<=y0<y1<=edited.height:
        raise ValueError("ROI supervision escaped its native parent")
    px=.008*edited.width+.06*(x1-x0);py=.008*edited.height+.06*(y1-y0)
    expanded=tuple(box) if box_is_padded else (max(0,int(x0-px)),max(0,int(y0-py)),min(edited.width,math.ceil(x1+px)),min(edited.height,math.ceil(y1+py)))
    crop=context_box(expanded,edited.size,.45)
    cropped=edited.crop(crop)
    restored=edited.copy();restored.paste(clean.crop(expanded),expanded)
    target=restored.crop(crop)
    width,height=dimensions(cropped,budget)
    image=cropped.resize((width,height),Image.Resampling.LANCZOS)
    truth=target.resize((width,height),Image.Resampling.LANCZOS)
    local=((expanded[0]-crop[0])*width/(crop[2]-crop[0]),
           (expanded[1]-crop[1])*height/(crop[3]-crop[1]),
           (expanded[2]-crop[0])*width/(crop[2]-crop[0]),
           (expanded[3]-crop[1])*height/(crop[3]-crop[1]))
    prompt_box=local
    radius=math.ceil(3*max(width/(crop[2]-crop[0]),height/(crop[3]-crop[1])))+1
    local=(max(0,int(local[0])-radius),max(0,int(local[1])-radius),min(width,math.ceil(local[2])+radius),min(height,math.ceil(local[3])+radius))
    support=Image.new("L",(width,height),0)
    ImageDraw.Draw(support).rectangle((local[0],local[1],local[2]-1,local[3]-1),fill=255)
    changed=np.any(np.asarray(image)!=np.asarray(truth),axis=2)
    if np.any(changed&(np.asarray(support)==0)):
        raise ValueError("Synthetic ROI target changed context outside its declared support")
    prompt=(build_small_overlay_prompt(prompt_box,(0,0,width,height))+scene_instruction("outdoor")
        if kind=="logo" else build_focused_prompt(kind,prompt_box,(0,0,width,height),scene="outdoor"))
    return image,truth,support,{"crop":list(crop),"expanded":list(expanded),"local":list(prompt_box),"supervision_support":list(local),"resampling_radius":radius,"prompt":prompt}
