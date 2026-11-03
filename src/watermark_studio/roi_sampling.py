"""Source-geometry-only ROI sampling budgets; no output feedback or extra candidates."""
from __future__ import annotations

import math
from contextlib import contextmanager
from itertools import pairwise

from .sampling import UNIFORM_STEPS, configure_uniform_sampling, original_turbo_grid


def source_area_fraction(boxes,size):
    width,height=size
    if width<=0 or height<=0 or not boxes:
        raise ValueError("Original geometry and at least one edit footprint required")
    for box in boxes:
        if len(box)!=4 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in box):
            raise ValueError("Finite source pixel coordinates required")
        a,b,c,d=box
        if not 0<=a<c<=width or not 0<=b<d<=height:
            raise ValueError("ROI must remain inside the original photograph")
    # Exact rectangle union: overlapping grouped members must not inflate cost.
    xx=sorted({b[i] for b in boxes for i in (0,2)})
    area=0.
    for left,right in pairwise(xx):
        spans=sorted((b[1],b[3]) for b in boxes if b[0]<right and b[2]>left)
        total=0.;end=-1.
        for top,bottom in spans:
            total+=max(0,bottom-max(top,end));end=max(end,bottom)
        area+=(right-left)*total
    return area/(width*height)


def prepare_roi_sampling(edits,config):
    if not config.roi_steps:
        return
    for edit in edits:
        if edit["state"]["whole_generation"]:
            fraction=None;selected=0
        else:
            members=edit.get("_roi_members",[edit])
            if any(m["state"] is not edit["state"] for m in members):
                raise ValueError("ROI sampling cannot group different source photographs")
            fraction=source_area_fraction([m["box"] for m in members],edit["state"]["source"].size)
            selected=config.roi_steps if fraction>=config.roi_step_min_area else 0
        edit["_roi_uniform_steps"]=selected
        edit["_roi_sampling_route"]={"policy":"source_area_v1","source_area_fraction":
            round(fraction,9) if fraction is not None else None,
            "min_area":config.roi_step_min_area,"selected_uniform_steps":selected,
            "whole_generation_ignored":edit["state"]["whole_generation"]}


def configure_roi_sampling(pipe,steps,min_area):
    if type(steps) is not int or steps not in UNIFORM_STEPS or not math.isfinite(min_area) or not 0<=min_area<=1:
        raise ValueError("Supported ROI steps and finite source-area threshold required")
    if getattr(pipe,"_watermark_uniform_steps",0):
        raise ValueError("Global and area-routed schedules are mutually exclusive")
    original_turbo_grid(pipe)
    pipe._watermark_roi_steps=steps
    pipe._watermark_roi_min_area=min_area


@contextmanager
def roi_sampling_scope(pipe,items):
    enabled=getattr(pipe,"_watermark_roi_steps",0)
    if not enabled:
        yield
        return
    if not items or any("_roi_uniform_steps" not in item for item in items):
        raise ValueError("Prepare source-area selection before running a routed batch")
    selections={item["_roi_uniform_steps"] for item in items}
    if len(selections)!=1 or not selections<={0,enabled}:
        raise ValueError("A diffusion batch must have one compatible sampling schedule")
    selected=next(iter(selections))
    if not selected:
        original_turbo_grid(pipe)
        if getattr(pipe,"_watermark_uniform_steps",0):
            raise ValueError("Native ROI schedule was not restored")
        yield
        return
    original=list(original_turbo_grid(pipe))
    old_steps=getattr(pipe,"_watermark_uniform_steps",0)
    old_info=getattr(pipe,"_watermark_sampling_info",{})
    try:
        info=configure_uniform_sampling(pipe,selected)
        pipe._watermark_uniform_steps=selected
        pipe._watermark_sampling_info={**info,"sampling":"source_area_routed_uniform_schedule"}
        yield
    finally:
        pipe.register_to_config(sample_sigmas=original)
        pipe._watermark_uniform_steps=old_steps
        pipe._watermark_sampling_info=old_info
