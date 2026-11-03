"""Explicit deployment sampling; checkpoint files are never mutated."""
from __future__ import annotations

import math
from itertools import pairwise

UNIFORM_STEPS=(12,16,20,24,40)


def configure_uniform_sampling(pipe,steps):
    if type(steps) is not int or steps not in UNIFORM_STEPS:
        raise ValueError("Explicit supported uniform sampling step count required")
    original=list(getattr(pipe.config,"sample_sigmas",None) or [])
    if (len(original)!=8 or any(isinstance(v,bool) or not isinstance(v,(int,float))
            or not math.isfinite(v) or not 0<v<=1 for v in original)
            or any(a<=b for a,b in pairwise(original))):
        raise ValueError("Original descending positive eight-sigma Turbo grid required")
    pipe.register_to_config(sample_sigmas=None)
    return {"requested_steps":steps,"original_sigmas":original,
        "sampling":"official_pipeline_uniform_schedule_on_turbo_weights",
        "checkpoint_modified":False,"quality_permission":False}


def runtime_step_options(pipe,explicit_options=None):
    options=dict(explicit_options or {})
    steps=getattr(pipe,"_watermark_uniform_steps",0)
    if type(steps) is not int or steps not in (0,*UNIFORM_STEPS):
        raise ValueError("Invalid deployed uniform step count")
    if steps:
        if getattr(pipe.config,"sample_sigmas",None):
            raise ValueError("Checkpoint sigmas would override the deployed uniform step count")
        if options and options!={"num_inference_steps":steps}:
            raise ValueError("A probe cannot override the deployed sampling schedule")
        options={"num_inference_steps":steps}
    return options


def verify_sampling_steps(pipe,options):
    requested=options.get("num_inference_steps")
    timesteps=getattr(getattr(pipe,"scheduler",None),"timesteps",None)
    actual=len(timesteps) if timesteps is not None else None
    if requested is not None and actual!=requested:
        raise RuntimeError("Actual denoising step count differs from the explicit sampling request")
    return {**getattr(pipe,"_watermark_sampling_info",{}),
            "actual_steps":actual,"uniform_steps":getattr(pipe,"_watermark_uniform_steps",0)}
