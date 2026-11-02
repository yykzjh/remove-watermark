"""Explicit offline tail-refinement experiments, never a recommended default."""
import math
from itertools import pairwise


def tail_sigma_grid(original,profile="native"):
    if profile not in ("native","tail9","tail10","tail12","mid10","mid12","turturbo3"):
        raise ValueError("Unknown explicit sigma experiment")
    values=list(original)
    if (len(values)!=8 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<v<=1 for v in values)
            or any(a<=b for a,b in pairwise(values))):
        raise ValueError("Original descending positive eight-sigma Turbo grid required")
    if profile=="turturbo3":
        # Published DiffSynth Turbo-T2I-3steps LoRA schedule, never bare Turbo.
        return [values[0],values[5],values[7]]
    if profile in {"mid10","mid12"}:
        # Keep every trained node. Only add evaluation points inside the large
        # low-noise Euler intervals; this is an explicitly unvalidated probe.
        added=[(values[-2]+values[-1])/2,values[-1]/2]
        if profile=="mid12":
            added.extend([(values[-3]+values[-2])/2,values[-1]/4])
        return sorted(values+added,reverse=True)
    count={"native":0,"tail9":1,"tail10":2,"tail12":4}[profile]
    return values+[values[-1]*.5**i for i in range(1,count+1)]


def probe_inference_step_options(items):
    """Only explicit homogeneous offline comparison batches can change step count."""
    values=[item.get("_probe_num_inference_steps") for item in items]
    if all(value is None for value in values):
        return {}
    if (not values or any(type(value) is not int or value not in (12,16,20,24,40) for value in values)
            or len(set(values))!=1):
        raise ValueError("Homogeneous explicit offline probe step counts required")
    return {"num_inference_steps":values[0]}


def validate_base_probe(model,steps):
    """Never discard a Turbo checkpoint's saved schedule to imitate a base model."""
    import hashlib
    import json
    from pathlib import Path
    if type(steps) is not int or steps not in (12,16,20,24,40):
        raise ValueError("Explicit base-model probe step count required")
    path=Path(model)/"model_index.json"
    raw=path.read_bytes();config=json.loads(raw)
    if config.get("_class_name")!="QwenImage21Pipeline" or config.get("sample_sigmas"):
        raise ValueError("Base-family checkpoint without a saved Turbo schedule required")
    return {"model_index_sha256":hashlib.sha256(raw).hexdigest(),"requested_steps":steps,
        "sampling":"official_pipeline_uniform_schedule","quality_permission":False}


def configure_turbo_uniform_probe(pipe,steps):
    """Explicit offline comparison, sharing the exact opt-in deployment sampler."""
    from .sampling import configure_uniform_sampling
    return configure_uniform_sampling(pipe,steps)
