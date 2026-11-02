"""Source-role-preserving curriculum and explicit static Euler training grids."""
from __future__ import annotations

import hashlib
import math
import random
import struct
from itertools import pairwise


def crop_groups(rows,metadata,split):
    lookup={r["id"]:r for r in metadata}
    if len(lookup)!=len(metadata) or len({r["id"] for r in rows})!=len(rows):
        raise ValueError("Unique dataset and cache identifiers required")
    groups={"lower":[],"parent":[]}
    for row in rows:
        source=lookup.get(row["id"])
        if (source is None or row["split"]!=split or source["split"]!=split
                or row["source_original_sha256"]!=source["source_original_sha256"]):
            raise ValueError("Cached source identity or source role differs from dataset")
        groups["lower" if source.get("training_crop_kind")=="lower" else "parent"].append(row)
    if not all(groups.values()):raise ValueError("Both fresh lower and parent crops required")
    return groups


def lower_balanced_cycle(rows,metadata,seed):
    """Alternate each group exactly; shuffle without replacement within a group."""
    groups=crop_groups(rows,metadata,"train");rng=random.Random(seed)
    def iterate():
        queues={key:[] for key in groups};positions={key:0 for key in groups}
        while True:
            for key in ("lower","parent"):
                if positions[key]>=len(queues[key]):
                    queues[key]=list(groups[key]);rng.shuffle(queues[key]);positions[key]=0
                value=queues[key][positions[key]];positions[key]+=1
                yield value
    return iterate()


def lower_balanced_validation(rows,metadata,limit):
    groups=crop_groups(rows,metadata,"validation")
    if type(limit) is not int or limit<4:raise ValueError("Bounded validation size required")
    need={"lower":limit//2,"parent":limit-limit//2}
    if any(len(groups[key])<need[key] for key in need):
        raise ValueError("Insufficient independent validation crops for balanced requested limit")
    selected=[]
    for key in ("lower","parent"):
        selected.extend(sorted(groups[key],key=lambda r:hashlib.sha256(r["id"].encode()).hexdigest())[:need[key]])
    return selected


def explicit_training_sigmas(profile,scheduler_config,model_config):
    """Match the actual fixed Euler nodes; no checkpoint configuration is modified."""
    if profile=="continuous":return None
    if profile not in ("turbo8","uniform12","uniform24","uniform40"):
        raise ValueError("Explicit known training sigma profile required")
    if (scheduler_config.get("use_dynamic_shifting",False) or scheduler_config.get("shift",1)!=1
            or any(scheduler_config.get(key,False) for key in
                ("invert_sigmas","use_karras_sigmas","use_exponential_sigmas","use_beta_sigmas"))):
        raise ValueError("Explicit grids require the verified static shift-one FlowMatch Euler configuration")
    if scheduler_config.get("_class_name")!="FlowMatchEulerDiscreteScheduler":
        raise ValueError("Pinned FlowMatch Euler scheduler required")
    if profile=="turbo8":
        values=model_config.get("sample_sigmas",[])
        if (len(values)!=8 or any(type(v) not in (float,int) or not math.isfinite(v) or not 0<v<=1 for v in values)
                or any(a<=b for a,b in pairwise(values))):
            raise ValueError("Exact descending original Turbo eight-node schedule required")
    else:
        count=int(profile.removeprefix("uniform"));values=[1-index/count for index in range(count)]
    return [struct.unpack("f",struct.pack("f",value))[0] for value in values]
