"""Explicit offline tail-refinement experiments, never a recommended default."""
import math
from itertools import pairwise


def tail_sigma_grid(original,profile="native"):
    if profile not in ("native","tail9","tail10","tail12","mid10","mid12"):
        raise ValueError("Unknown explicit sigma experiment")
    values=list(original)
    if (len(values)!=8 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<v<=1 for v in values)
            or any(a<=b for a,b in pairwise(values))):
        raise ValueError("Original descending positive eight-sigma Turbo grid required")
    if profile in {"mid10","mid12"}:
        # Keep every trained node. Only add evaluation points inside the large
        # low-noise Euler intervals; this is an explicitly unvalidated probe.
        added=[(values[-2]+values[-1])/2,values[-1]/2]
        if profile=="mid12":
            added.extend([(values[-3]+values[-2])/2,values[-1]/4])
        return sorted(values+added,reverse=True)
    count={"native":0,"tail9":1,"tail10":2,"tail12":4}[profile]
    return values+[values[-1]*.5**i for i in range(1,count+1)]
