"""Read-only provenance helpers for held-out image evaluation."""
import json
import re


def exclusion_sources(root, base, explicit):
    """Explicit benchmark arguments must never replace the old development exclusions."""
    automatic=[base.name]
    old128=root/"first-pass-holdout-neutral-v1"
    if (old128/"sources.json").is_file():
        automatic.append(old128.name)
    seen=set()
    for name in dict.fromkeys([*automatic,*explicit]):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+",name):
            raise ValueError("Invalid excluded benchmark name")
        values=json.loads((root/name/"sources.json").read_text())
        if not isinstance(values,list) or any(not isinstance(v,str) for v in values):
            raise ValueError("Benchmark provenance must contain source path strings")
        seen.update(values)
    return seen,automatic
