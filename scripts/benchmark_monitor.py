"""Validate a read-only benchmark monitor attachment without submitting a job."""
from __future__ import annotations

import copy


def attachment_health(stored,live,job,options,selection_sha256):
    if not job or stored.get("job")!=job or (live.get("job") or {}).get("id")!=job:
        raise ValueError("Cannot attach a monitor to another job")
    if stored.get("options")!=options or stored.get("selection_sha256")!=selection_sha256:
        raise ValueError("Frozen benchmark request or cohort differs")
    original=stored["health"]
    for key in ("optimization","quality_weight_identity","code_revision","gpus","backend"):
        if live.get(key)!=original.get(key):
            raise ValueError("Live benchmark deployment differs: "+key)
    return copy.deepcopy(original)
