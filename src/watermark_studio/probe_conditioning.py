"""Immutable observed-scene conditioning for offline whole-photo probes."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .controlnet_probe import bounded_caption


def frozen_source_captions(state,job,source_hashes):
    if not re.fullmatch(r"[a-f0-9]{32}",job):
        raise ValueError("Explicit completed caption job required")
    folder=Path(state)/job
    summary_path=folder/"summary.json";rows_path=folder/"results.jsonl"
    summary_bytes=summary_path.read_bytes();rows_bytes=rows_path.read_bytes()
    summary=json.loads(summary_bytes)
    rows=[json.loads(line) for line in rows_bytes.decode().splitlines()]
    if (summary.get("state")!="completed" or len(rows)!=summary.get("total") or
            len({r["file"] for r in rows})!=len(rows)):
        raise ValueError("Incomplete or duplicate frozen caption records")
    indexed={r["file"]:r for r in rows}
    result={}
    for name,sha in source_hashes.items():
        if summary.get("sources_sha256",{}).get(name)!=sha:
            raise ValueError("Caption must be bound to the exact same original image SHA")
        row=indexed.get(name,{})
        observation=row.get("controlnet_probe",{}).get("source_caption")
        if not row.get("ok") or not isinstance(observation,str):
            raise ValueError("A successful bounded source observation is required")
        text,audit=bounded_caption(observation,80)
        result[name]={"text":text,"conditioning_audit":audit,"caption_job":job,
            "source_file_sha256":sha,"caption_summary_sha256":hashlib.sha256(summary_bytes).hexdigest(),
            "caption_results_sha256":hashlib.sha256(rows_bytes).hexdigest(),
            "quality_permission":False}
    return result


def observed_scene_prompt(prompt,observation):
    return (prompt+"\nObserved original scene (conditioning only; not a quality verdict): "+
        observation["text"]+"\nKeep the scene materials and geometry as observed. "
        "This description does not authorize adding unseen structures or changing real objects.")
