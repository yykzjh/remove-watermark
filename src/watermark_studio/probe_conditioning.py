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


def fresh_rim_reports(paths,source_root,source_hashes):
    """Read fresh top certificates, never manufacture full-image legacy bindings."""
    root=Path(source_root).resolve();result={}
    for value in paths:
        path=Path(value).resolve(strict=True)
        if not 0<path.stat().st_size<=16*2**20:
            raise ValueError("Bounded source proof report required")
        raw=path.read_bytes();data=json.loads(raw)
        identity=data.get("weight_identity",{})
        if (data.get("mode")!="top" or data.get("sources_unchanged") is not True or
                data.get("false_approvals")!=0 or identity.get("relation")!="different_weights" or
                any(not re.fullmatch(r"[a-f0-9]{64}",identity.get(k,"")) for k in ("editor_sha256","verifier_sha256")) or
                identity["editor_sha256"]==identity["verifier_sha256"]):
            raise ValueError("Fresh independent top proof with zero negative approvals required")
        for row in data["rows"]:
            source=Path(row["source"]).resolve()
            if not source.is_relative_to(root):
                continue
            name=source.relative_to(root).as_posix();proof=row.get("proof",{})
            if name not in source_hashes or proof.get("approved") is not True:
                continue
            if row.get("source_file_sha256")!=source_hashes[name]:
                raise ValueError("Fresh certificate source file bytes changed")
            result.setdefault(name,[]).append({"proof":proof,"report":str(path),
                "report_sha256":hashlib.sha256(raw).hexdigest(),"proof_revision":data["revision"],
                "weight_identity":identity})
    return result


def refresh_verified_rims(source,old_records,entries):
    """Replace matching frozen geometry only after whole-image and exact-band checks."""
    from .source_panel_rims import full_source_binding, restore_proven_rims
    geometry=("mask","core","header_top_pixel","uncertainty_gap_pixels","restore_box",
              "restore_pixels","guard_region","source_size")
    records=list(old_records);traces=[];seen=set()
    for entry in entries:
        proof=entry["proof"];mask=tuple(proof["mask"])
        if mask in seen:
            raise ValueError("Ambiguous duplicate source card certificate")
        seen.add(mask)
        index=next((i for i,r in enumerate(records) if tuple(r.get("mask",[]))==mask and r.get("approved") is True),None)
        if index is None:
            raise ValueError("A refreshed identity cannot add a new frozen edit region")
        old=records[index]
        if json.dumps({k:old.get(k) for k in geometry},sort_keys=True)!=json.dumps(
                {k:proof.get(k) for k in geometry},sort_keys=True):
            raise ValueError("Fresh source proof changes frozen composition geometry")
        if not full_source_binding(source,proof):
            raise ValueError("Fresh proof does not bind the whole original RGB")
        _restored,count=restore_proven_rims(source,source,[proof])
        if count!=1:
            raise ValueError("Fresh exact-band and independent source proof is invalid")
        records[index]=proof
        traces.append({k:v for k,v in entry.items() if k!="proof"})
    return records,traces
