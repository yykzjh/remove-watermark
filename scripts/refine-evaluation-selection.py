"""Refreeze only source-alias reservations before inference, without any quality labels."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

from watermark_studio.source_fingerprints import FingerprintIndex, fingerprint_file


def replacement_slots(sources, matches):
    """Conservatively remove every flagged incoming source, preserving all other positions."""
    known=set(sources)
    flagged={m["sealed_path"] for m in matches}
    if not flagged.issubset(known):
        raise ValueError("Audit match is outside the immutable original reservation")
    return [i for i,path in enumerate(sources) if path in flagged]


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--sealed",type=Path,required=True)
    p.add_argument("--audit",type=Path,required=True)
    p.add_argument("--target",type=Path,required=True)
    p.add_argument("--extra-history",type=Path,required=True)
    p.add_argument("--seed",type=int,default=2026110202)
    a=p.parse_args()
    root=a.root.resolve(strict=True);data=root/"八标照片集"
    old=a.sealed.resolve(strict=True);audit=a.audit.resolve(strict=True)
    target=a.target.resolve()
    if old.parent!=root or audit.parent!=root or target.parent!=root or target.exists():
        p.error("Separate, new reservation and prior audit under one experiment root required")
    if a.extra_history.stat().st_size>100000:
        p.error("Bounded fingerprints only")
    extra_raw=a.extra_history.read_bytes();extra=json.loads(extra_raw)
    if not isinstance(extra,list) or not 1<=len(extra)<=32:
        p.error("Explicit recovered source fingerprints required")
    selection_raw=(old/"source-selection.json").read_bytes()
    original=json.loads(selection_raw);sources=original["sources"]
    audit_raw=(audit/"report.json").read_bytes();report=json.loads(audit_raw)
    if (report["state"]!="completed" or report["selection_sha256"]!=hashlib.sha256(selection_raw).hexdigest()
            or any(report[k] for k in ("fingerprint_errors","missing_history_paths","unresolved_catalog_paths"))
            or not 1000<=len(sources)<=4096):
        raise ValueError("Complete source-bound fingerprint audit required")
    matches=json.loads((audit/"matches.json").read_text())
    slots=replacement_slots(sources,matches)
    if not slots or len(slots)>len(sources)//10:
        raise ValueError("Only a bounded alias correction, not a new quality selection, is permitted")
    index=FingerprintIndex();frozen={};seen=set()
    for line in (audit/"fingerprints.jsonl").open():
        row=json.loads(line)
        if row["partition"]=="sealed":frozen[row["path"]]=row
        elif row["encoded_sha256"] not in seen:
            index.add(row);seen.add(row["encoded_sha256"])
    for row in extra:
        index.add({**row,"partition":"extra_history"})
    if set(frozen)!=set(sources):
        raise ValueError("Audit must cover the exact original reservation")
    selected=list(sources);source_hashes=dict(original["source_sha256"]);slot_set=set(slots)
    for i,path in enumerate(sources):
        photo=Path(path);relative=photo.resolve(strict=True).relative_to(data).as_posix()
        if photo.is_symlink() or hashlib.sha256(photo.read_bytes()).hexdigest()!=source_hashes[relative]:
            raise ValueError("Original reservation bytes changed")
        if i not in slot_set:
            if index.matches(frozen[path]):
                raise ValueError("Retained source has a newly discovered alias; complete a new audit first")
            index.add(frozen[path])
    # Also exclude discarded originals: they cannot re-enter under another encoding/name.
    for i in slots:index.add(frozen[sources[i]])
    manifest=json.loads((data/".download-state/manifest.json").read_text())
    prefix="/codex-download-1mQf6F6g-20261031/八标照片集/"
    candidates=collections.defaultdict(list)
    for row in manifest:
        raw=row["path"]
        if not raw.startswith(prefix):raise ValueError("Unexpected dataset manifest root")
        relative=Path(raw.removeprefix(prefix))
        if relative.suffix.lower() in {".jpg",".jpeg",".png"}:
            candidates[relative.parts[0]].append(data/relative)
    rng=random.Random(a.seed)
    for town in sorted(candidates):rng.shuffle(candidates[town])
    positions=collections.Counter();skips=collections.Counter();changes=[]
    for slot in slots:
        old_path=Path(sources[slot]);relative=old_path.relative_to(data);town=relative.parts[0]
        chosen=None
        while positions[town]<len(candidates[town]):
            candidate=candidates[town][positions[town]];positions[town]+=1
            if (not candidate.is_file() or candidate.is_symlink()
                    or not candidate.resolve().is_relative_to(data) or not candidate.stat().st_size):
                skips["unavailable"]+=1;continue
            try:fp=fingerprint_file(candidate)
            except (OSError,ValueError):
                skips["decode_or_source_error"]+=1;continue
            if index.matches(fp):
                skips["history_or_reserved_alias"]+=1;continue
            chosen=(candidate,fp);break
        if chosen is None:raise ValueError("Not enough disjoint replacements in the original town")
        candidate,fp=chosen;selected[slot]=str(candidate);index.add(fp)
        source_hashes.pop(relative.as_posix())
        source_hashes[candidate.relative_to(data).as_posix()]=fp["encoded_sha256"]
        changes.append({"slot":slot,"old":str(old_path),"new":str(candidate),"town":town,
            "old_sha256":frozen[str(old_path)]["encoded_sha256"],"new_sha256":fp["encoded_sha256"],
            "reason":"Pre-inference source-alias candidate; no quality labels or model outputs used"})
    before=collections.Counter(Path(s).relative_to(data).parts[0] for s in sources)
    after=collections.Counter(Path(s).relative_to(data).parts[0] for s in selected)
    if before!=after or len(set(source_hashes.values()))!=len(sources):
        raise ValueError("Town composition or encoded uniqueness changed")
    revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    selection={**original,"sources":selected,"source_sha256":source_hashes,"benchmark_revision":revision,
        "refinement_seed":a.seed,"refinement":{"parent_selection_sha256":hashlib.sha256(selection_raw).hexdigest(),
        "audit_report_sha256":hashlib.sha256(audit_raw).hexdigest(),
        "extra_history_sha256":hashlib.sha256(extra_raw).hexdigest(),
        "replaced":len(changes),"unchanged":len(sources)-len(changes),"town_counts_unchanged":True,
        "quality_labels_used":False,"qualification":"A new independent full isolation audit is required"}}
    target.mkdir();inputs=target/"input";inputs.mkdir()
    for path in selected:
        source=Path(path);destination=inputs/source.relative_to(data)
        destination.parent.mkdir(parents=True,exist_ok=True);os.link(source,destination)
    encoded=json.dumps(selection,ensure_ascii=False,sort_keys=True)
    (target/"source-selection.json").write_text(encoded)
    (target/"sources.json").write_text(json.dumps(selected,ensure_ascii=False))
    (target/"alias-refinement.json").write_text(json.dumps({"changes":changes,"skips":dict(skips),
        "town_counts":dict(after),"model_calls":0,"quality_labels_used":False},ensure_ascii=False,indent=2))
    if (old/"source-selection.json").read_bytes()!=selection_raw:
        raise ValueError("Immutable parent selection changed")
    print(json.dumps({"target":str(target),"images":len(selected),"replaced":len(changes),
        "unchanged":len(selected)-len(changes),"selection_sha256":hashlib.sha256(encoded.encode()).hexdigest(),
        "model_calls":0,"skips":dict(skips)}),flush=True)


if __name__=="__main__":main()
