"""CPU-only scope diagnostic; never edits images, approvals, or deployment defaults."""
from __future__ import annotations

import argparse
import json
import re
import runpy
from pathlib import Path

from watermark_studio.config import Region
from watermark_studio.images import load_photo
from watermark_studio.quality import repair_texture_warnings
from watermark_studio.source_panel_rims import effective_rim_scope, restore_proven_rims


def validated_bands(source, regions, cores, records):
    scoped, tight_cores, invalid=effective_rim_scope(source,regions,cores,records)
    if invalid:
        raise ValueError("Invalid source-proof record; no diagnostic exemption allowed")
    masks={tuple(r.box) for r in regions}
    boxes=[]
    for record in records:
        if not isinstance(record,dict) or record.get("approved") is not True:
            continue
        restored,count=restore_proven_rims(source,source,[record])
        del restored
        if count!=1 or tuple(record["mask"]) not in masks:
            raise ValueError("Missing source SHA-bound clear-band proof")
        boxes.append(record["restore_box"])
    return scoped,tight_cores,boxes


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--source-job")
    p.add_argument("--pair-manifest",type=Path)
    p.add_argument("--controls",type=Path)
    p.add_argument("--tag",required=True)
    a=p.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+",a.tag) or bool(a.source_job)==bool(a.pair_manifest):
        p.error("One immutable source selector and simple new tag required")
    h=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    state=Path.home()/".local/state/watermark-studio"
    cases=h["manifest_cases"](a.pair_manifest) if a.pair_manifest else h["frozen_job_cases"](state,a.source_job)
    if a.controls:
        cases.extend(h["known_control_cases"](a.controls))
    hashes=h["file_hashes"](cases)
    results=[]
    for case in cases:
        source=load_photo(case["source"])[0];output=load_photo(case["output"])[0]
        regions=[Region.model_validate(r) for r in case["regions"]]
        cores=case.get("source_cores",[])
        records=case.get("source_rim_protection",[])
        scoped,tight_cores,bands=validated_bands(source,regions,cores,records)
        provenance="source_sha_and_independent_clear_rim"
        if case.get("cohort")=="controls" and cores:
            # Ground-truth fixture overlay rectangle, no fabricated VLM proof.
            # Only the band strictly above its actual known opaque footprint.
            c=cores[0]["core"];m=regions[0].box
            top=max(m[1],c[1]-16/source.height)
            bands=[[c[0],top,c[2],c[1]]]
            scoped=[Region(kind=regions[0].kind,box=(m[0],c[1],m[2],m[3]))]
            tight_cores=[{"mask":list(scoped[0].box),"core":c}]
            provenance="synthetic_ground_truth_clear_source_strip"
        kwargs={"before":source,"core_boxes":tight_cores}
        old=repair_texture_warnings(output,scoped,**kwargs)
        new=repair_texture_warnings(output,scoped,_source_rim_reference_boxes=bands,**kwargs)
        results.append({"file":case["file"],"cohort":case.get("cohort"),"expected":case.get("expected"),
            "old_count":old["count"],"new_count":new["count"],"bands":bands,
            "reference_provenance":provenance,"reference_support":new["source_reference_support"],
            "changed":bool(old["count"])!=bool(new["count"])})
    if hashes!=h["file_hashes"](cases):
        raise RuntimeError("Original or output changed during read-only diagnostic")
    target=Path(".run")/a.tag
    target.mkdir(exist_ok=False)
    (target/"results.json").write_text(json.dumps(results,ensure_ascii=False,indent=2))
    changed=[{"file":r["file"],"old":r["old_count"],"new":r["new_count"],"expected":r["expected"]}
             for r in results if r["changed"]]
    summary={"images":len(results),"changed":changed,
        "new_defect_clearances":[r["file"] for r in results if r["changed"] and r["expected"]=="review" and not r["new_count"]],
        "file_sha256":hashes,"images_unchanged":True,"approvals_changed":False,"live_quality_policy_changed":False}
    (target/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!="file_sha256"},ensure_ascii=False))


if __name__=="__main__":
    main()
