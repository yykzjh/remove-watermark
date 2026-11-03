"""Offline source-bound tone-grid ablation on existing first-generation outputs."""
from __future__ import annotations

import argparse
import collections
import json
import runpy
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--target",type=Path,required=True)
    p.add_argument("--resolution",type=int,choices=[512,768,1024],required=True)
    a=p.parse_args()
    assert not a.target.exists()
    from watermark_studio.config import Region
    from watermark_studio.images import load_photo, pixel_box, protect_edge_completion_context
    from watermark_studio.rectangle_phase import rectangle_phase_tight
    from watermark_studio.source_geometry import broad_masks
    from watermark_studio.source_panel_rims import effective_rim_scope
    from watermark_studio.tone_repair import repair_tone_steps

    helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    cases=helper["manifest_cases"](a.manifest);hashes=helper["file_hashes"](cases)
    ground_truth={c["file"].split("-",1)[0]:c["output"] for c in cases
                  if "exact_removal" in c["file"] and c.get("expected")=="passed"}
    a.target.mkdir(parents=True);(a.target/"changed").mkdir();rows=[];derived=[];started=time.monotonic()
    for index,case in enumerate(cases):
        before=load_photo(case["source"])[0];after=load_photo(case["output"])[0]
        regions=[Region.model_validate(r) for r in case["regions"]]
        rims=case.get("source_rim_protection",[]);cores=case.get("source_cores",[])
        audit={"applied":False,"reason":"source_scope_not_eligible"};fixed=after
        if case["scene"]=="outdoor" and not broad_masks((regions,)):
            scoped,core,invalid=effective_rim_scope(before,regions,cores,rims,header_core=True)
            if not invalid:
                old=rectangle_phase_tight(before,after,scoped,core,1024)
                candidate,audit=repair_tone_steps(before,after,scoped,core,resolution=a.resolution)
                if audit["applied"]:
                    candidate,_=protect_edge_completion_context(after,candidate,case.get("source_edge_completion",[]))
                    for rim in rims:
                        if rim.get("approved") is True:
                            native=tuple(rim["restore_pixels"]);candidate.paste(after.crop(native),native)
                    repeated=rectangle_phase_tight(before,candidate,scoped,core,a.resolution)
                    existing=rectangle_phase_tight(before,candidate,scoped,core,1024)
                    old_boxes={tuple(r["box"]) for r in old["findings"]}
                    if (repeated["count"]>=audit["initial_phase_count"] or
                            any(tuple(r["box"]) not in old_boxes for r in existing["findings"])):
                        audit.update(applied=False,reason="protected_or_new_original_guard_warning")
                    else:fixed=candidate
        changed=not np.array_equal(np.asarray(fixed),np.asarray(after))
        output=case["output"]
        metrics={}
        if changed:
            scope=Image.new("L",after.size);draw=ImageDraw.Draw(scope)
            for region in regions:
                l,t,r,b=pixel_box(region.box,after.size);draw.rectangle((l,t,r-1,b-1),fill=255)
            valid=np.asarray(scope)>0
            aa=np.asarray(after,dtype=np.int16);bb=np.asarray(fixed,dtype=np.int16)
            assert np.array_equal(aa[~valid],bb[~valid]) and np.abs(aa-bb).max()<=32
            path=a.target/"changed"/f"{index:04d}.png";fixed.save(path);output=str(path.resolve())
            parent=ground_truth.get(case["file"].split("-",1)[0])
            if parent:
                gt=np.asarray(load_photo(parent)[0],dtype=np.int16)
                metrics={"before_gt_mae":float(np.abs(aa-gt).mean()),
                         "after_gt_mae":float(np.abs(bb-gt).mean())}
        row={"file":case["file"],"changed":changed,"audit":audit,"ground_truth_metrics":metrics,
             "original_expected":case.get("expected"),"output":output}
        rows.append(row)
        # A repaired synthetic defect is a different image: retain its original
        # label separately and measure exact GT error; never count it as a waived pass.
        derived.append(case|{"output":output,"expected":None if changed else case.get("expected"),
            "original_expected":case.get("expected"),"tone_resolution_audit":audit})
        if (index+1)%16==0:print(json.dumps({"done":index+1,"changed":sum(r["changed"] for r in rows)}),flush=True)
    assert helper["file_hashes"](cases)==hashes
    report={"state":"completed_diagnostic","images":len(rows),"resolution":a.resolution,
            "changed":sum(r["changed"] for r in rows),"seconds":time.monotonic()-started,
            "clean_control_changes":[r["file"] for r in rows if r["changed"] and r["original_expected"]=="passed"],
            "changed_baseline_counts":dict(collections.Counter(c.get("baseline_status") for c,r in zip(cases,rows) if r["changed"])),
            "additional_generations":0,"source_bytes_unchanged":True,"production_adopted":False,
            "scope":"Pixel-repair diagnostic only; all actual outputs still require fresh unchanged strict QA"}
    (a.target/"results.json").write_text(json.dumps(rows,indent=2))
    (a.target/"report.json").write_text(json.dumps(report,indent=2))
    (a.target/"derived-manifest.json").write_text(json.dumps({"version":1,"cases":derived,
        "file_sha256":helper["file_hashes"](derived),"scope":report["scope"]},indent=2))
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
