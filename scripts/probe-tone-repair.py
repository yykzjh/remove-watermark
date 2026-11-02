"""One deterministic color-step correction on immutable pairs; never performs diffusion."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from watermark_studio.config import Region
from watermark_studio.images import load_photo, protect_edge_completion_context
from watermark_studio.source_panel_rims import effective_rim_scope
from watermark_studio.tone_repair import repair_tone_steps


def main():
    p=argparse.ArgumentParser(__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--pair-manifest",type=Path)
    group.add_argument("--source-job")
    p.add_argument("--state-dir",type=Path,default=Path.home()/".local/state/watermark-studio")
    p.add_argument("--target",type=Path,required=True)
    a=p.parse_args()
    helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    if a.pair_manifest:
        cases=helper["manifest_cases"](a.pair_manifest)
    else:
        cases=helper["frozen_job_cases"](a.state_dir,a.source_job)
        rows={row["file"]:row for row in map(json.loads,
            (a.state_dir/a.source_job/"results.jsonl").read_text().splitlines())}
        for case in cases:
            case["source_edge_completion"]=rows[case["file"]].get("source_edge_completion",[])
    frozen_hashes=helper["file_hashes"](cases)
    truth={Path(c["source"]).name.split("-")[0]:c["output"] for c in cases
        if c.get("cohort")=="controls" and c.get("kind")=="exact_removal"}
    a.target.mkdir(exist_ok=False);work=a.target.resolve();(work/"output").mkdir()
    records=[];new_cases=[];totals=defaultdict(Counter);began=time.monotonic()
    for case in cases:
        source=load_photo(case["source"])[0];original=load_photo(case["output"])[0]
        regions=[Region.model_validate(r) for r in case["regions"]];cores=case.get("source_cores",[])
        rims=case.get("source_rim_protection",[])
        invalid=[]
        if rims and (case.get("cohort")=="controls" or "source_edge_completion" in case):
            regions,cores,invalid=effective_rim_scope(source,regions,cores,rims,header_core=True)
        if invalid:
            fixed=original.copy();audit={"applied":False,"reason":"source_rim_binding_not_valid","quality_permission":False}
        elif case.get("cohort")!="controls" and "source_edge_completion" not in case:
            fixed=original.copy();audit={"applied":False,"reason":"missing_protection_snapshot","quality_permission":False}
        elif case.get("scene")=="document":
            fixed=original.copy();audit={"applied":False,"reason":"preserve_document_appearance","quality_permission":False}
        elif "source_physical_ambiguity" in case.get("frozen_vetoes",[]):
            fixed=original.copy();audit={"applied":False,"reason":"immutable_source_ambiguity","quality_permission":False}
        else:
            fixed,audit=repair_tone_steps(source,original,regions,cores)
            if audit.get("applied"):
                fixed,_=protect_edge_completion_context(original,fixed,case.get("source_edge_completion",[]))
                for rim in rims:
                    if rim.get("approved"):
                        native=tuple(rim["restore_pixels"])
                        fixed.paste(original.crop(native),native)
        old=np.asarray(original.convert("RGB"),dtype=np.float32)
        new=np.asarray(fixed.convert("RGB"),dtype=np.float32)
        changed=bool(np.any(old!=new));new_case=dict(case)
        if changed:
            destination=work/"output"/(hashlib.sha256(case["file"].encode()).hexdigest()[:24]+".png")
            fixed.save(destination)
            new_case["output"]=str(destination)
        new_case["file"]="tone_once/"+case["file"]
        # Every source and inherited veto remains; the unchanged QA recomputes
        # all numeric, residual, naturalness and content checks on the new pixels.
        new_cases.append(new_case)
        record={"file":case["file"],"cohort":case.get("cohort","unknown"),"expected":case.get("expected"),
            "changed":changed,"audit":audit,"quality_permission":False,"extra_generations":0}
        cohort=record["cohort"];totals[cohort]["cases"]+=1;totals[cohort]["changed"]+=int(changed)
        if record["expected"]=="passed":totals[cohort]["known_clean_changed"]+=int(changed)
        target=truth.get(Path(case["source"]).name.split("-")[0]) if cohort=="controls" else None
        if target:
            clean=np.asarray(load_photo(target)[0].convert("RGB"),dtype=np.float32)
            record["known_clean_mae_before"]=float(np.abs(old-clean).mean())
            record["known_clean_mae_after"]=float(np.abs(new-clean).mean())
            record["known_gt_used_for_processing"]=False
            totals[cohort]["gt_improved"]+=int(record["known_clean_mae_after"]<record["known_clean_mae_before"]-1e-8)
            totals[cohort]["gt_worsened"]+=int(record["known_clean_mae_after"]>record["known_clean_mae_before"]+1e-8)
        records.append(record)
        print(json.dumps({"completed":len(records),"total":len(cases),"changed":changed}),flush=True)
    if helper["file_hashes"](cases)!=frozen_hashes:raise ValueError("Frozen source/candidate bytes changed")
    if a.pair_manifest:helper["manifest_cases"](a.pair_manifest)
    paired={"version":1,"cases":new_cases,"file_sha256":helper["file_hashes"](new_cases)}
    (work/"pair-manifest.json").write_text(json.dumps(paired,ensure_ascii=False))
    helper["manifest_cases"](work/"pair-manifest.json")
    report={"version":1,"state":"completed","cases":len(records),"cohorts":dict(totals),
        "seconds":round(time.monotonic()-began,3),"extra_generations":0,"quality_permission":False,
        "scope":"Offline single bounded photometric repair; no QA threshold changes, no native acceptance claim"}
    (work/"results.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    (work/"report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=="__main__":main()
