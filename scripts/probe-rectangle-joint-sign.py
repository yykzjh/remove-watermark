"""Offline joint-sign rectangle search; no production checks or image files mutate."""
from __future__ import annotations

import argparse
import json
import runpy
import time
from pathlib import Path

from watermark_studio import rectangle_phase as phase_module
from watermark_studio.config import Region
from watermark_studio.images import load_photo

p=argparse.ArgumentParser()
p.add_argument("--controls",type=Path,required=True)
p.add_argument("--baseline",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
if a.output.exists():
    p.error("Fresh report required")
helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
cases=helper["known_control_cases"](a.controls)
hashes=helper["file_hashes"](cases)
baseline={r["file"]:r["quality"]["status"] for r in map(json.loads,a.baseline.read_text().splitlines())
          if r["cohort"]=="controls" and r.get("context_variant","local")=="local"}
variant=phase_module._single_phase
rows=[];started=time.monotonic()
for case in cases:
    before,after=load_photo(case["source"])[0],load_photo(case["output"])[0]
    regions=[Region.model_validate(r) for r in case["regions"]]
    original=variant(before,after,regions)
    alternate=[variant(before,after,regions,forced_sign=sign) for sign in (-1,1)]
    rows.append({"file":case["file"],"expected":case["expected"],"kind":case["kind"],
                 "baseline":baseline[case["file"]],"original_count":original["count"],
                 "joint_count":sum(r["count"] for r in alternate)})
assert hashes==helper["file_hashes"](cases)
report={"scope":"CPU diagnostic only; no production threshold changes and no acceptance waivers",
        "images":len(rows),"seconds":round(time.monotonic()-started,3),
        "source_and_outputs_unchanged":True,
        "new_clean_warnings":[r["file"] for r in rows if r["expected"]=="passed" and
                              r["joint_count"] and not r["original_count"]],
        "lost_defect_warnings":[r["file"] for r in rows if r["expected"]=="review" and
                                r["original_count"] and not r["joint_count"]],
        "newly_caught_baseline_misses":[r["file"] for r in rows if r["expected"]=="review" and
                                       r["baseline"]=="passed" and r["joint_count"]],
        "all_defects_warned":sum(r["expected"]=="review" and bool(r["joint_count"]) for r in rows),
        "rows":rows}
a.output.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k!="rows"}),flush=True)
