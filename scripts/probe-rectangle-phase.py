"""CPU-only phase cue on frozen labeled controls; does not alter production checks."""
import argparse
import json
import runpy
import subprocess
import time
from pathlib import Path

from watermark_studio.config import Region
from watermark_studio.images import load_photo
from watermark_studio.rectangle_phase import (
    rectangle_phase,
    rectangle_phase_core,
    rectangle_phase_inset,
    rectangle_phase_sharp,
    rectangle_phase_tight,
)

p=argparse.ArgumentParser()
source=p.add_mutually_exclusive_group(required=True)
source.add_argument("--controls",type=Path)
source.add_argument("--source-job",help="Read-only real development outputs from a completed server job")
p.add_argument("--state-dir",type=Path,default=Path.home()/".local/state/watermark-studio")
p.add_argument("--baseline",type=Path)
p.add_argument("--core-boundaries",action="store_true")
p.add_argument("--sharp-core",action="store_true",help="Offline original smoothing with fixed native-source hypotheses; no approval")
p.add_argument("--inset-core",action="store_true",help="Offline fixed safety-padding offset; preserves every v5 veto")
p.add_argument("--tight-core",action="store_true",help="Offline only: fixed two-pixel source-core search; original thresholds and all side requirements")
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
if sum((a.core_boundaries,a.tight_core,a.inset_core,a.sharp_core))>1:
    p.error("Choose one diagnostic search")
if a.output.exists():
    p.error("Fresh output required")
helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
if a.controls:
    if a.baseline is None:
        p.error("Labeled controls need a baseline")
    cases=helper["known_control_cases"](a.controls)
    baseline={r["file"]:r["quality"]["status"] for r in map(json.loads,a.baseline.read_text().splitlines()) if r["cohort"]=="controls"}
else:
    cases=helper["frozen_job_cases"](a.state_dir,a.source_job)
    baseline={r["file"]:r["baseline_status"] for r in cases}
hashes=helper["file_hashes"](cases)
started=time.monotonic()
rows=[]
for c in cases:
    arguments=(load_photo(c["source"])[0],load_photo(c["output"])[0],
               [Region.model_validate(r) for r in c["regions"]])
    if a.sharp_core:
        result=rectangle_phase_sharp(*arguments,c["source_cores"])
    elif a.inset_core:
        result=rectangle_phase_inset(*arguments,c["source_cores"])
    elif a.tight_core:
        result=rectangle_phase_tight(*arguments,c["source_cores"])
    else:
        result=(rectangle_phase_core(*arguments,c["source_cores"]) if a.core_boundaries
                else rectangle_phase(*arguments))
    rows.append({"file":c["file"],"expected":c.get("expected"),"baseline":baseline[c["file"]],
                 "old_phase":rectangle_phase(*arguments),"phase":result})
assert hashes==helper["file_hashes"](cases)
report={"code_revision":subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip(),"images":len(cases),"sources_and_outputs_unchanged":True,"seconds":round(time.monotonic()-started,3),
        "clean_warned":[r["file"] for r in rows if r["expected"]=="passed" and r["phase"]["count"]],
        "newly_caught_defects":[r["file"] for r in rows if r["expected"]=="review" and r["baseline"]=="passed" and r["phase"]["count"]],
        "all_defects_warned":sum(r["expected"]=="review" and bool(r["phase"]["count"]) for r in rows),
        "new_warnings":[r["file"] for r in rows if r["baseline"]=="passed" and not r["old_phase"]["count"] and r["phase"]["count"]],
        "core_boundaries":a.core_boundaries,"tight_core":a.tight_core,"inset_core":a.inset_core,"sharp_core":a.sharp_core,"scope":"offline cue, not full QA acceptance","rows":rows}
a.output.write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!="rows"}),flush=True)
