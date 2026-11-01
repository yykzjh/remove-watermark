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
    rectangle_phase_tight,
)

p=argparse.ArgumentParser()
p.add_argument("--controls",type=Path,required=True)
p.add_argument("--baseline",type=Path,required=True)
p.add_argument("--core-boundaries",action="store_true")
p.add_argument("--tight-core",action="store_true",help="Offline only: fixed two-pixel source-core search; original thresholds and all side requirements")
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
if a.core_boundaries and a.tight_core:
    p.error("Choose one diagnostic search")
if a.output.exists():
    p.error("Fresh output required")
helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
cases=helper["known_control_cases"](a.controls)
baseline={r["file"]:r["quality"]["status"] for r in map(json.loads,a.baseline.read_text().splitlines()) if r["cohort"]=="controls"}
hashes=helper["file_hashes"](cases)
started=time.monotonic()
rows=[]
for c in cases:
    arguments=(load_photo(c["source"])[0],load_photo(c["output"])[0],
               [Region.model_validate(r) for r in c["regions"]])
    if a.tight_core:
        result=rectangle_phase_tight(*arguments,c["source_cores"])
    else:
        result=(rectangle_phase_core(*arguments,c["source_cores"]) if a.core_boundaries
                else rectangle_phase(*arguments))
    rows.append({"file":c["file"],"expected":c["expected"],"baseline":baseline[c["file"]],"phase":result})
assert hashes==helper["file_hashes"](cases)
report={"code_revision":subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip(),"images":len(cases),"sources_and_outputs_unchanged":True,"seconds":round(time.monotonic()-started,3),
        "clean_warned":[r["file"] for r in rows if r["expected"]=="passed" and r["phase"]["count"]],
        "newly_caught_defects":[r["file"] for r in rows if r["expected"]=="review" and r["baseline"]=="passed" and r["phase"]["count"]],
        "all_defects_warned":sum(r["expected"]=="review" and bool(r["phase"]["count"]) for r in rows),
        "core_boundaries":a.core_boundaries,"tight_core":a.tight_core,"scope":"offline cue, not full QA acceptance","rows":rows}
a.output.write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!="rows"}),flush=True)
