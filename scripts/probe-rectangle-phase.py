"""CPU-only phase cue on frozen labeled controls; does not alter production checks."""
import argparse
import json
import runpy
import time
from pathlib import Path

from watermark_studio.config import Region
from watermark_studio.images import load_photo
from watermark_studio.rectangle_phase import rectangle_phase

p=argparse.ArgumentParser()
p.add_argument("--controls",type=Path,required=True)
p.add_argument("--baseline",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
if a.output.exists():
    p.error("Fresh output required")
helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
cases=helper["known_control_cases"](a.controls)
baseline={r["file"]:r["quality"]["status"] for r in map(json.loads,a.baseline.read_text().splitlines()) if r["cohort"]=="controls"}
hashes=helper["file_hashes"](cases)
started=time.monotonic()
rows=[]
for c in cases:
    result=rectangle_phase(load_photo(c["source"])[0],load_photo(c["output"])[0],
                           [Region.model_validate(r) for r in c["regions"]])
    rows.append({"file":c["file"],"expected":c["expected"],"baseline":baseline[c["file"]],"phase":result})
assert hashes==helper["file_hashes"](cases)
report={"images":len(cases),"sources_and_outputs_unchanged":True,"seconds":round(time.monotonic()-started,3),
        "clean_warned":[r["file"] for r in rows if r["expected"]=="passed" and r["phase"]["count"]],
        "newly_caught_defects":[r["file"] for r in rows if r["expected"]=="review" and r["baseline"]=="passed" and r["phase"]["count"]],
        "all_defects_warned":sum(r["expected"]=="review" and bool(r["phase"]["count"]) for r in rows),
        "scope":"offline cue, not full QA acceptance","rows":rows}
a.output.write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!="rows"}),flush=True)
