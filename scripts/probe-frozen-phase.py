"""Read-only prediction of added review cues on frozen accepted candidates."""
import argparse
import json
import runpy
import time
from pathlib import Path

from watermark_studio.config import Region
from watermark_studio.images import load_photo
from watermark_studio.rectangle_phase import rectangle_phase

p=argparse.ArgumentParser()
p.add_argument("--job",required=True)
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
if a.output.exists():
    p.error("Fresh report required")
state=Path.home()/".local/state/watermark-studio"
helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
cases=helper["frozen_job_cases"](state,a.job)
before=helper["file_hashes"](cases)
baseline={r["file"]:r["quality"] for r in map(json.loads,(state/a.job/"results.jsonl").read_text().splitlines())}
started=time.monotonic();rows=[]
for case in cases:
    if baseline[case["file"]]["status"]!="passed":
        continue
    result=rectangle_phase(load_photo(case["source"])[0],load_photo(case["output"])[0],
                           [Region.model_validate(r) for r in case["regions"]])
    rows.append({"file":case["file"],"cue":result})
assert before==helper["file_hashes"](cases)
summary={"job":a.job,"all_candidates":len(cases),"previously_passed":len(rows),
         "added_review_cues":[r["file"] for r in rows if r["cue"]["count"]],
         "seconds":round(time.monotonic()-started,3),
         "sources_and_outputs_unchanged":True,
         "scope":"new diagnostic cues only, no correctness labels and no regenerated candidates"}
a.output.write_text(json.dumps(summary|{"rows":rows},indent=2))
print(json.dumps(summary),flush=True)
