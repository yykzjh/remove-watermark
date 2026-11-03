"""Run exactly one no-retry pass on a pre-frozen, SHA-bound thousand-image cohort."""
import argparse
import collections
import hashlib
import json
import time
from pathlib import Path

from PIL import Image

from watermark_studio.cli import request

p=argparse.ArgumentParser(__doc__)
p.add_argument("--frozen",type=Path,required=True)
p.add_argument("--work",type=Path,required=True)
p.add_argument("--baseline-job",required=True)
p.add_argument("--context",type=float,default=.45)
p.add_argument("--url",default="http://127.0.0.1:8765")
p.add_argument("--expected-uniform-steps",type=int,choices=[0,12,16,20,24,40],required=True)
p.add_argument("--expected-roi-steps",type=int,choices=[0,12,16,20,24,40],required=True)
p.add_argument("--expected-roi-min-area",type=float,default=.14)
a=p.parse_args()
assert .1<=a.context<=2
frozen=a.frozen.resolve(strict=True);selection=frozen/"source-selection.json"
raw=selection.read_bytes();manifest=json.loads(raw);hashes=manifest["source_sha256"]
assert len(hashes)==manifest["images"]==1000 and len(set(hashes.values()))==1000
inputs=frozen/"input"
actual={p.relative_to(inputs).as_posix() for p in inputs.rglob("*") if p.is_file()}
assert actual==set(hashes)
state=Path.home()/".local/state/watermark-studio"
base=json.loads((state/a.baseline_job/"summary.json").read_text())
opts=dict(base["options"],context=a.context,quality_auto_retry=False,quality_cross_review=False,
          quality_retry_localized=False,overwrite=False)
h=request(a.url,"/health")
assert h["stage"]=="ready" and h["backend"]=="qwen" and h["gpus"]==[0,1,2,3]
assert h["optimization"]["uniform_steps"]==a.expected_uniform_steps
assert h["optimization"]["roi_steps"]==a.expected_roi_steps
assert not a.expected_roi_steps or abs(h["optimization"]["roi_step_min_area"]-a.expected_roi_min_area)<1e-12
assert h["quality_weight_identity"]["relation"]=="different_weights" and h["quality_rectangle_guard"]
assert not h.get("job") or h["job"]["state"] in {"completed","failed","cancelled"}
assert not a.work.exists(),"Fresh outputs only; inspect existing request instead of resubmitting"
for name,expected in hashes.items():
    with (inputs/name).open("rb") as f:assert hashlib.file_digest(f,"sha256").hexdigest()==expected
a.work.mkdir(parents=True)
out=a.work/"output"
job=request(a.url,"/jobs",{"input_dir":str(inputs),"output_dir":str(out),"options":opts})["id"]
(a.work/"request.json").write_text(json.dumps({"job":job,"options":opts,"health":h,
    "selection_sha256":hashlib.sha256(raw).hexdigest()},ensure_ascii=False,indent=2))
print(json.dumps({"phase":"submitted","job":job,"total":1000,"context":a.context,"loaded_revision":h["code_revision"]}),flush=True)
start=time.monotonic();last=0
while True:
    current=request(a.url,"/health");j=current["job"];assert j["id"]==job
    if time.monotonic()-last>=30:
        print(json.dumps({"phase":"progress","done":j["completed"],"total":j["total"],
            "state":j["state"],"first_counts":j.get("first_pass_counts",{}),"errors":j.get("failed",0)}),flush=True)
        last=time.monotonic()
    if j["state"] in {"completed","failed","cancelled","interrupted"}:break
    if time.monotonic()-start>4*3600:raise TimeoutError(job)
    time.sleep(2)
s=json.loads((state/job/"summary.json").read_text())
rows=list(map(json.loads,(state/job/"results.jsonl").read_text().splitlines()))
assert len(rows)==1000 and {r["file"] for r in rows}==set(hashes)
assert all(r.get("quality_attempts",1)==1 for r in rows),"A retry cannot count as first pass"
bad_sizes=missing=0
for name,expected in hashes.items():
    with (inputs/name).open("rb") as f:assert hashlib.file_digest(f,"sha256").hexdigest()==expected
    if not (out/name).is_file():missing+=1;continue
    with Image.open(inputs/name) as before,Image.open(out/name) as after:
        bad_sizes+=before.size!=after.size
output_names={p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
input_dirs={p.relative_to(inputs).as_posix() for p in inputs.rglob("*") if p.is_dir()}
output_dirs={p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_dir()}
counts=dict(collections.Counter(r.get("quality",{}).get("status","error") for r in rows))
assert counts==s["first_pass_counts"]==s["quality_counts"]
seconds=s["finished_at"]-s["started_at"]
sampling=[log for row in rows for log in row.get("model_sampling",[])]
model_inputs=sum(r.get("model_inferences",0) for r in rows)
expected_steps={8,a.expected_roi_steps} if a.expected_roi_steps else {a.expected_uniform_steps or 8}
sampling_complete=(len(sampling)==model_inputs>0 and all(log.get("actual_steps") in expected_steps for log in sampling))
oom_retries=sum(r.get("oom_retries",0) for r in rows)
report={"job":job,"images":1000,"first_counts":counts,"final_counts":counts,
    "first_pass_rate":counts.get("passed",0)/1000,"seconds":seconds,"images_per_hour":3600000/seconds,
    "failed":s["failed"],"invalid":s["invalid"],"bad_sizes":bad_sizes,"missing_outputs":missing,
    "filename_mismatches":len(set(hashes)^output_names),"directory_mismatches":len(input_dirs^output_dirs),
    "directories":len(input_dirs),"source_bytes_unchanged":True,"no_generation_retries":True,
    "generation_region_inputs":model_inputs,"actual_denoising_steps":dict(collections.Counter(log.get("actual_steps") for log in sampling)),
    "sampling_audit_complete":sampling_complete,"generation_oom_retries":oom_retries,
    "paths":dict(collections.Counter(r.get("processing_path") for r in rows)),
    "selection_sha256":hashlib.sha256(raw).hexdigest(),"loaded_revision":current["code_revision"],
    "options":opts,"optimization":current["optimization"],"weight_identity":current["quality_weight_identity"],
    "qualified90":counts.get("passed",0)>=900 and sampling_complete and not oom_retries and not any([s["failed"],s["invalid"],bad_sizes,missing,
        len(set(hashes)^output_names),len(input_dirs^output_dirs)])}
(a.work/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in {"options","optimization","weight_identity"}}),flush=True)
