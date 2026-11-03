"""Run exactly one no-retry pass on a pre-frozen, SHA-bound thousand-image cohort."""
import argparse
import collections
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageOps

from watermark_studio.cli import request

p=argparse.ArgumentParser(__doc__)
p.add_argument("--frozen",type=Path,required=True)
p.add_argument("--work",type=Path,required=True)
p.add_argument("--baseline-job",required=True)
p.add_argument("--attach-job",default="",help="Read-only continuation of the same recorded benchmark; never submits another job")
p.add_argument("--context",type=float,default=.45)
p.add_argument("--model-resolution",type=int,choices=[512,768,1024,1536],default=1024)
p.add_argument("--url",default="http://127.0.0.1:8765")
p.add_argument("--expected-uniform-steps",type=int,choices=[0,12,16,20,24,40],required=True)
p.add_argument("--expected-roi-steps",type=int,choices=[0,12,16,20,24,40],required=True)
p.add_argument("--expected-roi-min-area",type=float,default=.14)
p.add_argument("--futility-stop",action="store_true",help="Cancel when even every remaining pass cannot reach 900/1000; never qualify a partial run")
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
opts=dict(base["options"],context=a.context,resolution=a.model_resolution,quality_auto_retry=False,quality_cross_review=False,
          quality_retry_localized=False,overwrite=False)
h=request(a.url,"/health")
attached=None
if a.attach_job:
    from benchmark_monitor import attachment_health
    attached=json.loads((a.work/"request.json").read_text())
    h=attachment_health(attached,h,a.attach_job,opts,hashlib.sha256(raw).hexdigest())
assert h["stage"]=="ready" and h["backend"]=="qwen" and h["gpus"]==[0,1,2,3]
assert h["optimization"]["uniform_steps"]==a.expected_uniform_steps
assert h["optimization"]["roi_steps"]==a.expected_roi_steps
assert not a.expected_roi_steps or abs(h["optimization"]["roi_step_min_area"]-a.expected_roi_min_area)<1e-12
assert h["quality_weight_identity"]["relation"]=="different_weights" and h["quality_rectangle_guard"]
assert not h.get("job") or h["job"]["state"] in {"completed","failed","cancelled"}
if not a.attach_job:
    assert not a.work.exists(),"Fresh outputs only; use --attach-job to monitor the recorded request"
source_tree=subprocess.check_output(["git","rev-parse","HEAD:src"],text=True).strip()
loaded_tree=subprocess.check_output(["git","rev-parse",h["code_revision"]+":src"],text=True).strip()
assert source_tree==loaded_tree,"Restart before measuring changed inference sources"
assert subprocess.run(["git","diff","--quiet","HEAD","--","src","uv.lock"],check=False).returncode==0
lock_sha256=hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest()
for name,expected in hashes.items():
    with (inputs/name).open("rb") as f:assert hashlib.file_digest(f,"sha256").hexdigest()==expected
out=a.work/"output"
if a.attach_job:
    assert a.work.is_dir() and out.is_dir()
    job=a.attach_job
    with (a.work/"monitor-attachments.jsonl").open("a") as log:
        log.write(json.dumps({"job":job,"attached_at":time.time(),
            "futility_stop":a.futility_stop,"new_job_submitted":False})+chr(10))
    print(json.dumps({"phase":"attached","job":job,"new_job_submitted":False}),flush=True)
else:
    a.work.mkdir(parents=True)
    out=a.work/"output"
    job=request(a.url,"/jobs",{"input_dir":str(inputs),"output_dir":str(out),"options":opts})["id"]
    (a.work/"request.json").write_text(json.dumps({"job":job,"options":opts,"health":h,
        "selection_sha256":hashlib.sha256(raw).hexdigest()},ensure_ascii=False,indent=2))
    print(json.dumps({"phase":"submitted","job":job,"total":1000,"context":a.context,"loaded_revision":h["code_revision"]}),flush=True)
start=time.monotonic();last=0;futility_requested=False
while True:
    current=request(a.url,"/health");j=current["job"];assert j["id"]==job
    if time.monotonic()-last>=30:
        print(json.dumps({"phase":"progress","done":j["completed"],"total":j["total"],
            "state":j["state"],"first_counts":j.get("first_pass_counts",{}),"errors":j.get("failed",0)}),flush=True)
        last=time.monotonic()
    first=j.get("first_pass_counts",{})
    counted=sum(first.values())
    ceiling=first.get("passed",0)+1000-counted
    if a.futility_stop and not futility_requested and ceiling<900 and j["state"] not in {"completed","failed","cancelled","interrupted"}:
        request(a.url,"/jobs/cancel",{})
        futility_requested=True
        (a.work/"futility-stop.json").write_text(json.dumps({"job":job,"counted":counted,
            "first_counts":first,"maximum_possible_passed":ceiling,"target":900},indent=2))
        print(json.dumps({"phase":"futility_cancel_requested","maximum_possible_passed":ceiling,"target":900}),flush=True)
    if j["state"] in {"completed","failed","cancelled","interrupted"}:break
    if time.monotonic()-start>4*3600:raise TimeoutError(job)
    time.sleep(2)
s=json.loads((state/job/"summary.json").read_text())
rows=list(map(json.loads,(state/job/"results.jsonl").read_text().splitlines()))
if s["state"]!="completed" or len(rows)!=1000:
    assert len({r["file"] for r in rows})==len(rows) and {r["file"] for r in rows}<=set(hashes)
    for name,expected in hashes.items():
        with (inputs/name).open("rb") as f:assert hashlib.file_digest(f,"sha256").hexdigest()==expected
    partial_counts=dict(collections.Counter(r.get("quality",{}).get("status","error") for r in rows))
    stopped={"job":job,"state":s["state"],"images":1000,"evaluated":len(rows),
        "unprocessed":1000-len(rows),"first_counts":partial_counts,"qualified90":False,
        "maximum_possible_passed":partial_counts.get("passed",0)+1000-len(rows),
        "futility_requested":futility_requested,"source_bytes_unchanged":True,
        "scope":"Incomplete evaluation; no thousand-image pass rate or qualified performance claim"}
    (a.work/"stopped-report.json").write_text(json.dumps(stopped,ensure_ascii=False,indent=2))
    print(json.dumps(stopped,ensure_ascii=False),flush=True)
    raise SystemExit(2)
assert len(rows)==1000 and {r["file"] for r in rows}==set(hashes)
assert all(r.get("quality_attempts",1)==1 for r in rows),"A retry cannot count as first pass"
bad_sizes=missing=0
for name,expected in hashes.items():
    with (inputs/name).open("rb") as f:assert hashlib.file_digest(f,"sha256").hexdigest()==expected
    if not (out/name).is_file():missing+=1;continue
    with Image.open(inputs/name) as before,Image.open(out/name) as after:
        bad_sizes+=ImageOps.exif_transpose(before).size!=ImageOps.exif_transpose(after).size
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
routes_match=True
if a.expected_roi_steps:
    for log in sampling:
        route=log.get("roi_route",{})
        fraction=route.get("source_area_fraction")
        if route.get("whole_generation_ignored") is True:
            selected=0
        elif type(fraction) not in (int,float) or not math.isfinite(fraction):
            routes_match=False;continue
        else:
            selected=a.expected_roi_steps if fraction>=a.expected_roi_min_area else 0
        routes_match=routes_match and route.get("policy")=="source_area_v1" and (
            route.get("selected_uniform_steps")==selected and log.get("actual_steps")==(selected or 8))
assert source_tree==subprocess.check_output(["git","rev-parse","HEAD:src"],text=True).strip()
assert subprocess.run(["git","diff","--quiet","HEAD","--","src","uv.lock"],check=False).returncode==0
assert current["optimization"]==h["optimization"] and current["quality_weight_identity"]==h["quality_weight_identity"]
assert lock_sha256==hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest()
existing=s.get("first_pass_existing",0)+s.get("existing_outputs",0)
report={"job":job,"images":1000,"first_counts":counts,"final_counts":counts,
    "first_pass_rate":counts.get("passed",0)/1000,"seconds":seconds,"images_per_hour":3600000/seconds,
    "failed":s["failed"],"invalid":s["invalid"],"bad_sizes":bad_sizes,"missing_outputs":missing,
    "filename_mismatches":len(set(hashes)^output_names),"directory_mismatches":len(input_dirs^output_dirs),
    "directories":len(input_dirs),"source_bytes_unchanged":True,"no_generation_retries":True,
    "physical_text_inquiries":sum(r.get("quality",{}).get("physical_text_review",{}).get("attempted",False) for r in rows),
    "physical_text_corrections":sum(r.get("quality",{}).get("physical_text_review",{}).get("status")=="corrected" for r in rows),
    "physical_text_inquiry_seconds_sum":sum(r.get("quality",{}).get("physical_text_review",{}).get("seconds",0) for r in rows),
    "generation_region_inputs":model_inputs,"actual_denoising_steps":dict(collections.Counter(log.get("actual_steps") for log in sampling)),
    "sampling_audit_complete":sampling_complete,"sampling_routes_match":routes_match,
    "source_tree":source_tree,"uv_lock_sha256":lock_sha256,
    "generation_oom_retries":oom_retries,"existing_outputs":existing,
    "paths":dict(collections.Counter(r.get("processing_path") for r in rows)),
    "selection_sha256":hashlib.sha256(raw).hexdigest(),"loaded_revision":current["code_revision"],
    "options":opts,"optimization":current["optimization"],"weight_identity":current["quality_weight_identity"],
    "qualified90":counts.get("passed",0)>=900 and sampling_complete and routes_match and not oom_retries and not existing and not any([s["failed"],s["invalid"],bad_sizes,missing,
        len(set(hashes)^output_names),len(input_dirs^output_dirs)])}
(a.work/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in {"options","optimization","weight_identity"}}),flush=True)
