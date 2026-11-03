"""Paired route and bounded-retry ablations on a fixed reused-source cohort."""
import argparse
import collections
import hashlib
import json
import os
import random
import time
from pathlib import Path

from watermark_studio.cli import request
from watermark_studio.images import load_photo

p=argparse.ArgumentParser()
p.add_argument("--work",type=Path,required=True)
p.add_argument("--baseline-job",required=True)
p.add_argument("--images",type=int,default=64)
p.add_argument("--selection-start",type=int,default=0)
p.add_argument("--balanced-grouping",choices=["separate","adaptive"],default="adaptive")
p.add_argument("--balanced-context",type=float,default=.85)
p.add_argument("--url",default="http://127.0.0.1:8765")
p.add_argument("--profiles",nargs="+",default=["roi","roi-context","whole-protected"])
p.add_argument("--roi-context",type=float,default=.45)
a=p.parse_args()
assert .1<=a.roi_context<=2
work=a.work.resolve();work.mkdir(parents=True,exist_ok=True)
state=Path.home()/".local/state/watermark-studio"
baseline=json.loads((state/a.baseline_job/"summary.json").read_text())
rows=[json.loads(s) for s in (state/a.baseline_job/"results.jsonl").read_text().splitlines()]
selection=work/"selection.json"
if selection.exists():
    chosen=json.loads(selection.read_text())
else:
    rng=random.Random(20261103)
    names=sorted(r["file"] for r in rows);rng.shuffle(names)
    chosen=names[a.selection_start:a.selection_start+a.images]
    selection.write_text(json.dumps(chosen,ensure_ascii=False,indent=2))
assert len(chosen)==a.images and len(set(chosen))==a.images
inputs=work/"input";inputs.mkdir(exist_ok=True)
hashes={}
for name in chosen:
    src=Path(baseline["input_dir"])/name;dst=inputs/name
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():os.link(src,dst)
    with dst.open("rb") as f:hashes[name]=hashlib.file_digest(f,"sha256").hexdigest()
options=baseline["options"]
original={r["file"]:r for r in rows}
print(json.dumps({"phase":"prepared","images":len(chosen),"development_reuse":True,
 "baseline_subset":dict(collections.Counter(original[n]["quality"]["status"] for n in chosen))}),flush=True)
for profile in a.profiles:
    out=work/profile;report=work/(profile+".json")
    if report.exists():
        print(json.dumps({"phase":"existing_report","profile":profile}),flush=True);continue
    if out.exists():raise ValueError("Output already exists without final report; inspect existing job")
    h=request(a.url,"/health")
    assert h["stage"]=="ready" and (not h.get("job") or h["job"]["state"] in {"completed","failed","cancelled"})
    opts=dict(options,quality_auto_retry=False,quality_cross_review=False,overwrite=False)
    if profile=="roi":opts["context"]=a.roi_context
    elif profile=="roi-context":opts["context"]=.85
    elif profile=="roi-grouped":opts["roi_grouping"]="adaptive"
    elif profile=="roi-evidence":opts.update(roi_grouping="separate",context=.45,quality_auto_retry=True,quality_retry_localized=True,quality_max_attempts=2,quality_retry_resolution=1024)
    elif profile=="roi-balanced":opts.update(roi_grouping=a.balanced_grouping,context=a.balanced_context,quality_auto_retry=True,quality_max_attempts=2,quality_retry_resolution=1024)
    elif profile=="roi-grouped-retry":opts.update(roi_grouping="adaptive",quality_auto_retry=True,quality_max_attempts=2,quality_retry_resolution=1024)
    elif profile=="whole-protected":opts.update(first_pass_mode="whole",whole_prompt_profile="neutral")
    elif profile!="roi":raise ValueError(profile)
    started=time.monotonic()
    job=request(a.url,"/jobs",{"input_dir":str(inputs),"output_dir":str(out),"options":opts})["id"]
    (work/(profile+"-request.json")).write_text(json.dumps({"job":job,"options":opts,
        "loaded_revision":h["code_revision"],"selection_sha256":hashlib.sha256(selection.read_bytes()).hexdigest()},indent=2))
    print(json.dumps({"phase":"submitted","profile":profile,"job":job}),flush=True)
    last=0
    while True:
        current=request(a.url,"/health");j=current["job"];assert j["id"]==job
        if time.monotonic()-last>30:
            print(json.dumps({"profile":profile,"completed":j["completed"],"total":j["total"],
                "counts":j.get("quality_counts",{})}),flush=True);last=time.monotonic()
        if j["state"] in {"completed","failed","cancelled","interrupted"}:break
        if time.monotonic()-started>7200:raise TimeoutError(job)
        time.sleep(2)
    s=json.loads((state/job/"summary.json").read_text())
    rs=[json.loads(x) for x in (state/job/"results.jsonl").read_text().splitlines()]
    assert len(rs)==len(chosen)==len({r["file"] for r in rs})
    assert {r["file"] for r in rs}==set(chosen) and not s["failed"] and not s["invalid"]
    if not opts["quality_auto_retry"]:assert all(r.get("quality_attempts",1)==1 for r in rs)
    dims=0
    for name in chosen:
        with (inputs/name).open("rb") as f:assert hashlib.file_digest(f,"sha256").hexdigest()==hashes[name]
        dims+=load_photo(str(inputs/name))[0].size!=load_photo(str(out/name))[0].size
    assert dims==0
    seconds=s["finished_at"]-s["started_at"]
    result={"profile":profile,"job":job,"images":len(chosen),"development_reuse":True,
        "seconds":seconds,"images_per_hour":len(chosen)*3600/seconds,
        "first_counts":s["first_pass_counts"],"final_counts":s["quality_counts"],
        "bad_sizes":dims,"failed":s["failed"],"options":opts,"loaded_revision":current["code_revision"],
        "paths":dict(collections.Counter(r.get("processing_path") for r in rs)),
        "model_inputs":sum(sum(t.get("model_inferences",0) for t in r.get("attempt_history",[r])) for r in rs),
        "attempt_counts":dict(collections.Counter(r.get("quality_attempts",1) for r in rs)),
        "baseline_pass_to_review":sum(original[r["file"]]["quality"]["status"]=="passed" and r["quality"]["status"]!="passed" for r in rs),
        "baseline_review_to_pass":sum(original[r["file"]]["quality"]["status"]!="passed" and r["quality"]["status"]=="passed" for r in rs)}
    report.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!="options"}),flush=True)
