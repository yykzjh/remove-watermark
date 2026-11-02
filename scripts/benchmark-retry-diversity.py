"""Bounded rework-only development screen: all localized failures, fixed paths and seed schedule."""
import argparse
import collections
import hashlib
import json
import os
import time
from pathlib import Path

from watermark_studio.cli import request

p=argparse.ArgumentParser()
p.add_argument("--source-job",required=True)
p.add_argument("--work",type=Path,required=True)
p.add_argument("--url",default="http://127.0.0.1:8765")
a=p.parse_args()
state=Path.home()/".local/state/watermark-studio"
s=json.loads((state/a.source_job/"summary.json").read_text())
rs=[json.loads(x) for x in (state/a.source_job/"results.jsonl").read_text().splitlines()]
selected=[r for r in rs if r.get("quality_attempts",1)>1 and r["quality"]["status"]!="passed"]
assert selected and len(selected)<=16
a.work.mkdir(parents=True,exist_ok=False);inputs=a.work/"input";inputs.mkdir()
hashes={}
for r in selected:
    src=Path(s["input_dir"])/r["file"];dst=inputs/r["file"];dst.parent.mkdir(parents=True,exist_ok=True)
    os.link(src,dst);hashes[r["file"]]=hashlib.sha256(src.read_bytes()).hexdigest()
(a.work/"selection.json").write_text(json.dumps(hashes,ensure_ascii=False,indent=2))
for label,attempts,resolution in [("one-retry-1024",2,1024),("two-retries-1024",3,1024),("two-retries-1536",3,1536)]:
    h=request(a.url,"/health");assert h["stage"]=="ready" and (not h.get("job") or h["job"]["state"]=="completed")
    options=dict(s["options"],quality_max_attempts=attempts,quality_retry_resolution=resolution,overwrite=False)
    job=request(a.url,"/jobs",{"input_dir":str(inputs),"output_dir":str(a.work/label),"options":options})["id"]
    print(json.dumps({"profile":label,"job":job,"images":len(selected)}),flush=True)
    began=time.monotonic()
    while True:
        h=request(a.url,"/health");j=h["job"];assert j["id"]==job
        if j["state"] in {"completed","failed","cancelled","interrupted"}:break
        if time.monotonic()-began>1200:raise TimeoutError(job)
        time.sleep(2)
    summary=json.loads((state/job/"summary.json").read_text())
    results=[json.loads(x) for x in (state/job/"results.jsonl").read_text().splitlines()]
    assert len(results)==len(selected)==len({r["file"] for r in results})
    histories=[t for r in results for t in r.get("attempt_history",[r])]
    record={"profile":label,"job":job,"scope":"reused localized failures; not full-batch acceptance",
        "images":len(results),"first_counts":summary["first_pass_counts"],"final_counts":summary["quality_counts"],
        "seconds":summary["finished_at"]-summary["started_at"],"attempt_counts":dict(collections.Counter(r.get("quality_attempts",1) for r in results)),
        "generation_inputs":sum(t.get("model_inferences",0) for t in histories),
        "attempt_errors":sum(not t.get("ok") for t in histories),"oom":sum(t.get("oom_retries",0) for t in histories),
        "rejected_recovered":sum((r.get("attempt_history") or [r])[0].get("quality",{}).get("status")!="passed" and r["quality"]["status"]=="passed" for r in results),
        "seeds":[[t.get("model_seeds",[]) for t in r.get("attempt_history",[r])] for r in results],
        "feedback_observations":sum(bool(f.get("observation")) for r in results for t in r.get("attempt_history",[]) for f in t.get("repair_feedback_used",[])),
        "loaded_revision":h["code_revision"]}
    assert all(hashlib.sha256((inputs/name).read_bytes()).hexdigest()==sha for name,sha in hashes.items())
    (a.work/(label+".json")).write_text(json.dumps(record,indent=2))
    print(json.dumps(record),flush=True)
