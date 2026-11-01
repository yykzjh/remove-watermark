"""Frozen protected whole-image first/final acceptance and total throughput; no uploads."""
import argparse
import collections
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

from PIL import Image

from watermark_studio.cli import request

benchmark_revision=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
p=argparse.ArgumentParser()
p.add_argument("--work",type=Path,required=True)
p.add_argument("--url",default="http://127.0.0.1:8765")
p.add_argument("--concurrency",type=int,default=20)
p.add_argument("--input-resolution",type=int,choices=[512,768,1024,1280,1536,2048],default=1024)
p.add_argument("--extra-prompt",default="",help="Explicit paired prompt suffix; recorded with frozen processing options")
p.add_argument("--whole-prompt-profile",choices=["baseline","fidelity","scene","located","concise","neutral","clean-photo","clean-photo-detailed"],default="fidelity")
p.add_argument("--run-tag",default="protected-e2e",help="Fresh output name inside the immutable selection; lets paired runs retain identical paths/seeds")
p.add_argument("--development",action="store_true",help="Explicit reused-source development cohort; never qualifies as an unseen holdout")
p.add_argument("--auto-retry",action="store_true")
p.add_argument("--cross-review",action="store_true")
p.add_argument("--attempts",type=int,default=3)
p.add_argument("--retry-resolution",type=int,choices=[1024,1280,1536,2048],default=1024)
a=p.parse_args()
if not re.fullmatch(r"[a-zA-Z0-9_-]+",a.run_tag):
    p.error("Simple run tag required")
work=a.work.resolve();inputs=work/"input";target=work/a.run_tag
output=work/("e2e-summary.json" if a.run_tag=="protected-e2e" else a.run_tag+"-summary.json")
if target.exists() or output.exists() or not (work/"source-selection.json").is_file():
    p.error("Prepared immutable selection and fresh output required")
selection=json.loads((work/"source-selection.json").read_text())
hashes=selection["source_sha256"]
if selection["unique_sha256"]!=len(hashes):
    p.error("Nonunique test inputs")
if selection["prior_sha256_overlap"] and not (a.development and selection.get("development_reuse")):
    p.error("Overlapping inputs require an explicitly marked development cohort")
if selection.get("development_reuse") and not a.development:
    p.error("Reused development sources cannot be relabelled as a fresh holdout")
def audit_sources():
    return all(hashlib.sha256((inputs/name).read_bytes()).hexdigest()==digest for name,digest in hashes.items())
assert audit_sources(),"Source differs from prepared selection"
while request(a.url,"/health")["stage"]!="ready":
    time.sleep(2)
health=request(a.url,"/health")
if health.get("code_revision")!=benchmark_revision:
    raise RuntimeError("Server and benchmark Git revisions differ; no job submitted")
assert health["job"] is None or health["job"]["state"] in {"completed","failed","cancelled"}
if a.auto_retry:
    assert health["optimization"].get("retain_rejected_first"),"Retain rejected first candidates for first/final audit"
options={"scope":"full_image","preset":"auto","preserve_background":True,"first_pass_mode":"whole",
         "whole_prompt_profile":a.whole_prompt_profile,"resolution":a.input_resolution,"concurrency":a.concurrency,
         "extra_prompt":a.extra_prompt,
         "quality_check":True,"quality_batch_size":2,"quality_detail_patches":4,
         "quality_cross_review":a.cross_review,"quality_auto_retry":a.auto_retry,
         "quality_max_attempts":a.attempts,"quality_retry_resolution":a.retry_resolution,
         "overwrite":True,"seed":42}
started=time.monotonic()
job=request(a.url,"/jobs",{"input_dir":str(inputs),"output_dir":str(target),"options":options})["id"]
print(json.dumps({"job":job,"images":len(hashes),"selection_sha256":hashlib.sha256(
    (work/"source-selection.json").read_bytes()).hexdigest()}),flush=True)
first_completed_seconds=None
last=started
while True:
    health=request(a.url,"/health");state=health["job"]
    assert state["id"]==job,"Do not mix job histories"
    if first_completed_seconds is None and state.get("first_pass_total",0)==len(hashes):
        first_completed_seconds=round(time.monotonic()-started,3)
    if time.monotonic()-last>=30:
        print(json.dumps({"completed":state["completed"],"total":state["total"],
            "first_total":state.get("first_pass_total",0),"first_counts":state.get("first_pass_counts",{}),
            "final_counts":state.get("quality_counts",{}),"pending_retries":health.get("pending_retries"),
            "elapsed":round(time.monotonic()-started,1)}),flush=True)
        last=time.monotonic()
    if state["state"] in {"completed","failed","cancelled","interrupted"}:
        break
    if time.monotonic()-started>21600:
        raise TimeoutError(job)
    time.sleep(1)
elapsed=time.monotonic()-started
folder=Path.home()/".local/state/watermark-studio"/job
rows=[json.loads(line) for line in (folder/"results.jsonl").read_text().splitlines()]
assert len(rows)==len(hashes)==len({r["file"] for r in rows})
assert set(hashes)=={r["file"] for r in rows}
retained=[r for r in rows if r.get("first_rejected_candidate")]
retained_ok=all(Path(r["first_rejected_candidate"]).is_file() and
    hashlib.sha256(Path(r["first_rejected_candidate"]).read_bytes()).hexdigest()==r["first_rejected_sha256"] for r in retained)
assert retained_ok and not any(r.get("first_candidate_retention_error") for r in rows)
assert audit_sources(),"Immutable source changed"
sizes=[]
for row in rows:
    if row["ok"]:
        with Image.open(inputs/row["file"]) as before,Image.open(target/row["file"]) as after:
            if before.size!=after.size:
                sizes.append(row["file"])
histories=[h for r in rows for h in r.get("attempt_history",[]) or [r]]
first=[(r.get("attempt_history") or [r])[0] for r in rows]
assert state.get("first_pass_total",0)==len(hashes),"Incomplete first-pass accounting"
def generation_attempts(row):
    return sum(h.get("model_inferences",0)>0 for h in (row.get("attempt_history") or [row]))

assert subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()==benchmark_revision, \
    "Benchmark checkout changed during execution"
summary={"scope":("reused-source development; not holdout qualification" if a.development else
                   "automatic independent checks, not human-labelled accuracy"),
    "development_reuse":a.development,"job":job,
    "code_revision":health["code_revision"],"benchmark_revision":benchmark_revision,"options":options,
    "selection":{k:v for k,v in selection.items() if k not in {"source_sha256","sources"}},
    "images":len(hashes),"state":state["state"],"processing_failed":state["failed"],
    "seconds":round(elapsed,3),"images_per_minute":round(len(hashes)*60/elapsed,4),
    "first_completed_seconds":first_completed_seconds,"first_counts":state.get("first_pass_counts",{}),
    "first_total":state.get("first_pass_total",0),
    "final_counts":dict(collections.Counter(r.get("quality",{}).get("status","none") for r in rows)),
    "final_guard_counts":dict(collections.Counter(g for r in rows for g in r.get("quality",{}).get("guard_codes",[]))),
    "first_checker_before_cross_counts":dict(collections.Counter(
        ((h.get("quality") or {}).get("pre_cross_review") or h.get("quality") or {}).get("status","none") for h in first)),
    "source_vetting_protected_first":sum(v.get("status")=="protected_physical_text"
        for h in first for v in h.get("source_physical_vetting",[])),
    "first_source_scene_counts":dict(collections.Counter(h.get("source_scene","unknown") for h in first)),
    "first_guard_counts":dict(collections.Counter(g for h in first for g in (h.get("quality") or {}).get("guard_codes",[]))),
    "outside_change_attempts":sum("outside_repair_change" in (h.get("quality") or {}).get("guard_codes",[]) for h in histories),
    "first_cross_review_corrected":sum((h.get("quality") or {}).get("cross_review",{}).get("status")=="corrected" for h in first),
    "first_model_input_budgets":dict(collections.Counter(str(b) for h in first for b in h.get("model_input_budgets",[]))),
    "total_model_input_budgets":dict(collections.Counter(str(b) for h in histories for b in h.get("model_input_budgets",[]))),
    "successful_diffusion_forwards":round(sum(1/b for h in histories for b in h.get("batch_sizes",[]) if b),6),
    "generation_attempt_counts":dict(collections.Counter(str(generation_attempts(r)) for r in rows)),
    "final_passed_by_generation_attempts":dict(collections.Counter(str(generation_attempts(r))
        for r in rows if (r.get("quality") or {}).get("status")=="passed")),
    "first_rejected_later_accepted_unchanged":sum((r.get("quality") or {}).get("status")=="passed" and
        hashlib.sha256((target/r["file"]).read_bytes()).hexdigest()==r["first_rejected_sha256"] for r in retained),
    "best_attempt_counts":dict(collections.Counter(str(r.get("best_attempt",1)) for r in rows)),
    "attempt_counts":dict(collections.Counter(str(r.get("quality_attempts",1)) for r in rows)),
    "generation_attempt_errors":sum(not h["ok"] for h in histories),
    "attempt_oom_retries":sum(h.get("oom_retries",0) for h in histories),
    "peak_allocated_mib":max((h.get("peak_allocated_mib",0) for h in histories),default=0),
    "quality_attempt_errors":sum((h.get("quality") or {}).get("status")=="error" for h in histories),
    "retained_first_rejected":len(retained),"retained_first_unchanged":retained_ok,
    "source_inputs_unchanged":True,"unique_sources":len(set(hashes.values())),"bad_sizes":sizes,
    "acceptance_complete":state.get("acceptance_complete",False),"optimization":health["optimization"],
    "deployment":{k:health.get(k) for k in ["worker_count","batch_size","small_batch_size","batch_window",
        "quality_backend","quality_quantization","quality_phase_offload","quality_phase_offload_above",
        "separate_vae_decode","quality_rectangle_guard","quality_cross_full_context","capacity"]},
    "worker_count":health["worker_count"],"batch_size":health["batch_size"],
    "quality_weight_identity":health.get("quality_weight_identity"),
    "review_files":[r["file"] for r in rows if r.get("quality",{}).get("status")!="passed"]}
output.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({k:v for k,v in summary.items() if k not in {"review_files","selection","options","quality_weight_identity"}},
                 ensure_ascii=False),flush=True)
if state["state"]!="completed" or state["failed"] or sizes:
    raise RuntimeError("Incomplete/invalid benchmark; retained report is not a successful run")
