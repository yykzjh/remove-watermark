"""First-generation acceptance, with retries/appeals off and fresh output roots."""
import argparse
import collections
import hashlib
import itertools
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path

from PIL import Image

from watermark_studio.cli import request

parser=argparse.ArgumentParser()
parser.add_argument("--url",default="http://127.0.0.1:8765")
parser.add_argument("--tag",required=True)
parser.add_argument("--subset",choices=["hard","all","holdout","replay","frozen"],default="hard")
parser.add_argument("--images",type=int,default=64)
parser.add_argument("--seed",type=int,default=20261101)
parser.add_argument("--exclude-benchmarks",nargs="*",default=[])
parser.add_argument("--cross-review",action="store_true")
parser.add_argument("--generation-only",action="store_true")
parser.add_argument("--preserve-background",action="store_true",help="Explicit legacy whole-image composite comparison; default whole is direct")
parser.add_argument("--concurrency",type=int,default=20)
parser.add_argument("--quality-batch-size",type=int,choices=[1,2,3,4],default=2)
parser.add_argument("--files",nargs="*")
parser.add_argument("--select-model-review",action="store_true",help="Development replay: generator VLM rejections plus deterministic model-passed controls")
parser.add_argument("--select-review",action="store_true",help="Frozen job: all unaccepted rows plus deterministic passed controls")
parser.add_argument("--review-controls",type=int,default=32)
parser.add_argument("--source-job",help="Completed job ID for a frozen source replay; requires --files")
parser.add_argument("--blending",choices=["harmonic","harmonic-fine"],default="harmonic")
parser.add_argument("--whole-prompt-profiles",nargs="+",choices=["baseline","fidelity","scene"],default=["baseline"])
parser.add_argument("--profiles",nargs="+",default=["whole-1024","focused-1024","thorough-1024"])
args=parser.parse_args()
benchmark_revision=subprocess.check_output(['git','rev-parse','--short','HEAD'],text=True).strip()
if not re.fullmatch(r"[a-zA-Z0-9_-]+",args.tag) or not 1<=args.images<=10000:
    parser.error("Use a simple tag and 1..10000 images")
root=Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031")
base=root/"quality-benchmark-cross-review-v7-64"
work=root/("first-pass-"+args.tag)
work.mkdir(parents=True,exist_ok=False)
inputs=work/"input";inputs.mkdir()
prior_hashes=set()
automatic_exclusions=[]
if args.subset in {"hard","all"}:
    names=(["000.jpg","008.jpg","043.jpg","028.jpg","031.jpg","011.jpg","019.jpg","017.jpg","054.jpg",
            "001.jpg","002.jpg","010.jpg","057.jpg","063.jpg"] if args.subset=="hard" else
           [p.name for p in sorted((base/"input").iterdir())][:args.images])
    paths=[base/"input"/name for name in names]
elif args.subset=="replay":
    old=root/"first-pass-holdout-neutral-v1"
    records=[json.loads(line) for line in (old/"results.jsonl").read_text().splitlines()]
    bad={f["file"] for r in records for f in r["failures"]}
    controls=[p.name for p in sorted((old/"input").iterdir()) if p.name not in bad][:12]
    hard=["000.jpg","008.jpg","043.jpg","028.jpg","031.jpg","011.jpg","019.jpg","017.jpg","054.jpg",
          "001.jpg","002.jpg","010.jpg","057.jpg","063.jpg"]
    names=hard+sorted(bad)+controls
    paths=[base/"input"/name for name in hard]+[old/"input"/name for name in sorted(bad)+controls]
elif args.subset=="frozen":
    if not args.source_job or not re.fullmatch(r"[a-f0-9]{32}",args.source_job) or (not args.files and not args.select_review and not args.select_model_review):
        parser.error("Frozen replay requires a completed --source-job and explicit --files or --select-review")
    folder=Path.home()/".local/state/watermark-studio"/args.source_job
    snapshot=json.loads((folder/"summary.json").read_text())
    if snapshot["state"]!="completed":
        parser.error("Source job must be completed")
    frozen=Path(snapshot["input_dir"]).resolve()
    records=[json.loads(line) for line in (folder/"results.jsonl").read_text().splitlines()]
    if args.select_review and args.select_model_review:
        parser.error("Choose one review selection method")
    if args.select_review or args.select_model_review:
        if args.files or not 0<=args.review_controls<=1024:
            parser.error("Review selection excludes --files and requires 0..1024 controls")
        criterion="model_status" if args.select_model_review else "status"
        rejected=sorted(r["file"] for r in records if r.get("quality",{}).get(criterion)!="passed")
        controls=sorted(r["file"] for r in records if r.get("quality",{}).get(criterion)=="passed")
        controls=random.Random(args.seed).sample(controls,min(args.review_controls,len(controls)))
        args.files=rejected+sorted(controls)
    available={r["file"] for r in records}
    if not set(args.files).issubset(available) or len(set(args.files))!=len(args.files):
        parser.error("Frozen source names must exist and be unique")
    paths=[(frozen/name).resolve() for name in args.files]
    if any(not p.is_relative_to(frozen) or not p.is_file() for p in paths):
        parser.error("Frozen sources must stay inside the original input root")
    names=list(args.files)
    # Flattening is never allowed: nested replay inputs retain their exact names.
else:
    data=root/"八标照片集"
    manifest=json.loads((data/".download-state/manifest.json").read_text())
    original="/codex-download-1mQf6F6g-20261031/八标照片集/"
    from watermark_studio.evaluation import exclusion_sources
    seen,automatic_exclusions=exclusion_sources(root,base,args.exclude_benchmarks)
    candidates=[data/r["path"].removeprefix(original) for r in manifest
                if Path(r["path"]).suffix.lower() in {".jpg",".jpeg",".png"}]
    seen_hashes={hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in seen if Path(p).is_file()}
    prior_hashes=set(seen_hashes)
    random.Random(args.seed).shuffle(candidates)
    paths=[]
    for p in candidates:
        if str(p) in seen or not p.is_file() or not p.stat().st_size:
            continue
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if digest in seen_hashes: continue
        try:
            with Image.open(p) as im: im.load()
        except OSError:
            continue
        paths.append(p)
        seen_hashes.add(digest)
        if len(paths)==args.images: break
    names=[f"{i:04d}"+p.suffix for i,p in enumerate(paths)]
if args.files:
    selected=[(p,n) for p,n in zip(paths,names) if n in args.files]
    paths=[p for p,n in selected]; names=[n for p,n in selected]
assert len(paths)==len(names) and paths
if args.subset=="holdout" and not args.files and len(paths)!=args.images:
    raise ValueError("Incomplete requested holdout selection")
for p,name in zip(paths,names):
    (inputs/name).parent.mkdir(parents=True,exist_ok=True)
    if not (inputs/name).exists(): os.link(p,inputs/name)
(work/"sources.json").write_text(json.dumps([str(p) for p in paths],ensure_ascii=False))
source_hashes={name:hashlib.sha256((inputs/name).read_bytes()).hexdigest() for name in names}
if args.subset=="holdout" and (len(set(source_hashes.values()))!=len(paths)
        or set(source_hashes.values())&prior_hashes):
    raise ValueError("Holdout hashes must be unique and disjoint from all excluded inputs")
selection={"benchmark_revision":benchmark_revision,"automatic_exclusions":automatic_exclusions,"subset":args.subset,"sampling_seed":args.seed,"select_model_review":args.select_model_review,"images":len(paths),
    "exclusions":args.exclude_benchmarks,"prior_sha256_overlap":len(set(source_hashes.values())&prior_hashes),
    "unique_sha256":len(set(source_hashes.values())),"source_sha256":source_hashes,
    "sources":[str(p) for p in paths]}
selection_text=json.dumps(selection,ensure_ascii=False,sort_keys=True)
(work/"source-selection.json").write_text(selection_text)
selection_sha256=hashlib.sha256(selection_text.encode()).hexdigest()

while request(args.url,"/health")["stage"]!="ready": time.sleep(2)
for profile,whole_prompt_profile in itertools.product(args.profiles,args.whole_prompt_profiles):
    match=re.fullmatch(r"(whole|focused|thorough|complete)(?:@([0-9.]+))?-([0-9]+)",profile)
    if not match: raise ValueError("Invalid profile")
    mode,context,budget=match.groups();budget=int(budget);context=float(context or .6)
    profile=profile+("--"+whole_prompt_profile if len(args.whole_prompt_profiles)>1 or whole_prompt_profile!="baseline" else "")
    target=work/profile
    if target.exists(): raise RuntimeError("Fresh tag/output required; existing accepted outputs must not bias first-pass metrics")
    preserve_background=mode!="whole" or args.preserve_background
    started=time.monotonic()
    job=request(args.url,"/jobs",{"input_dir":str(inputs),"output_dir":str(target),"options":{
        "scope":"full_image","whole_prompt_profile":whole_prompt_profile,"concurrency":args.concurrency,"resolution":budget,"preserve_background":preserve_background,"first_pass_mode":mode,"context":context,
        "repair_blending":args.blending,"quality_check":not args.generation_only,"quality_batch_size":args.quality_batch_size,
        "quality_detail_patches":4,"quality_cross_review":args.cross_review,"quality_auto_retry":False,"seed":42,"overwrite":True}})["id"]
    last_progress=started
    while True:
        state=request(args.url,"/health");status=state["job"]
        assert status["id"]==job
        if status["state"] in {"completed","failed","cancelled"}: break
        if time.monotonic()-last_progress>=30:
            print(json.dumps({"profile":profile,"completed":status["completed"],"total":status["total"],
                "first_counts":status.get("first_pass_counts",{}),"elapsed":round(time.monotonic()-started,1)}),flush=True)
            last_progress=time.monotonic()
        if time.monotonic()-started>10800: raise TimeoutError(job)
        time.sleep(.5)
    rows=[json.loads(line) for line in (Path.home()/".local/state/watermark-studio"/job/"results.jsonl").read_text().splitlines()]
    if len(rows)!=len(paths) or {r["file"] for r in rows}!=set(names):
        raise ValueError("Incomplete or duplicate first-pass image results")
    elapsed=time.monotonic()-started
    audit_started=time.monotonic()
    source_unchanged=all(hashlib.sha256((inputs/name).read_bytes()).hexdigest()==digest
                         for name,digest in source_hashes.items())
    if not source_unchanged:
        raise ValueError("Frozen source input changed during generation; benchmark invalid")
    source_audit_seconds=round(time.monotonic()-audit_started,3)
    if any(r.get("quality_attempts",1)>1 for r in rows):
        raise ValueError("A generation retry cannot count as a first-pass result")
    counts=collections.Counter(r.get("quality",{}).get("status","none") for r in rows)
    record={"processing_path":("whole_direct" if mode=="whole" and not preserve_background else "whole_composite" if mode=="whole" else "region_repair"),
        "preserve_background":preserve_background,"scope":"full_image","whole_prompt_profile":whole_prompt_profile,"concurrency":args.concurrency,
        "deployment":{"batch_size":state.get("batch_size"),"small_batch_size":state.get("small_batch_size"),
                      "batch_window":state.get("batch_window"),"worker_count":state.get("worker_count"),
                      "optimization":state.get("optimization")},
        "benchmark_revision":benchmark_revision,"profile":profile,"tag":args.tag,"subset":args.subset,"images":len(paths),"job":job,
        "blending":args.blending,"context":context,"seed":args.seed,"cross_review":args.cross_review,"code_revision":state.get("code_revision","unrecorded"),"quality_weight_identity":state.get("quality_weight_identity"),
        "quality_model":state.get("quality_model"),"quality_backend":state.get("quality_backend"),"seconds":round(elapsed,3),"source_inputs_unchanged":source_unchanged,"source_hash_audit_seconds":source_audit_seconds,
        "source_selection_sha256":selection_sha256,"sampling_seed":args.seed,"model_seed":42,"first_counts":dict(counts),
        "first_pass_rate":None if args.generation_only else round(counts["passed"]/len(paths),5),
        "model_status_counts":dict(collections.Counter(r.get("quality",{}).get("model_status","unknown") for r in rows)),
        "guard_counts":dict(collections.Counter(g for r in rows for g in r.get("quality",{}).get("guard_codes",[]))),
        "outside_change_tiles_sum":sum(r.get("quality",{}).get("outside_repair_change",{}).get("tiles",0) for r in rows),
        "outside_change_max_mae_mean":round(sum(r.get("quality",{}).get("outside_repair_change",{}).get("max_tile_mae",0) for r in rows)/len(rows),4),
        "generation_only":args.generation_only,"quality_batch_size":args.quality_batch_size,"processing_failed":status["failed"],
        "model_inferences":sum(r.get("model_inferences",0) for r in rows),
        "rows_with_oom_recovery":sum(r.get("oom_retries",0)>0 for r in rows),
        "max_oom_retries_in_one_worker_request":max((r.get("oom_retries",0) for r in rows),default=0),
        "prefilled_model_inputs":sum(bool(p) for r in rows for p in r.get("conditioning_prefill",[])),
        "prefill_overlays":state.get("optimization",{}).get("prefill_overlays",False),
        "edit_batch_participation":dict(collections.Counter(str(b) for r in rows for b in r.get("batch_sizes",[]))),
        "successful_diffusion_forwards":round(sum(1/b for r in rows for b in r.get("batch_sizes",[])),6),
        "small_batch_size":state.get("small_batch_size",0),
        "quality_rectangle_guard":state.get("quality_rectangle_guard",False),
        "rectangle_guard_reviews":sum("repair_rectangle_phase" in (r.get("quality",{}).get("guard_codes") or []) for r in rows),
        "rectangle_guard_cpu_seconds_sum":round(sum(r.get("quality",{}).get("repair_rectangle_phase",{}).get("seconds",0) for r in rows),6),
        "residual_geometry_requeries":sum(r.get("quality",{}).get("residual_scan_requery",{}).get("attempts",0) for r in rows),
        "generation_retries":sum(max(0,r.get("quality_attempts",1)-1) for r in rows),
        "input_budgets":dict(collections.Counter(str(n) for r in rows for n in r.get("model_input_budgets",[]))),
        "source_geometry_requeries":sum((r.get("source_geometry_requery") or {}).get("attempts",0) for r in rows),
        "core_feather_images":sum(bool(r.get("protected_core_boxes")) for r in rows),
        "source_brief_errors":sum((r.get("source_brief") or {}).get("status")=="error" for r in rows),
        "source_brief_seconds_sum":round(sum((r.get("source_brief") or {}).get("seconds",0) for r in rows),4),
        "source_detail_views":sum(r.get("source_detail_views",0) for r in rows),
        "bad_sizes":sum(Image.open(inputs/r["file"]).size!=Image.open(target/r["file"]).size for r in rows if r["ok"]),
        "quality_errors":sum(r.get("quality",{}).get("status")=="error" for r in rows),
        "peak_allocated_mib":max(r.get("peak_allocated_mib",0) for r in rows),
        "failures":[{"file":r["file"],"status":r.get("quality",{}).get("status"),
                     "reason":r.get("quality",{}).get("reason"),"guards":r.get("quality",{}).get("guard_codes",[])}
                    for r in rows if not args.generation_only and r.get("quality",{}).get("status")!="passed"]}
    with (work/"results.jsonl").open("a") as f:f.write(json.dumps(record,ensure_ascii=False)+"\n")
    print(json.dumps({k:v for k,v in record.items() if k!="failures"}|
        {"failure_examples":record["failures"][:2]},ensure_ascii=False),flush=True)
    if status["failed"]:raise RuntimeError("Processing failed")
