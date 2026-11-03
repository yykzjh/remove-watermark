"""First-generation paired ROI conditioning probe; never retries or promotes QA results."""
import argparse
import collections
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

p=argparse.ArgumentParser(__doc__)
p.add_argument("--gpu",type=int,required=True)
p.add_argument("--variant",choices=["raw","prefill","split"],required=True)
p.add_argument("--baseline-job",required=True)
p.add_argument("--selection",type=Path,required=True)
p.add_argument("--work",type=Path,required=True)
p.add_argument("--prompt-profile",choices=["detailed","compact-zh"],default="detailed")
a=p.parse_args()
loaded_revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
assert a.gpu in (0,1,2,3)
assert not a.work.exists(), "Use a fresh output directory"
names=json.loads(a.selection.read_text())
assert 1<=len(names)<=128 and len(set(names))==len(names)
os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch

from watermark_studio.config import DeploymentConfig, ProcessingConfig
from watermark_studio.images import load_photo
from watermark_studio.quality_loading import load_quality_encoder, load_quality_processor
from watermark_studio.worker import load_pipeline, process_batch

if a.prompt_profile=="compact-zh":
    from watermark_studio import worker
    from watermark_studio.prompts import build_compact_focused_prompt
    worker.build_focused_prompt=build_compact_focused_prompt
    worker.build_small_overlay_prompt=lambda box,crop,extra="":build_compact_focused_prompt("panel",box,crop,extra)

assert torch.cuda.mem_get_info()[0]>40*2**30, "GPU must be free before probe"
state=Path.home()/".local/state/watermark-studio"
b=state/a.baseline_job
summary=json.loads((b/"summary.json").read_text())
original={r["file"]:r for r in map(json.loads,(b/"results.jsonl").read_text().splitlines())}
assert set(names)<=set(original)
cfg=DeploymentConfig.model_validate(json.loads(Path(".run/deployment.json").read_text())).model_copy(
    update={"gpus":[a.gpu],"prefill_overlays":a.variant!="raw",
            "prefill_original_semantics":a.variant=="split"})
opts=ProcessingConfig.model_validate(summary["options"]).model_copy(
    update={"quality_auto_retry":False,"quality_cross_review":False,"overwrite":False})
assert opts.first_pass_mode=="complete" and opts.context==.45 and opts.resolution==1024
a.work.mkdir(parents=True)
out=a.work/"output"
hashes={}
tasks=[]
root=Path(summary["input_dir"]).resolve()
for name in names:
    src=(root/name).resolve()
    assert src.is_relative_to(root) and src.is_file()
    with src.open("rb") as f:hashes[name]=hashlib.file_digest(f,"sha256").hexdigest()
    dst=out/name;dst.parent.mkdir(parents=True,exist_ok=True)
    tasks.append({"source":str(src),"destination":str(dst),"relative":name})
cache=Path(cfg.cache_dir)/"models"
model=cache/"Qwen--Qwen-Image-2.1-Turbo/snapshots/master"
load_start=time.monotonic()
pipe=load_pipeline(cfg,str(model))
pipe._quality_encoder=load_quality_encoder(cfg.quality_model,cfg,pipe._execution_device)
pipe._quality_processor=load_quality_processor(cfg.quality_model)
pipe._quality_weights_relation="different_weights"
print(json.dumps({"phase":"loaded","variant":a.variant,"seconds":time.monotonic()-load_start}),flush=True)
started=time.monotonic();rows=[];limits={}
for start in range(0,len(tasks),cfg.batch_window):
    group=tasks[start:start+cfg.batch_window]
    results=process_batch(group,opts,cfg,pipe,limits)
    assert len(results)==len(group)
    with (a.work/"results.jsonl").open("a") as f:
        for row in results:f.write(json.dumps(row,ensure_ascii=False)+"\n")
    rows.extend(results)
    print(json.dumps({"phase":"progress","variant":a.variant,"done":len(rows),
        "counts":dict(collections.Counter(r.get("quality",{}).get("status","error") for r in rows))}),flush=True)
bad_sizes=0
for task in tasks:
    with open(task["source"],"rb") as f:assert hashlib.file_digest(f,"sha256").hexdigest()==hashes[task["relative"]]
    if Path(task["destination"]).exists():
        bad_sizes+=load_photo(task["source"])[0].size!=load_photo(task["destination"])[0].size
counts=dict(collections.Counter(r.get("quality",{}).get("status","error") for r in rows))
report={"variant":a.variant,"prompt_profile":a.prompt_profile,"images":len(rows),"seconds":time.monotonic()-started,
    "counts":counts,"errors":sum(not r.get("ok") for r in rows),"bad_sizes":bad_sizes,
    "source_sha256":hashes,"selection_sha256":hashlib.sha256(a.selection.read_bytes()).hexdigest(),
    "config":cfg.model_dump(mode="json"),"options":opts.model_dump(mode="json"),
    "revision":loaded_revision,
    "baseline_review_to_pass":sum(original[r["file"]]["quality"]["status"]!="passed" and r.get("quality",{}).get("status")=="passed" for r in rows),
    "baseline_pass_to_review":sum(original[r["file"]]["quality"]["status"]=="passed" and r.get("quality",{}).get("status")!="passed" for r in rows),
    "single_gpu":True,"first_generation_only":True,"development_reuse":True}
(a.work/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in {"config","options","source_sha256"}}),flush=True)
