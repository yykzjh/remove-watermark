"""Read-only source classification speed/invariance probe; service must be stopped."""
import argparse
import collections
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--gpu",type=int,required=True)
p.add_argument("--job",required=True)
p.add_argument("--selection",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--deployment",type=Path,default=Path(".run/deployment.json"))
p.add_argument("--batch-size",type=int,choices=[2,3,4],default=4)
p.add_argument("--calibration",type=int,default=64)
p.add_argument("--shards",type=int,choices=range(1,9),default=1)
p.add_argument("--shard-index",type=int,default=0)
a=p.parse_args()
if a.output.exists() or not 8<=a.calibration<=128 or not 0<=a.shard_index<a.shards:
    p.error("Fresh report and 8..128 paired calibration required")
os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
os.environ["TOKENIZERS_PARALLELISM"]="false"
if json.loads(a.deployment.read_text()).get("expandable_segments"):
    os.environ["PYTORCH_ALLOC_CONF"]="expandable_segments:True"

import torch

from watermark_studio.config import DeploymentConfig, Region
from watermark_studio.images import load_photo
from watermark_studio.quality_loading import load_quality_encoder, load_quality_processor
from watermark_studio.source_vetting import vet_sources
from watermark_studio.worker import load_pipeline

assert torch.cuda.mem_get_info()[0]>35*2**30,"Stop owned service and verify GPU is free first"
folder=Path.home()/".local/state/watermark-studio"/a.job
job=json.loads((folder/"summary.json").read_text())
assert job["state"]=="completed","Only completed qualification job can supply frozen references"
inputs=Path(job["input_dir"]).resolve()
rows=list(map(json.loads,(folder/"results.jsonl").read_text().splitlines()))
selection=json.loads(a.selection.read_text())
assert len(rows)==len(selection["source_sha256"])
fields=("physical_scene_text","digital_overlay","object_attachment","surface_evidence","uncertain")
items=[];missing=0
for row in rows:
    first=(row.get("attempt_history") or [row])[0]
    for evidence in first.get("source_physical_vetting",[]):
        baseline=evidence.get("editor")
        if not baseline or any(type(baseline.get(f)) is not bool for f in fields):
            missing+=1;continue
        path=(inputs/row["file"]).resolve()
        assert path.is_relative_to(inputs)
        items.append({"file":row["file"],"path":path,"box":evidence["box"],"flags":{f:baseline[f] for f in fields},
                      "status":evidence["status"],"requires_review":bool(evidence.get("requires_review"))})
full_candidates=len(items)
selected_candidate_ids=[i for i in range(full_candidates) if i%a.shards==a.shard_index]
items=[items[i] for i in selected_candidate_ids]
assert len(items)>=8
digests={i["file"]:selection["source_sha256"][i["file"]] for i in items}
def audit():
    return all(hashlib.sha256((inputs/name).read_bytes()).hexdigest()==digest for name,digest in digests.items())
assert audit()
cfg=DeploymentConfig.model_validate(json.loads(a.deployment.read_text())|{"gpus":[a.gpu]})
torch.set_num_threads(cfg.cpu_threads)
started=time.monotonic()
pipe=load_pipeline(cfg,str(Path(cfg.model).expanduser()))
verifier=load_quality_encoder(str(Path(cfg.quality_model).expanduser()),cfg,"cuda:0")
processor=load_quality_processor(str(Path(cfg.quality_model).expanduser()))
print(json.dumps({"stage":"loaded","seconds":round(time.monotonic()-started,3),"candidates":len(items)}),flush=True)

def run(indices,batch_size):
    observed={};sizes=[];errors=[];started=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for offset in range(0,len(indices),batch_size):
        keys=indices[offset:offset+batch_size]
        images=[load_photo(items[k]["path"])[0] for k in keys]
        detections=[([Region(kind="panel",box=items[k]["box"])],"frozen candidate") for k in keys]
        _,audits=vet_sources(images,detections,pipe.text_encoder,pipe.processor,verifier,processor,
                             pipe._execution_device,"different_weights",batch_size)
        for position,key in enumerate(keys):
            record=audits[position][0]
            primary=record.get("editor",{})
            observed[key]={"flags":{f:primary.get(f) for f in fields},"status":record["status"],
                           "requires_review":bool(record.get("requires_review"))}
            sizes.append(record.get("primary_batch_size",0))
            if record.get("error"):errors.append({"key":key,"error":record["error"]})
        if offset%64==0:
            print(json.dumps({"stage":"classifying","batch":batch_size,"done":min(offset+batch_size,len(indices)),
                              "total":len(indices)}),flush=True)
    torch.cuda.synchronize()
    return observed,{"items":len(indices),"seconds":round(time.monotonic()-started,3),
        "peak_allocated_mib":round(torch.cuda.max_memory_allocated()/2**20),
        "primary_batch_participation":dict(collections.Counter(sizes)),"errors":errors}

rng=random.Random(202611042)
special=[i for i,item in enumerate(items) if item["status"]!="retained_for_edit"]
special_set=set(special)
normal=[i for i in range(len(items)) if i not in special_set]
rng.shuffle(special);rng.shuffle(normal)
priority=special[:a.calibration//2]
calibration=(priority+normal+[i for i in special if i not in set(priority)])[:min(a.calibration,len(items))]
# Warm identical roles, precision, and full deployment residency before paired timing.
run(calibration[:min(4,len(calibration))],a.batch_size)
serial,serial_stats=run(calibration,1)
batched,batch_stats=run(calibration,a.batch_size)
remaining=[i for i in range(len(items)) if i not in set(calibration)]
extra,extra_stats=run(remaining,a.batch_size)
all_batched=batched|extra

def differing(observed):
    changed=[]
    for key,value in observed.items():
        expected={k:items[key][k] for k in ["flags","status","requires_review"]}
        if value!=expected:
            changed.append({"key":key,"file":items[key]["file"],"expected":expected,"observed":value})
    return changed
serial_diff=differing(serial);batch_diff=differing(all_batched)
paired_diff=[key for key in calibration if serial[key]!=batched[key]]
same_sources=audit()
assert same_sources
speed=serial_stats["seconds"]/batch_stats["seconds"]
report={"job":a.job,"code_revision":job.get("code_revision"),"probe_revision":subprocess.check_output(
    ["git","rev-parse","--short","HEAD"],text=True).strip(),"gpu":a.gpu,"items":len(items),
    "source_images":len(digests),"full_candidates":full_candidates,
    "shards":a.shards,"shard_index":a.shard_index,"selected_candidate_ids":selected_candidate_ids,"baseline_missing_primary_reports":missing,"precision":cfg.dtype,
    "full_deployment_weights_resident":True,"batch_size":a.batch_size,"serial_calibration":serial_stats,
    "batched_calibration":batch_stats,"batched_remaining":extra_stats,"calibration_speedup":round(speed,4),
    "serial_reference_differences":serial_diff,"batched_reference_differences":batch_diff,
    "paired_calibration_difference_keys":paired_diff,"source_hashes_unchanged":same_sources,
    "recommend_batching":not (missing or serial_diff or batch_diff or paired_diff or serial_stats["errors"]
        or batch_stats["errors"] or extra_stats["errors"]) and speed>1.05}
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k not in {"serial_reference_differences",
    "batched_reference_differences","paired_calibration_difference_keys"}}|
    {"serial_reference_difference_count":len(serial_diff),"batched_reference_difference_count":len(batch_diff),
     "paired_calibration_differences":len(paired_diff)},ensure_ascii=False),flush=True)
