"""GPU generation capacity and optional verifier residency cycles; not end-to-end throughput."""
import argparse
import gc
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--gpu",type=int,required=True)
p.add_argument("--source",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--resolution",type=int,choices=[1024,1536,2048],default=1024)
p.add_argument("--batches",default="2,3,4")
p.add_argument("--quality-phase-offload",action="store_true")
p.add_argument("--whole-prompt-profile",choices=["fidelity","neutral","concise"],default="fidelity")
a=p.parse_args()
batches=[int(s) for s in a.batches.split(",")]
if not batches or len(set(batches))!=len(batches) or any(n<1 or n>4 for n in batches):
    p.error("Distinct batch sizes 1..4 required")
if a.output.exists(): p.error("Fresh report required")
os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["PYTORCH_ALLOC_CONF"]="expandable_segments:True"
import torch

from watermark_studio.batching import dimensions, independent_batch, preencode_window
from watermark_studio.config import DeploymentConfig
from watermark_studio.images import load_photo
from watermark_studio.prompts import build_full_prompt
from watermark_studio.quality_loading import (
    load_quality_encoder,
    park_quality_encoder,
    restore_quality_encoder,
)
from watermark_studio.worker import load_pipeline

torch.set_num_threads(1)
assert torch.cuda.mem_get_info()[0]>35*2**30
before=hashlib.sha256(a.source.read_bytes()).hexdigest()
settings=json.loads(Path(".run/deployment.json").read_text())|{"gpus":[a.gpu]}
if a.quality_phase_offload:
    settings["quality_phase_offload"]=True
cfg=DeploymentConfig.model_validate(settings)
pipe=load_pipeline(cfg,str(Path(cfg.model).expanduser()))
verifier=load_quality_encoder(str(Path(cfg.quality_model).expanduser()),cfg,"cuda:0")
assert all(param.device.type=="cuda" for param in verifier.parameters())
photo=load_photo(a.source)[0]
rows=[]
for size in batches:
    for repeat in range(3):
        items=[{"image":photo,"prompt":build_full_prompt(profile=a.whole_prompt_profile),
                "seed":42+i,"size":dimensions(photo,a.resolution)} for i in range(size)]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cycle_began=time.monotonic()
        resident_peak=torch.cuda.memory_allocated()
        parked=restored=0.
        error=None
        if a.quality_phase_offload:
            at=time.monotonic()
            verifier=park_quality_encoder(verifier)
            torch.cuda.empty_cache()
            parked=time.monotonic()-at
        torch.cuda.reset_peak_memory_stats()
        began=time.monotonic()
        try:
            preencode_window(pipe,items,4)
            outputs=independent_batch(pipe,items,a.resolution,True)
            assert len(outputs)==size
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError as failure:
            error=str(failure)[:160]
        generation=time.monotonic()-began
        generation_peak=torch.cuda.max_memory_allocated()
        for item in items:
            item.pop("_encoded_prompt",None)
        del items
        gc.collect();torch.cuda.empty_cache()
        if a.quality_phase_offload:
            at=time.monotonic()
            verifier=restore_quality_encoder(verifier,"cuda:0")
            restored=time.monotonic()-at
        assert all(param.device.type=="cuda" for param in verifier.parameters())
        cycle=time.monotonic()-cycle_began
        row={"physical_gpu":a.gpu,"resolution":a.resolution,"batch":size,"repeat":repeat,
             "warmup":repeat==0,"ok":error is None,"seconds":round(generation,4),
             "cycle_seconds":round(cycle,4),"park_seconds":round(parked,4),
             "restore_seconds":round(restored,4),
             "generation_peak_mib":round(generation_peak/2**20),
             "peak_allocated_mib":round(max(resident_peak,torch.cuda.max_memory_allocated())/2**20),
             "error":error}
        rows.append(row);print(json.dumps(row),flush=True)
        if error:break
assert before==hashlib.sha256(a.source.read_bytes()).hexdigest()
report={"scope":"GPU generation+encoder and optional residency cycles; no detector/QA/IO",
        "physical_gpu":a.gpu,"resolution":a.resolution,"source_unchanged":True,
        "verifier_resident_at_cycle_boundaries":True,"phase_offload":a.quality_phase_offload,
        "host_pinned_mib":round(getattr(verifier,"_quality_host_pinned_bytes",0)/2**20),
        "prompt_profile":a.whole_prompt_profile,"saved_sigma_count":len(pipe.config.sample_sigmas),
        "rows":rows}
for label,key in [("median_seconds_per_image","seconds"),
                  ("median_cycle_seconds_per_image","cycle_seconds")]:
    report[label]={str(n):round(statistics.median(r[key]/n for r in rows
        if r["batch"]==n and not r["warmup"] and r["ok"]),4) for n in batches
        if any(r["batch"]==n and not r["warmup"] and r["ok"] for r in rows)}
a.output.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k!="rows"}),flush=True)
