"""Resident-verifier GPU batch capacity probe; no QA/IO throughput claim."""
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
a=p.parse_args()
if a.output.exists(): p.error("Fresh report required")
os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["PYTORCH_ALLOC_CONF"]="expandable_segments:True"
import torch

from watermark_studio.batching import dimensions, independent_batch, preencode_window
from watermark_studio.config import DeploymentConfig
from watermark_studio.images import load_photo
from watermark_studio.prompts import build_full_prompt
from watermark_studio.quality_loading import load_quality_encoder
from watermark_studio.worker import load_pipeline

torch.set_num_threads(1)
assert torch.cuda.mem_get_info()[0]>35*2**30
before=hashlib.sha256(a.source.read_bytes()).hexdigest()
cfg=DeploymentConfig.model_validate(json.loads(Path(".run/deployment.json").read_text())|{"gpus":[a.gpu]})
pipe=load_pipeline(cfg,str(Path(cfg.model).expanduser()))
verifier=load_quality_encoder(str(Path(cfg.quality_model).expanduser()),cfg,"cuda:0")
assert all(p.device.type=="cuda" for p in verifier.parameters())
photo=load_photo(a.source)[0]
rows=[]
for size in (2,3,4):
 for repeat in range(3):
  items=[{"image":photo,"prompt":build_full_prompt(),"seed":42+i,"size":dimensions(photo,1024)} for i in range(size)]
  torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();began=time.monotonic();error=None
  try:
   preencode_window(pipe,items,4)
   outputs=independent_batch(pipe,items,1024,True)
   assert len(outputs)==size
   torch.cuda.synchronize()
  except torch.cuda.OutOfMemoryError as failure:
   error=str(failure)[:160]
  row={"batch":size,"repeat":repeat,"warmup":repeat==0,"ok":error is None,
       "seconds":round(time.monotonic()-began,4),"peak_allocated_mib":round(torch.cuda.max_memory_allocated()/2**20),
       "error":error}
  rows.append(row);print(json.dumps(row),flush=True)
  for item in items:item.pop("_encoded_prompt",None)
  del items;gc.collect();torch.cuda.empty_cache()
  if error:break
assert before==hashlib.sha256(a.source.read_bytes()).hexdigest()
report={"scope":"GPU generation+encoder only; verifier weights resident, no detector/QA/IO",
 "source_unchanged":True,"verifier_resident":True,"rows":rows,
 "median_seconds_per_image":{str(n):round(statistics.median(r["seconds"]/n for r in rows if r["batch"]==n and not r["warmup"] and r["ok"]),4) for n in (2,3,4) if any(r["batch"]==n and not r["warmup"] and r["ok"] for r in rows)}}
a.output.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k!="rows"}),flush=True)
