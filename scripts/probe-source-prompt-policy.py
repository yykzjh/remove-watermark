"""Read-only source-only prompt routing screen; no generated pixels are selected in production."""
import argparse
import json
import os
import time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--gpu",type=int,required=True)
p.add_argument("--fidelity-job",required=True)
p.add_argument("--concise-job",required=True)
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
if a.output.exists(): p.error("Fresh report required")
os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch

from watermark_studio.config import DeploymentConfig
from watermark_studio.images import load_photo
from watermark_studio.quality_loading import load_quality_encoder
from watermark_studio.source_brief import inspect_sources

torch.set_num_threads(1)
assert torch.cuda.mem_get_info()[0]>35*2**30
root=Path.home()/".local/state/watermark-studio"
def read(job):
 s=json.loads((root/job/"summary.json").read_text())
 assert s["state"]=="completed"
 rows={r["file"]:r for r in map(json.loads,(root/job/"results.jsonl").read_text().splitlines())}
 return s,rows
summary,fid=read(a.fidelity_job);_,short=read(a.concise_job)
assert set(fid)==set(short)
model=Path.home()/".cache/watermark-models/models/Qwen--Qwen-Image-2.1-Turbo/snapshots/master"
cfg=DeploymentConfig(quality_verifier=True,quality_backend="independent")
began=time.monotonic()
encoder=load_quality_encoder(str(model/"text_encoder"),cfg,"cuda:0")
from transformers import Qwen3VLProcessor

processor=Qwen3VLProcessor.from_pretrained(str(model/"processor"),local_files_only=True)
loaded=time.monotonic()-began
rows=[];began=time.monotonic()
for name in sorted(fid):
 photo=load_photo(Path(summary["input_dir"])/name)[0]
 brief=inspect_sources([photo],encoder,processor,"cuda:0",1)[0]
 # Predeclared material-only routing; unknown, paper, vegetation and people use fidelity.
 choose=brief.get("status")=="ok" and brief.get("uncertain") is False and brief.get("mineral") is True and all(brief.get(k) is False for k in ("paper","vegetation","people"))
 selected=(short if choose else fid)[name]
 rows.append({"file":name,"source_brief":brief,"use_concise":choose,
              "fidelity":fid[name]["quality"]["status"],"concise":short[name]["quality"]["status"],
              "selected":selected["quality"]["status"]})
 if len(rows)%16==0: print(json.dumps({"classified":len(rows),"total":len(fid)}),flush=True)
report={"scope":"development replay, source-only predeclared policy; not held-out accuracy or fresh generation",
        "images":len(rows),"load_seconds":round(loaded,3),"seconds":round(time.monotonic()-began,3),
        "chosen_concise":sum(r["use_concise"] for r in rows),
        "fidelity_passed":sum(r["fidelity"]=="passed" for r in rows),
        "policy_passed":sum(r["selected"]=="passed" for r in rows),
        "gained":[r["file"] for r in rows if r["fidelity"]!="passed" and r["selected"]=="passed"],
        "lost":[r["file"] for r in rows if r["fidelity"]=="passed" and r["selected"]!="passed"],
        "rows":rows}
a.output.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k!="rows"}),flush=True)
