"""Read-only source re-localization of broad masks; no candidate generation/approval."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--snapshot",type=Path,required=True)
p.add_argument("--gpu",type=int,required=True)
p.add_argument("--profile",choices=["editor","verifier"],required=True)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--area",type=float,default=.35)
p.add_argument("--limit",type=int,default=16)
a=p.parse_args()
if a.output.exists() or a.gpu<0 or not .15<=a.area<=.6 or not 1<=a.limit<=32:
    p.error("Fresh report, explicit GPU and bounded source-only selection required")
os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch

from watermark_studio.config import DeploymentConfig
from watermark_studio.detection import locate_watermarks_batch
from watermark_studio.frozen_candidates import digest, record_digest
from watermark_studio.images import load_photo
from watermark_studio.model_identity import weight_fingerprint
from watermark_studio.quality_loading import load_quality_encoder, load_quality_processor
from watermark_studio.source_geometry import SOURCE_RELOCALIZATION_PROMPT

torch.set_num_threads(1)
assert torch.cuda.mem_get_info()[0]>25*2**30
root=Path.home()/".cache/watermark-models/models"
editor=root/"Qwen--Qwen-Image-2.1-Turbo/snapshots/master"
model=(editor/"text_encoder" if a.profile=="editor" else
       root/"Qwen--Qwen3-VL-4B-Instruct/snapshots/master")
began=time.monotonic()
encoder=load_quality_encoder(model,DeploymentConfig(quality_verifier=True,quality_backend="independent"),"cuda:0")
if a.profile=="editor":
    from transformers import Qwen3VLProcessor
    processor=Qwen3VLProcessor.from_pretrained(editor/"processor",local_files_only=True)
else:
    processor=load_quality_processor(model)
load_seconds=time.monotonic()-began
model_identity=weight_fingerprint(model)["sha256"]
summary=json.loads((a.snapshot/"summary.json").read_text())
manifest={r["file"]:r for r in json.loads((a.snapshot/"manifest.json").read_text())}
rows=list(map(json.loads,(a.snapshot/"results.jsonl").read_text().splitlines()))
area=lambda r:max([(g["box"][2]-g["box"][0])*(g["box"][3]-g["box"][1])
                   for g in r.get("repair_regions",[]) if g["kind"]=="panel"] or [0])
cases=[r for r in rows if r.get("source_scene")=="outdoor" and area(r)>=a.area]
cases.sort(key=area,reverse=True)
prompt=SOURCE_RELOCALIZATION_PROMPT
records=[]
for row in cases[:a.limit]:
    name=row["file"];source=Path(summary["input_dir"])/name
    assert digest(source)==manifest[name]["source_sha256"] and record_digest(row)==manifest[name]["record_sha256"]
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    at=time.monotonic()
    result=locate_watermarks_batch([load_photo(source)[0]],encoder,processor,"cuda:0",1536,1,True,prompt)[0]
    record={"file":name,"source_sha256":before,"original_regions":row["repair_regions"],
            "seconds":round(time.monotonic()-at,3)}
    if isinstance(result,Exception):
        record["error"]=str(result)[:200]
    else:
        record["regions"]=[r.model_dump() for r in result[0]]
    assert before==digest(source)
    records.append(record)
    print(json.dumps({"completed":len(records),"total":min(a.limit,len(cases))}),flush=True)
report={"scope":"source-only broad-mask localization; requires physical/carrier validation before adoption",
        "model_sha256":model_identity,"profile":a.profile,"physical_gpu":a.gpu,"code_revision":subprocess.check_output(
            ["git","rev-parse","--short","HEAD"],text=True).strip(),
        "source_area_threshold":a.area,"selected":len(cases),"tested":len(records),
        "load_seconds":round(load_seconds,3),"sources_unchanged":True,"rows":records}
a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!="rows"}),flush=True)
