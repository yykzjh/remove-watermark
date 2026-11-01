"""Isolate raw generation from composition on frozen native-scale defect cases."""
import argparse, hashlib, json, os, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("--gpu",type=int,required=True)
p.add_argument("--cases",type=Path,required=True)
p.add_argument("--input",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--no-compile",action="store_true")
p.add_argument("--quantization",choices=["none","float8"],default="float8")
p.add_argument("--granularity",choices=["tensor","row"],default="tensor")
a=p.parse_args()
if a.output.exists(): p.error("Fresh output required")
os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
os.environ["TOKENIZERS_PARALLELISM"]="false"
import numpy as np
import torch
from watermark_studio.config import DeploymentConfig, Region
from watermark_studio.worker import load_pipeline
from watermark_studio.images import load_photo, harmonize_repair
from watermark_studio.batching import independent_batch, preencode_window
from watermark_studio.prompts import build_full_prompt, scene_instruction
torch.set_num_threads(1)
assert torch.cuda.mem_get_info()[0]>35*2**30
cfg=DeploymentConfig.model_validate(json.loads(Path(".run/deployment.json").read_text())|{
    "gpus":[a.gpu],"compile":not a.no_compile,"quality_verifier":False,
    "quantization":a.quantization,"float8_granularity":a.granularity})
pipe=load_pipeline(cfg,str(Path(cfg.model).expanduser()))
cases=json.loads(a.cases.read_text())[:2]
a.output.mkdir(parents=True)
items=[]
for row in cases:
 source=load_photo(a.input/row["file"])[0]
 stable=int.from_bytes(hashlib.sha256(row["file"].encode()).digest()[:4],"big")
 items.append({"image":source,"prompt":build_full_prompt(scene_instruction("outdoor"),"fidelity"),"seed":(42+stable)%2**32})
for label,bs,kv in [("single",1,True),("batch",2,True),("batch_no_kv",2,False)]:
 current=[dict(item) for item in items]
 if bs==2: preencode_window(pipe,current,4)
 started=time.monotonic()
 outputs=[]
 for offset in range(0,len(current),bs):
  outputs.extend(independent_batch(pipe,current[offset:offset+bs],1024,kv))
 rows=[]
 for i,(result,case) in enumerate(zip(outputs,cases)):
  result.save(a.output/f"{label}-{i}-raw.png")
  source=items[i]["image"]
  result=result.resize(source.size)
  regions=[Region.model_validate(r) for r in case["regions"]]
  fixed=harmonize_repair(source,result,regions)
  fixed.save(a.output/f"{label}-{i}-harmonic.png")
  arr=np.asarray(result,dtype=float)
  rows.append({"i":i,"raw_mean":arr.mean(axis=(0,1)).round(2).tolist(),"harmonization":fixed.info["harmonization"]})
 print(json.dumps({"variant":label,"seconds":round(time.monotonic()-started,2),"rows":rows}),flush=True)
(a.output/"complete.json").write_text(json.dumps({"done":True}))
