"""GPU-only arithmetic/weight-integrity probe; no input/output images leave the server."""
import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--gpu",type=int,required=True)
p.add_argument("--source",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--model",default="~/.cache/watermark-models/models/Qwen--Qwen-Image-2.1-Turbo/snapshots/master")
p.add_argument("--quality-model",default="~/.cache/watermark-models/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master")
a=p.parse_args()
if a.output.exists():
    p.error("Fresh report required")
os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
os.environ["TOKENIZERS_PARALLELISM"]="false"

import torch

from watermark_studio.batching import independent_batch
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
report={"gpu":a.gpu,"source":str(a.source)}
before=hashlib.sha256(a.source.read_bytes()).hexdigest()
cfg=DeploymentConfig(gpus=[a.gpu],quality_verifier=True,quality_backend="independent",quality_phase_offload=True)
started=time.monotonic()
encoder=load_quality_encoder(str(Path(a.quality_model).expanduser()),cfg,"cuda:0")
def tensor_hash(values):
    digest=hashlib.sha256()
    for name,value in values:
        digest.update(name.encode())
        digest.update(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()
initial=tensor_hash(encoder._quality_host_parameters.items())
print(json.dumps({"stage":"verifier_loaded","seconds":round(time.monotonic()-started,3)}),flush=True)
started=time.monotonic();encoder=park_quality_encoder(encoder);torch.cuda.empty_cache()
parked_seconds=time.monotonic()-started
parked_mib=round(torch.cuda.memory_allocated()/2**20)
assert all(p.device.type=="cpu" for p in encoder.parameters())
started=time.monotonic();encoder=restore_quality_encoder(encoder,"cuda:0")
restore_seconds=time.monotonic()-started
restored=tensor_hash(encoder.named_parameters())
assert initial==restored,"Weight mirror changed a verifier parameter"
report["verifier"]={"weights_identical":True,"sha256":initial,"park_seconds":round(parked_seconds,3),
                   "restore_seconds":round(restore_seconds,3),"parked_allocated_mib":parked_mib,
                   "host_pinned_bytes":encoder._quality_host_pinned_bytes}
encoder=park_quality_encoder(encoder)
del encoder;gc.collect();torch.cuda.empty_cache()
print(json.dumps({"stage":"verifier_probe_complete",**report["verifier"]}),flush=True)
cfg=DeploymentConfig(gpus=[a.gpu],quantization="float8",compile=True,compile_dynamic=True,
                     compile_scope="projections",prompt_encoder="lean",kv_cache=True,
                     tf32=True,vae_slicing=True)
started=time.monotonic()
pipe=load_pipeline(cfg,str(Path(a.model).expanduser()))
print(json.dumps({"stage":"editor_loaded","seconds":round(time.monotonic()-started,3)}),flush=True)
source=load_photo(str(a.source))[0]
rows=[]
for label,separate in (("warmup",False),("native",False),("separate",True),("native_repeat",False)):
    pipe._watermark_separate_vae_decode=separate
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    image=independent_batch(pipe,[{"image":source,"prompt":build_full_prompt(),"seed":42}],1024,True)[0]
    torch.cuda.synchronize()
    row={"label":label,"seconds":round(time.monotonic()-started,3),"size":list(image.size),"mode":image.mode,
         "sha256":hashlib.sha256(image.tobytes()).hexdigest(),"peak_allocated_mib":round(torch.cuda.max_memory_allocated()/2**20)}
    rows.append(row);print(json.dumps(row),flush=True)
assert rows[1]["sha256"]==rows[2]["sha256"]==rows[3]["sha256"],"Separate decode changed final pixels"
assert rows[1]["size"]==rows[2]["size"]==rows[3]["size"]
assert rows[1]["mode"]==rows[2]["mode"]==rows[3]["mode"]
report["decode"]={"pixel_identical":True,"rows":rows}
try:
    pipe._watermark_separate_vae_decode=True
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    image=independent_batch(pipe,[{"image":source,"prompt":build_full_prompt(),"seed":42}],1536,True)[0]
    torch.cuda.synchronize()
    report["high_resolution"]={"budget":1536,"ok":True,"size":list(image.size),
        "seconds":round(time.monotonic()-started,3),"peak_allocated_mib":round(torch.cuda.max_memory_allocated()/2**20)}
except torch.cuda.OutOfMemoryError as error:
    report["high_resolution"]={"budget":1536,"ok":False,"error":str(error)[:160]}
assert before==hashlib.sha256(a.source.read_bytes()).hexdigest()
report["source_unchanged"]=True
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"complete":True,"report":str(a.output),"high_resolution":report["high_resolution"]}),flush=True)
