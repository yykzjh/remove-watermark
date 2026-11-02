"""Read-only wide-source proof test; native disputes are unlabelled, physical fixtures are veto controls."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--gpu",type=int,required=True)
p.add_argument("--job",required=True)
p.add_argument("--physical-fixtures",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
assert not a.output.exists()
os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch

from watermark_studio.config import DeploymentConfig
from watermark_studio.cross_review import _classify
from watermark_studio.images import load_photo
from watermark_studio.quality_loading import load_quality_encoder, load_quality_processor
from watermark_studio.source_vetting import confirmed_digital, source_carrier_chats

torch.set_num_threads(1)
assert torch.cuda.mem_get_info()[0]>30*2**30
root=Path.home()/".local/state/watermark-studio"/a.job
summary=json.loads((root/"summary.json").read_text())
cases=[]
for row in map(json.loads,(root/"results.jsonl").read_text().splitlines()):
    for v in row.get("source_physical_vetting",[]):
        if v.get("requires_review"):
            cases.append({"source":str(Path(summary["input_dir"])/row["file"]),
                          "box":v["box"],"cohort":"unlabelled_native_dispute"})
for c in json.loads(a.physical_fixtures.read_text()):
    cases.append({"source":c["source"],"box":c["box"],"cohort":"known_physical_veto"})
cfg=DeploymentConfig(quality_verifier=True,quality_backend="independent")
cache=Path.home()/".cache/watermark-models/models"
model=cache/"Qwen--Qwen-Image-2.1-Turbo/snapshots/master"
editor=load_quality_encoder(model/"text_encoder",cfg,"cuda:0")
editor_processor=load_quality_processor(model/"processor")
quality=cache/"Qwen--Qwen3-VL-4B-Instruct/snapshots/master"
verifier=load_quality_encoder(quality,cfg,"cuda:0")
verifier_processor=load_quality_processor(quality)
records=[];started=time.monotonic()
for i,c in enumerate(cases):
    path=Path(c["source"]);before=hashlib.sha256(path.read_bytes()).hexdigest()
    image=load_photo(str(path))[0]
    chats=source_carrier_chats(image,c["box"],wide=True)
    at=time.monotonic()
    first=_classify(chats,editor,editor_processor,"cuda:0")[0]
    second=_classify(chats,verifier,verifier_processor,"cuda:0")[0]
    assert before==hashlib.sha256(path.read_bytes()).hexdigest()
    row=c|{"editor":first,"independent":second,"source_sha256":before,
           "digital_proof":confirmed_digital(first) and confirmed_digital(second),
           "seconds":time.monotonic()-at}
    records.append(row)
    print(json.dumps({"done":i+1,"total":len(cases),"cohort":c["cohort"],"digital_proof":row["digital_proof"]}),flush=True)
bad=sum(c["digital_proof"] for c in records if c["cohort"]=="known_physical_veto")
result={"physical_controls":sum(c["cohort"]=="known_physical_veto" for c in records),
        "physical_misapproved":bad,
        "native_disputes":sum(c["cohort"]=="unlabelled_native_dispute" for c in records),
        "native_confirmed_digital":sum(c["digital_proof"] for c in records if c["cohort"]=="unlabelled_native_dispute"),
        "seconds":time.monotonic()-started,"originals_unchanged":True,"rows":records}
a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!="rows"}),flush=True)
