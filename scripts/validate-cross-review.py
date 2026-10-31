"""Real-model controls. Injected initial false alarms are labelled, never hidden."""
import argparse
import json
import os
import time
from copy import deepcopy
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument("--gpu",default="1")
args=parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
os.environ["TOKENIZERS_PARALLELISM"]="false"

import torch
from PIL import ImageDraw
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from watermark_studio.cross_review import reconsider
from watermark_studio.images import load_photo

torch.set_num_threads(1)
base=Path.home()/".cache/watermark-models/models"
editor_path=base/"Qwen--Qwen-Image-2.1-Turbo/snapshots/master"
validator_path=base/"Qwen--Qwen3-VL-4B-Instruct/snapshots/master"
editor=Qwen3VLForConditionalGeneration.from_pretrained(editor_path/"text_encoder",dtype=torch.bfloat16,local_files_only=True).to("cuda").eval()
editor_processor=Qwen3VLProcessor.from_pretrained(editor_path/"processor",local_files_only=True)
validator=Qwen3VLForConditionalGeneration.from_pretrained(validator_path,dtype=torch.bfloat16,local_files_only=True).to("cuda").eval()
processor=Qwen3VLProcessor.from_pretrained(validator_path,local_files_only=True)
root=Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031")
bench=root/"quality-benchmark-protected-v3-64"
old={r["file"]:r for r in map(json.loads,(Path.home()/".local/state/watermark-studio/9dd02bb8463c4f9dba5be6f8137e7214/results.jsonl").read_text().splitlines())}
cases=[]
for label,name,box in (("physical_brand","057.jpg",[.48,.18,.55,.25]),
                       ("physical_sign","010.jpg",[.37,.34,.61,.53])):
    before=load_photo(bench/"input"/name)[0]
    # Use corrected backgrounds; the sign/brand pixels are from the source.
    after=load_photo(root/"quality-benchmark-lighting-final-64/quality-768-b2"/name)[0]
    cases.append((label,before,after,box,old[name]["quality"]["source_regions"],True))
# Retained right-hand anti-fraud code, outside the known left panel.
before=load_photo(bench/"input/054.jpg")[0]
after=load_photo(root/"quality-benchmark-lighting-final-64/quality-768-b2/054.jpg")[0]
cases.append(("retained_tiny_camera_code",before,after,[.84384,.96574,1,1],
              [{"kind":"panel","box":[0,.6172,.4043,1]}],False))
# Known synthetic overlay present in both before/after: equality is NOT legitimacy.
before,after=cases[0][1].copy(),cases[0][2].copy()
box=[.40,.05,.64,.16]
for photo in (before,after):
    coords=tuple(round(v*s) for v,s in zip(box,(photo.width,photo.height,photo.width,photo.height)))
    draw=ImageDraw.Draw(photo);draw.rectangle(coords,fill=(25,100,220))
    draw.text((coords[0]+12,coords[1]+12),"CAMERA SITE 2026",fill="white",font_size=34)
cases.append(("unchanged_synthetic_digital_panel",before,after,box,cases[0][4],False))
# Known original overlay is never exempted even if it is unchanged.
cases.append(("known_original_panel",cases[0][1],cases[0][1].copy(),[.02,.8,.3,.95],cases[0][4],False))
records=[]
for label,before,after,box,source_regions,expected in cases:
    q={"status":"review","watermark_removed":False,"natural":True,"content_preserved":True,
       "uncertain":False,"reason":"controlled injected text false-alarm hypothesis", "seconds":0,
       "residual_regions":[{"kind":"logo","box":box}],"source_regions":source_regions,
       "model_explanation":{"status":"ok","summary":"受控初始争议，随后使用真实模型复核。","findings":[]}}
    start=time.monotonic()
    reconsider(before,after,q,editor,editor_processor,validator,processor,"cuda",editor_context={"test":"injected_initial_verdict"})
    records.append({"case":label,"expected_correction":expected,"injected_initial_verdict":True,
        "final_status":q["status"],"seconds":round(time.monotonic()-start,3),"review":deepcopy(q["cross_review"])})
    print(json.dumps({k:v for k,v in records[-1].items() if k!="review"} |
        {"review_status":q["cross_review"]["status"],"reason":q["cross_review"].get("reason"),
         "error":q["cross_review"].get("error")},ensure_ascii=False),flush=True)
target=root/"review-investigation/cross-review-controls.json"
target.write_text(json.dumps(records,ensure_ascii=False,indent=2))
assert all(r["final_status"]!="passed" for r in records if not r["expected_correction"]),"Digital watermark control falsely passed"
print(json.dumps({"corrected_physical":sum(r["final_status"]=="passed" for r in records if r["expected_correction"]),
                  "physical_cases":sum(r["expected_correction"] for r in records),
                  "bad_false_pass":0,"record":str(target)},ensure_ascii=False),flush=True)
