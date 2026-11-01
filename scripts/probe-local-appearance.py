"""Read-only localized repair-naturalness screen; never changes QA verdicts."""
import argparse
import json
import os
import runpy
import time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--gpu",type=int,required=True)
p.add_argument("--profile",choices=["4b","8b","35-4b"],required=True)
p.add_argument("--controls",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--all-backgrounds",action="store_true")
a=p.parse_args()
if a.output.exists() or a.gpu<0:p.error("Fresh report and explicit GPU required")
os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch

from watermark_studio.config import DeploymentConfig
from watermark_studio.images import context_box, load_photo, pixel_box
from watermark_studio.quality import _structured_vqa, preview
from watermark_studio.quality_loading import load_quality_encoder, load_quality_processor

torch.set_num_threads(1)
assert torch.cuda.mem_get_info()[0]>25*2**30
helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
names={"06-exact_removal.png","06-flat_repair.png","06-repair_blur.png",
       "02-exact_removal.png","02-shadow_32.png","07-residual_background.png","08-exact_removal.png"}
kinds={"exact_removal","exact_jpeg","repair_blur","flat_repair","shadow_18","shadow_32","residual_background"}
cases=[c for c in helper["known_control_cases"](a.controls) if c["kind"] in kinds and
       (a.all_backgrounds or c["file"] in names)]
before=helper["file_hashes"](cases)
root=Path.home()/".cache/watermark-models/models"
editor=root/"Qwen--Qwen-Image-2.1-Turbo/snapshots/master"
model=(editor/"text_encoder" if a.profile=="8b" else
       root/("Qwen--Qwen3.5-4B" if a.profile=="35-4b" else "Qwen--Qwen3-VL-4B-Instruct")/"snapshots/master")
at=time.monotonic()
encoder=load_quality_encoder(model,DeploymentConfig(quality_verifier=True,quality_backend="independent"),"cuda:0")
if a.profile=="8b":
    from transformers import Qwen3VLProcessor
    processor=Qwen3VLProcessor.from_pretrained(editor/"processor",local_files_only=True)
else:
    processor=load_quality_processor(model)
load_seconds=time.monotonic()-at
prompt=(
    "图1为原始照片局部，其中有数字叠加信息卡。图2是同位置处理结果。只验收图2修补自然度。"
    "只检查新出现的问题；原图未被覆盖部分原本的焦外、噪声、真实阴影不算修补缺陷。"
    "原水印内与上下左右未遮挡邻域的材质、清晰度、光照、工程线条和人体是否自然连续？"
    "灰色纯色填块、低清晰度矩形模糊、同轮廓暗亮板、材质替换、重复颗粒和断裂构件均须natural=false。"
    "不能用没有水印文字代替自然度判断，也不能以旁边实体标牌正常来豁免目标里的异常。"
    "原图的信息卡不是图2残留；不要求未知遮挡内部逐像素还原历史。局部裁剪边缘不是照片接缝。"
    "细节不足时uncertain=true。只返回JSON，natural和uncertain为布尔值，"
    "area为当前局部主要缺陷位置none/top_left/top/top_right/left/center/right/bottom_left/bottom/bottom_right。")
records=[]
for case in cases:
    source=load_photo(case["source"])[0];target=load_photo(case["output"])[0]
    core=max(case["source_cores"],key=lambda r:(r["core"][2]-r["core"][0])*(r["core"][3]-r["core"][1]))["core"]
    box=context_box(pixel_box(core,source.size),source.size,.15)
    chats=[[{"role":"user","content":[{"type":"text","text":"图1原始局部："},
        {"type":"image","image":preview(source.crop(box),768)},
        {"type":"text","text":"图2结果局部："},{"type":"image","image":preview(target.crop(box),768)},
        {"type":"text","text":prompt}]}]]
    at=time.monotonic()
    values,_=_structured_vqa(chats,("natural","uncertain"),encoder,processor,"cuda:0",128)
    records.append({"file":case["file"],"kind":case["kind"],"expected_natural":case["kind"] in {"exact_removal","exact_jpeg"},
                    "checks":values[0],"seconds":round(time.monotonic()-at,3)})
assert before==helper["file_hashes"](cases)
report={"scope":"localized naturalness only, not full QA acceptance","profile":a.profile,
        "load_seconds":round(load_seconds,3),"cases":len(records),"unchanged":True,
        "clean_refused":[r["file"] for r in records if r["expected_natural"] and
                         (not r["checks"]["natural"] or r["checks"]["uncertain"])],
        "defects_missed":[r["file"] for r in records if not r["expected_natural"] and
                          r["checks"]["natural"] and not r["checks"]["uncertain"]],"rows":records}
a.output.write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!="rows"}),flush=True)
