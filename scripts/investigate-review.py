"""Diagnose the two unresolved cases without changing acceptance or output images."""
import argparse
import json
import os
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="1")
parser.add_argument("--backend", choices=["resident", "independent"], default="resident")
parser.add_argument("--explain", action="store_true")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from watermark_studio import quality
from watermark_studio.config import Region
from watermark_studio.images import load_photo

torch.set_num_threads(1)
base = Path.home()/".cache/watermark-models/models"
folder = base/("Qwen--Qwen-Image-2.1-Turbo" if args.backend=="resident" else "Qwen--Qwen3-VL-4B-Instruct")/"snapshots/master"
encoder = Qwen3VLForConditionalGeneration.from_pretrained(
    folder/"text_encoder" if args.backend=="resident" else folder,
    dtype=torch.bfloat16, local_files_only=True,
).to("cuda").eval()
processor = Qwen3VLProcessor.from_pretrained(
    folder/"processor" if args.backend=="resident" else folder, local_files_only=True,
)
root = Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031")
bench = root/"quality-benchmark-protected-v3-64"
state = Path.home()/".local/state/watermark-studio/9dd02bb8463c4f9dba5be6f8137e7214/results.jsonl"
old = {r["file"]:r for r in map(json.loads,state.read_text().splitlines())}
out = root/"review-investigation"
out.mkdir(exist_ok=True)
original_vqa = quality._structured_vqa
original_apply = processor.apply_chat_template
phase, traces = "", []
def trace_apply(*a, **kw):
    inputs = original_apply(*a, **kw)
    if phase:
        traces.append({"phase":phase,"input_tokens":int(inputs.input_ids.shape[1]),
                       "vision_patches":int(inputs["image_grid_thw"].prod(-1).sum())})
    return inputs
processor.apply_chat_template = trace_apply
active_pair, mode = None, "canvas"
def inspect(chats, fields, *arguments):
    global phase
    if mode=="separate" and fields[0]=="watermark_removed":
        before, after = active_pair
        _, boxes = quality.detail_sheets(before,after,4)
        content = [{"type":"text","text":"图1是完整结果照片，后续图像是该照片的局部裁切，均不是新增面板。"},
                   {"type":"image","image":quality.preview(after,768)}]
        for index, box in enumerate(boxes):
            crop=after.crop(tuple(round(v*s) for v,s in zip(box,(after.width,after.height,after.width,after.height))))
            crop.thumbnail((256,256),Image.Resampling.LANCZOS)
            content.extend([{"type":"text","text":f"局部{index+1}在完整照片中的归一化范围：{box}。裁切边缘不是水印。"},
                            {"type":"image","image":crop}])
        prompt=quality.OUTPUT_PROMPT.replace(
            "这是一张去水印结果的检验画布：上方为整图，下方（如有）为同一结果图的放大细节。",
            "图1是去水印结果整图，后续图是标明原照片坐标的放大局部。结合整图的透视、物体和遮挡关系判断真实印刷文字。")
        content.append({"type":"text","text":prompt})
        chats=[[{"role":"user","content":content}]]
    phase=fields[0]
    try:
        return original_vqa(chats,fields,*arguments)
    finally:
        phase=""
quality._structured_vqa=inspect
records=[]
for name in ("010.jpg","057.jpg"):
    before=load_photo(bench/"input"/name)[0]
    after=load_photo(bench/"quality-768-b2"/name)[0]
    active_pair=(before,after)
    for mode in ("overview","canvas","separate"):
        traces.clear()
        started=time.monotonic()
        verdict=quality.verify_batch([(before,after)],encoder,processor,"cuda",768,128,
            reviewer=args.backend,detail_patches=0 if mode=="overview" else 4,
            source_regions=[[Region.model_validate(r) for r in old[name]["quality"]["source_regions"]]])[0]
        records.append({"file":name,"mode":mode,"backend":args.backend,"seconds":round(time.monotonic()-started,3),
                        "traces":list(traces),"quality":verdict})
    if args.explain:
        crop=(0,.35,.7,1) if name=="010.jpg" else (0,.25,.55,1)
        views=[]
        for image in (before,after):
            box=tuple(round(v*s) for v,s in zip(crop,(image.width,image.height,image.width,image.height)))
            views.append(quality.preview(image.crop(box),1024))
        prompt="""图1是原照片的同一局部，含数字水印。图2是去水印结果的同一局部。
请只根据可见证据，分析图2是否残留数字水印、是否改变真实场景文字、修补是否自然。
区分浮在场景上的数字文字、真实警示牌/机械品牌/贴纸、修补中新造的不可核实文字。
不要把图1水印当图2残留。先列出图1、图2各自实际可见的文字及附着的物体，
再判断是否有残留或新造标识、边线/阴影/材质异常；看不清时写uncertain，不能猜测。
返回JSON：original_visible_text, output_visible_text, digital_residue_evidence,
new_or_unverifiable_details, naturalness_issues, uncertain。不要因为收到“待复核”就假设存在水印。"""
        chats=[[{"role":"user","content":[*({"type":"image","image":v} for v in views),{"type":"text","text":prompt}]}]]
        inputs=original_apply(chats,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt")
        inputs.pop("token_type_ids",None)
        inputs=inputs.to("cuda")
        with torch.inference_mode():
            generated=encoder.generate(**inputs,do_sample=False,use_cache=True,max_new_tokens=640)
        explanation=processor.batch_decode(generated[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
        records.append({"file":name,"backend":args.backend,"explanation":explanation})
path=out/(args.backend+".json")
path.write_text(json.dumps(records,ensure_ascii=False,indent=2))
print(json.dumps([{k:v for k,v in r.items() if k!="quality"} |
                  ({"flags":{k:r["quality"].get(k) for k in ("status","watermark_removed","natural","content_preserved","uncertain","reason")}} if "quality" in r else {})
                 for r in records],ensure_ascii=False),flush=True)
