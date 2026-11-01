"""Small read-only screen of unconstrained verifier JSON; not a QA approval pipeline."""
import argparse
import json
import os
import time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--gpu",type=int,required=True)
p.add_argument("--profile",choices=["4b","35-4b"],required=True)
p.add_argument("--thinking",action="store_true")
p.add_argument("--max-tokens",type=int,default=1024)
p.add_argument("--controls",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
if a.output.exists() or not 128<=a.max_tokens<=2048 or a.gpu<0 or (a.thinking and a.profile!="35-4b"):
    p.error("Fresh report, valid explicit GPU/token bound, and native thinking architecture required")
os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch

from watermark_studio.config import DeploymentConfig, Region
from watermark_studio.images import load_photo
from watermark_studio.quality import OUTPUT_PROMPT, _verdict_json, output_detail_views, preview
from watermark_studio.quality_loading import load_quality_encoder, load_quality_processor

torch.set_num_threads(1)
assert torch.cuda.mem_get_info()[0]>25*2**30
import runpy

helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
names=["00-identity.png","00-retained_panel.png","02-exact_removal.png","02-shadow_32.png",
       "06-exact_removal.png","06-repair_blur.png","06-flat_repair.png",
       "07-residual_background.png","08-exact_removal.png","08-small_code.png"]
cases=[c for c in helper["known_control_cases"](a.controls) if c["file"] in names]
assert len(cases)==len(names)
before=helper["file_hashes"](cases)
model=Path.home()/".cache/watermark-models/models"/("Qwen--Qwen3-VL-4B-Instruct" if a.profile=="4b" else "Qwen--Qwen3.5-4B")/"snapshots/master"
began=time.monotonic()
encoder=load_quality_encoder(model,DeploymentConfig(quality_verifier=True,quality_backend="independent"),"cuda:0")
processor=load_quality_processor(model)
if a.thinking:
    processor=processor._processor
load_seconds=time.monotonic()-began
rows=[]
for case in cases:
    source=load_photo(case["source"])[0];output=load_photo(case["output"])[0]
    regions=[Region.model_validate(r) for r in case["regions"]]
    content=[{"type":"text","text":"所有图片仅来自同一张结果照片，没有原图。图1是完整结果，其余为原生局部。"},
             {"type":"image","image":preview(output,1024)}]
    details,boxes=output_detail_views(source,output,4,regions)
    for detail,box in zip(details,boxes):
        content.extend([{"type":"text","text":f"结果局部完整照片坐标{box}，裁切边缘不是修补缺陷。"},
                        {"type":"image","image":detail}])
    content.append({"type":"text","text":OUTPUT_PROMPT.replace(
        "这是一张去水印结果的检验画布：上方为整图，下方（如有）为同一结果图的放大细节。",
        "图1是完整结果，后续图像是结果局部。")+"\n同时增加evidence字段，用不超过30字描述结果中直接可见的依据。"})
    at=time.monotonic()
    inputs=processor.apply_chat_template([{"role":"user","content":content}],tokenize=True,
        add_generation_prompt=True,return_dict=True,return_tensors="pt")
    inputs.pop("token_type_ids",None);inputs=inputs.to("cuda:0")
    with torch.inference_mode():
        ids=encoder.generate(**inputs,do_sample=False,use_cache=True,max_new_tokens=a.max_tokens)
    text=processor.batch_decode(ids[:,inputs.input_ids.shape[1]:],skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0]
    # Never store hidden reasoning. A truncated thinking answer cannot approve.
    final=text.rsplit("</think>",1)[-1] if "</think>" in text else text
    record={"file":case["file"],"expected":case["expected"],"seconds":round(time.monotonic()-at,3),
            "tokens":ids.shape[1]-inputs.input_ids.shape[1]}
    try:
        if "<think>" in final:
            raise ValueError("Reasoning exceeded token budget")
        flags=_verdict_json(final,("watermark_removed","natural","uncertain"))
        if not isinstance(flags.get("evidence"),str) or not flags["evidence"].strip():
            raise ValueError("Missing visible observation")
        record["checks"]=flags
    except (ValueError,TypeError) as error:
        record["error"]=str(error)[:160]
    rows.append(record)
    print(json.dumps({k:v for k,v in record.items() if k not in {"checks","file"}}),flush=True)
assert before==helper["file_hashes"](cases)
report={"scope":"output-only screening, no source-content approval or final QA",
        "profile":a.profile,"thinking":a.thinking,"max_tokens":a.max_tokens,
        "load_seconds":round(load_seconds,3),"sources_outputs_unchanged":True,
        "files_sha256":before,
        "rows":rows}
a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({"complete":True,"cases":len(rows),"report":str(a.output)}),flush=True)
