"""Read-only probe of candidate grounding protocol before/after structured VQA."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def shape(text):
    value=text.strip()
    if value.startswith("```"):
        value=value.split("\n",1)[1].rsplit("```",1)[0].strip()
    try:
        data=json.loads(value)
    except ValueError:
        return {"shape":"non_json","characters":len(text)}
    known={"watermarks","watermark_removed","natural","uncertain","area","bbox","box","label"}
    return {"shape":type(data).__name__,"known_keys":sorted(set(data)&known) if isinstance(data,dict) else [],
            "watermarks_type":type(data.get("watermarks")).__name__ if isinstance(data,dict) else None,
            "items":len(data) if isinstance(data,(dict,list)) else None}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--gpu",type=int,required=True)
    args=parser.parse_args()
    if args.gpu<0: parser.error("Explicit nonnegative GPU required")
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
    from PIL import Image
    from transformers.utils import logging

    from watermark_studio.config import DeploymentConfig
    from watermark_studio.detection import DETECTION_PROMPT, locate_watermarks_batch
    from watermark_studio.quality import OUTPUT_PROMPT, _structured_vqa, preview
    from watermark_studio.quality_loading import load_quality_encoder, load_quality_processor

    logging.disable_progress_bar()
    root=Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031/adversarial-duel-controls-v1/output")
    files=[root/name for name in ("00-identity.png","00-retained_panel.png")]
    hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    photos=[]
    for p in files:
        with Image.open(p) as raw: photos.append(raw.convert("RGB"))
    model=Path.home()/".cache/watermark-models/models/Qwen--Qwen3.5-4B/snapshots/master"
    cfg=DeploymentConfig(quality_verifier=True,quality_backend="independent")
    encoder=load_quality_encoder(model,cfg,"cuda:0")
    base=load_quality_processor(model)
    audit=[];phase="direct"
    class Processor:
        def __getattr__(self,name): return getattr(base,name)
        def batch_decode(self,*args,**kwargs):
            texts=base.batch_decode(*args,**kwargs)
            audit.extend({"phase":phase,"raw":t,"shape":shape(t)} for t in texts)
            return texts
    processor=Processor()
    started=time.monotonic()
    locate_watermarks_batch(photos,encoder,processor,"cuda:0",1024,2)
    phase="flags"
    chats=[[{"role":"user","content":[{"type":"image","image":preview(p,768)},
                                     {"type":"text","text":OUTPUT_PROMPT}]}] for p in photos]
    _structured_vqa(chats,("watermark_removed","natural","uncertain"),encoder,processor,"cuda:0",128)
    phase="after_flags"
    locate_watermarks_batch(photos,encoder,processor,"cuda:0",1024,2)
    phase="explicit_schema"
    prompt=DETECTION_PROMPT+(
        "\n输出只能是一个JSON对象，唯一必需键watermarks的值必须是数组，不是null或布尔值。"
        "没有数字覆盖层时精确返回{\"watermarks\":[]}。有覆盖层时每项只含kind和box，"
        "kind严格为logo或panel，box是0至1000的4个有序整数。"
        "不能返回最外层数组，不能返回watermark_removed或natural等质量判断字段。")
    locate_watermarks_batch(photos,encoder,processor,"cuda:0",1024,2,False,prompt)
    assert hashes==[hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    target=Path(".run")/("q35-protocol-"+str(time.time_ns())+".json")
    target.write_text(json.dumps(audit,indent=2))
    print(json.dumps({"seconds":round(time.monotonic()-started,3),"files_unchanged":True,
                      "schema_only":[{"phase":r["phase"],**r["shape"]} for r in audit]}),flush=True)

if __name__=="__main__":
    main()
