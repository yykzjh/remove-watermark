"""Compare source-only camera-footer prompts on one frozen photograph; no edits."""
import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--job",required=True)
    parser.add_argument("--file",required=True)
    parser.add_argument("--gpu",type=int,required=True)
    parser.add_argument("--state-dir",type=Path,default=Path.home()/".local/state/watermark-studio")
    parser.add_argument("--models-root",type=Path,default=Path.home()/".cache/watermark-models/models")
    args=parser.parse_args()
    if args.gpu<0 or not re.fullmatch(r"[a-f0-9]{32}",args.job):
        parser.error("Explicit GPU and exact job ID required")
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
    from transformers import Qwen3VLProcessor
    from transformers.utils import logging

    from watermark_studio.config import DeploymentConfig
    from watermark_studio.detection import (
        COMPLETENESS_PROMPT,
        DETECTION_PROMPT,
        locate_watermarks_batch,
    )
    from watermark_studio.images import load_photo
    from watermark_studio.quality_loading import load_quality_encoder
    logging.disable_progress_bar()
    folder=args.state_dir/args.job
    summary=json.loads((folder/"summary.json").read_text())
    input_root=Path(summary["input_dir"]).resolve()
    source=(input_root/args.file).resolve()
    if not source.is_relative_to(input_root):
        raise ValueError("Image must stay inside the frozen source root")
    rows=[json.loads(line) for line in (folder/"results.jsonl").read_text().splitlines()]
    row=next(r for r in rows if r["file"]==args.file)
    logo=next(r for r in row["quality"]["source_regions"] if r["kind"]=="logo")
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    image=load_photo(str(source))[0]
    x0,y0,x1,y1=logo["box"]
    dx,dy=max(.04,(x1-x0)*.5),max(.035,y1-y0)
    crop=(round(max(0,x0-dx)*image.width),round(max(0,y0-dy)*image.height),
          round(min(1,x1+dx)*image.width),round(min(1,y1+dy)*image.height))
    model=args.models_root/"Qwen--Qwen-Image-2.1-Turbo/snapshots/master"
    cfg=DeploymentConfig(quality_verifier=True,quality_backend="independent")
    encoder=load_quality_encoder(model/"text_encoder",cfg,"cuda:0")
    processor=Qwen3VLProcessor.from_pretrained(model/"processor",local_files_only=True)
    named=DETECTION_PROMPT+(
        "\nCamera groups can include 今日水印、相机、真实时间 and a final line beginning 防伪 or 防伪码 "
        "followed by tiny letters and digits. Include the ENTIRE connected visible cluster, "
        "especially its lowest code line even when touching the photograph edge. Large brand text "
        "is still a digital overlay; do not ignore it because this is an enlarged crop. "
        "Do not invent text or confuse real physical printing with digital overlays.")
    whole_line=named+(
        "\nThe final 防伪 line may extend LEFT and RIGHT beyond the brand. "
        "Include the entire lowest line from its FIRST to LAST visible symbol, "
        "not just symbols directly below the larger logo. Use a SEPARATE tight box "
        "for the complete code line if necessary. Inspect the very bottom edge.")
    results=[]
    for name,prompt in (("dedicated",COMPLETENESS_PROMPT),("general",DETECTION_PROMPT),("named_footer",named),("whole_footer_line",whole_line)):
        started=time.monotonic()
        result=locate_watermarks_batch([image.crop(crop)],encoder,processor,"cuda:0",1280,1,False,prompt)[0]
        record={"mode":name,"seconds":round(time.monotonic()-started,3)}
        if isinstance(result,Exception):
            record["error"]=str(result)[:240]
        else:
            record["raw"]=result[1]
            record["regions"]=[]
            for region in result[0]:
                a,b,c,d=region.box
                mapped=[(crop[0]+a*(crop[2]-crop[0]))/image.width,
                        (crop[1]+b*(crop[3]-crop[1]))/image.height,
                        (crop[0]+c*(crop[2]-crop[0]))/image.width,
                        (crop[1]+d*(crop[3]-crop[1]))/image.height]
                record["regions"].append({"kind":region.kind,"box":mapped,
                    "area":round((mapped[2]-mapped[0])*(mapped[3]-mapped[1]),6)})
        results.append(record)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    record={"file":args.file,"image_size":list(image.size),"crop":crop,"source_unchanged":True,"results":results}
    target=folder/("source-prompt-probe-"+str(time.time_ns())+".json")
    target.write_text(json.dumps(record,indent=2))
    print(json.dumps(record|{"results":[{k:v for k,v in r.items() if k!="raw"} for r in results]}),flush=True)


if __name__=="__main__":
    main()
