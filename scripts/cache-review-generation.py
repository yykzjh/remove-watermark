"""Cache two real model generations on the data disk, then replay CPU blending."""
import argparse
import json
import os
import time
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument("--gpu",default="2")
parser.add_argument("--replay",action="store_true")
args=parser.parse_args()
root=Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031")
bench=root/"quality-benchmark-protected-v3-64"
work=root/"review-investigation/raw-cache"
work.mkdir(parents=True,exist_ok=True)
if not args.replay:
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    os.environ["WATERMARK_REPAIR_CAPTURE"]=str(work)
    from watermark_studio.config import DeploymentConfig, ProcessingConfig
    cfg=DeploymentConfig.model_validate(json.loads(Path(".run/deployment.json").read_text()))
    if cfg.expandable_segments:
        os.environ["PYTORCH_ALLOC_CONF"]="expandable_segments:True"
    from watermark_studio.images import load_photo
    from watermark_studio.worker import load_pipeline, process_image
    pipe=load_pipeline(cfg,cfg.model)
    for name in ("010.jpg","057.jpg"):
        source=bench/"input"/name
        started=time.monotonic()
        result=process_image({"source":str(source),"destination":str(work/("generated-"+name)),
            "relative":name},ProcessingConfig(scope="full_image",preserve_background=True,
            repair_blending="harmonic",overwrite=True),cfg,pipe)
        width,height=load_photo(source)[0].size
        record={"file":name,"capture":str(work/f"{os.getpid()}-{width}x{height}"),
                "seconds":round(time.monotonic()-started,3),"result":result}
        (work/("case-"+name+".json")).write_text(json.dumps(record,ensure_ascii=False,indent=2))
        print(json.dumps({k:v for k,v in record.items() if k!="result"} |
            {"ok":result["ok"],"harmonization":result.get("harmonization")}),flush=True)
else:
    from PIL import Image

    from watermark_studio.config import ProcessingConfig, Region
    from watermark_studio.images import composite_protected, load_photo, save_output
    from watermark_studio.quality import outside_repair_changes, repair_boundary_warnings
    for name in ("010.jpg","057.jpg"):
        item=json.loads((work/("case-"+name+".json")).read_text())
        folder=Path(item["capture"])
        original=Image.open(folder/"original.png").convert("RGB")
        generated=Image.open(folder/"generated.png").convert("RGB")
        regions=[Region.model_validate(r) for r in json.loads((folder/"regions.json").read_text())]
        _,exif,icc,_=load_photo(bench/"input"/name)
        for mode in ("feather","harmonic"):
            started=time.monotonic()
            image=composite_protected(original,generated,regions,12,mode)
            path=work/("replay-"+mode+"-"+name)
            save_output(image,bench/"input"/name,path,exif,icc,ProcessingConfig(overwrite=True))
            encoded=load_photo(path)[0]
            record={"file":name,"strategy":mode,"seconds":round(time.monotonic()-started,3),
                "boundary":repair_boundary_warnings(encoded,regions,before=original),
                "outside":outside_repair_changes(original,encoded,regions),
                "harmonization":image.info.get("harmonization")}
            (work/("replay-"+mode+"-"+name+".json")).write_text(json.dumps(record,indent=2))
            print(json.dumps(record),flush=True)
            encoded.thumbnail((1280,1280))
            preview=Path(".run")/("replay-"+mode+"-"+name)
            while True:
                for quality in (80,70,60,50,40,30):
                    encoded.save(preview,quality=quality,optimize=True)
                    if preview.stat().st_size<=98304:
                        break
                if preview.stat().st_size<=98304:
                    break
                encoded.thumbnail((round(encoded.width*.8),round(encoded.height*.8)))
