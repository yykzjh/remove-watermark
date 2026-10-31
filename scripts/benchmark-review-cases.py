"""Matched two-case ablation, using original files, seeds and strict quality checks."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from PIL import Image, ImageOps

from watermark_studio.cli import request

parser=argparse.ArgumentParser()
parser.add_argument("--url",default="http://127.0.0.1:18770")
parser.add_argument("--tag",default="repair-lighting")
parser.add_argument("--strategies",nargs="+",choices=["feather","harmonic"],default=["feather","harmonic"])
parser.add_argument("--names",nargs="+",default=["010.jpg","057.jpg"])
parser.add_argument("--quality-auto-retry",action="store_true")
args=parser.parse_args()
root=Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031")
base=root/"quality-benchmark-protected-v3-64"
work=root/"review-investigation"/args.tag
inputs=work/"input"
inputs.mkdir(parents=True,exist_ok=True)
for name in args.names:
    if not (inputs/name).exists():
        os.link(base/"input"/name,inputs/name)
def hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
before=hashes()
deadline=time.monotonic()+600
while request(args.url,"/health")["stage"]!="ready":
    if time.monotonic()>deadline:
        raise TimeoutError("Model loading")
    time.sleep(2)
for strategy in args.strategies:
    start=time.monotonic()
    job=request(args.url,"/jobs",{"input_dir":str(inputs),"output_dir":str(work/strategy),
        "options":{"concurrency":8,"resolution":1024,"preserve_background":True,"repair_blending":strategy,
                   "quality_check":True,"quality_auto_retry":args.quality_auto_retry,"quality_batch_size":2,"quality_detail_patches":4,"overwrite":True}})["id"]
    while True:
        state=request(args.url,"/health")
        status=state["job"]
        if status["state"] in {"completed","failed","cancelled"}:
            break
        if time.monotonic()-start>600:
            raise TimeoutError(job)
        time.sleep(.25)
    rows=[json.loads(line) for line in (Path.home()/".local/state/watermark-studio"/job/"results.jsonl").read_text().splitlines()]
    record={"strategy":strategy,"seconds":round(time.monotonic()-start,3),"job":job,
            "state":status["state"],"source_unchanged":hashes()==before,
            "quality_counts":status["quality_counts"],"failed":status["failed"],
            "cases":[{"file":r["file"],"quality":r.get("quality"),"output_size":r.get("output_size"),
                      "harmonization":r.get("harmonization"),
                      "peak_allocated_mib":r.get("peak_allocated_mib"),"oom_retries":r.get("oom_retries")} for r in rows]}
    (work/(strategy+".json")).write_text(json.dumps(record,ensure_ascii=False,indent=2))
    for row in rows:
        if not row["ok"]:
            continue
        with Image.open(work/strategy/row["file"]) as im:
            photo=ImageOps.exif_transpose(im).convert("RGB")
        photo.thumbnail((1280,1280))
        path=Path(".run")/("new-"+strategy+"-"+row["file"])
        while True:
            for quality in (80,70,60,50,40,30):
                photo.save(path,quality=quality,optimize=True)
                if path.stat().st_size<=98304:
                    break
            if path.stat().st_size<=98304:
                break
            photo.thumbnail((round(photo.width*.8),round(photo.height*.8)))
    print(json.dumps({k:v for k,v in record.items() if k!="cases"} | {
        "cases":[{"file":r["file"],"status":r["quality"]["status"],"reason":r["quality"]["reason"],
                  "boundary":r["quality"].get("repair_boundary"),
                  "outside_tiles":r["quality"].get("outside_repair_change",{}).get("tiles"),
                  "size":r["output_size"],"harmonization":r.get("harmonization")} for r in record["cases"]]},ensure_ascii=False),flush=True)
    assert status["failed"]==0 and record["source_unchanged"]
