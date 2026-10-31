"""First-generation acceptance, with retries/appeals off and fresh output roots."""
import argparse
import collections
import json
import os
import random
import time
from pathlib import Path

from PIL import Image

from watermark_studio.cli import request

parser=argparse.ArgumentParser()
parser.add_argument("--url",default="http://127.0.0.1:8765")
parser.add_argument("--tag",required=True)
parser.add_argument("--subset",choices=["hard","all","holdout"],default="hard")
parser.add_argument("--images",type=int,default=64)
parser.add_argument("--profiles",nargs="+",default=["whole-1024","focused-1024","thorough-1024"])
args=parser.parse_args()
root=Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031")
base=root/"quality-benchmark-cross-review-v7-64"
work=root/("first-pass-"+args.tag)
inputs=work/"input";inputs.mkdir(parents=True,exist_ok=True)
if args.subset in {"hard","all"}:
    names=(["000.jpg","008.jpg","043.jpg","028.jpg","031.jpg","011.jpg","019.jpg","017.jpg","054.jpg",
            "001.jpg","002.jpg","010.jpg","057.jpg","063.jpg"] if args.subset=="hard" else
           [p.name for p in sorted((base/"input").iterdir())][:args.images])
    paths=[base/"input"/name for name in names]
else:
    data=root/"八标照片集"
    manifest=json.loads((data/".download-state/manifest.json").read_text())
    original="/codex-download-1mQf6F6g-20261031/八标照片集/"
    seen=set(json.loads((base/"sources.json").read_text()))
    candidates=[data/r["path"].removeprefix(original) for r in manifest
                if Path(r["path"]).suffix.lower() in {".jpg",".jpeg",".png"}]
    random.Random(20261101).shuffle(candidates)
    paths=[]
    for p in candidates:
        if str(p) in seen or not p.is_file() or not p.stat().st_size:
            continue
        try:
            with Image.open(p) as im: im.load()
        except OSError:
            continue
        paths.append(p)
        if len(paths)==args.images: break
    names=[f"{i:04d}"+p.suffix for i,p in enumerate(paths)]
assert len(paths)==len(names) and paths
for p,name in zip(paths,names):
    if not (inputs/name).exists(): os.link(p,inputs/name)
(work/"sources.json").write_text(json.dumps([str(p) for p in paths],ensure_ascii=False))
while request(args.url,"/health")["stage"]!="ready": time.sleep(2)
for profile in args.profiles:
    mode,budget=profile.rsplit("-",1);budget=int(budget)
    target=work/profile
    if target.exists(): raise RuntimeError("Fresh tag/output required; existing accepted outputs must not bias first-pass metrics")
    started=time.monotonic()
    job=request(args.url,"/jobs",{"input_dir":str(inputs),"output_dir":str(target),"options":{
        "concurrency":20,"resolution":budget,"preserve_background":True,"first_pass_mode":mode,
        "repair_blending":"harmonic","quality_check":True,"quality_batch_size":2,
        "quality_detail_patches":4,"overwrite":True}})["id"]
    while True:
        state=request(args.url,"/health");status=state["job"]
        assert status["id"]==job
        if status["state"] in {"completed","failed","cancelled"}: break
        if time.monotonic()-started>3600: raise TimeoutError(job)
        time.sleep(.5)
    rows=[json.loads(line) for line in (Path.home()/".local/state/watermark-studio"/job/"results.jsonl").read_text().splitlines()]
    counts=collections.Counter(r.get("quality",{}).get("status","none") for r in rows)
    record={"profile":profile,"tag":args.tag,"subset":args.subset,"images":len(paths),"job":job,
        "seconds":round(time.monotonic()-started,3),"first_counts":dict(counts),
        "first_pass_rate":round(counts["passed"]/len(paths),5),"processing_failed":status["failed"],
        "model_inferences":sum(r.get("model_inferences",0) for r in rows),
        "input_budgets":dict(collections.Counter(str(n) for r in rows for n in r.get("model_input_budgets",[]))),
        "source_detail_views":sum(r.get("source_detail_views",0) for r in rows),
        "bad_sizes":sum(Image.open(inputs/r["file"]).size!=Image.open(target/r["file"]).size for r in rows if r["ok"]),
        "quality_errors":sum(r.get("quality",{}).get("status")=="error" for r in rows),
        "peak_allocated_mib":max(r.get("peak_allocated_mib",0) for r in rows),
        "failures":[{"file":r["file"],"status":r.get("quality",{}).get("status"),
                     "reason":r.get("quality",{}).get("reason"),"guards":r.get("quality",{}).get("guard_codes",[])}
                    for r in rows if r.get("quality",{}).get("status")!="passed"]}
    with (work/"results.jsonl").open("a") as f:f.write(json.dumps(record,ensure_ascii=False)+"\n")
    print(json.dumps(record,ensure_ascii=False),flush=True)
    if status["failed"]:raise RuntimeError("Processing failed")
