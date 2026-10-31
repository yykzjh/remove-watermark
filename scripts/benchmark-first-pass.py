"""First-generation acceptance, with retries/appeals off and fresh output roots."""
import argparse
import collections
import hashlib
import json
import os
import re
import random
import time
from pathlib import Path

from PIL import Image

from watermark_studio.cli import request

parser=argparse.ArgumentParser()
parser.add_argument("--url",default="http://127.0.0.1:8765")
parser.add_argument("--tag",required=True)
parser.add_argument("--subset",choices=["hard","all","holdout","replay"],default="hard")
parser.add_argument("--images",type=int,default=64)
parser.add_argument("--seed",type=int,default=20261101)
parser.add_argument("--exclude-benchmarks",nargs="*",default=[])
parser.add_argument("--cross-review",action="store_true")
parser.add_argument("--blending",choices=["harmonic","harmonic-fine"],default="harmonic")
parser.add_argument("--profiles",nargs="+",default=["whole-1024","focused-1024","thorough-1024"])
args=parser.parse_args()
if not re.fullmatch(r"[a-zA-Z0-9_-]+",args.tag) or not 1<=args.images<=10000:
    parser.error("Use a simple tag and 1..10000 images")
root=Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031")
base=root/"quality-benchmark-cross-review-v7-64"
work=root/("first-pass-"+args.tag)
inputs=work/"input";inputs.mkdir(parents=True,exist_ok=True)
if args.subset in {"hard","all"}:
    names=(["000.jpg","008.jpg","043.jpg","028.jpg","031.jpg","011.jpg","019.jpg","017.jpg","054.jpg",
            "001.jpg","002.jpg","010.jpg","057.jpg","063.jpg"] if args.subset=="hard" else
           [p.name for p in sorted((base/"input").iterdir())][:args.images])
    paths=[base/"input"/name for name in names]
elif args.subset=="replay":
    old=root/"first-pass-holdout-neutral-v1"
    records=[json.loads(line) for line in (old/"results.jsonl").read_text().splitlines()]
    bad={f["file"] for r in records for f in r["failures"]}
    controls=[p.name for p in sorted((old/"input").iterdir()) if p.name not in bad][:12]
    hard=["000.jpg","008.jpg","043.jpg","028.jpg","031.jpg","011.jpg","019.jpg","017.jpg","054.jpg",
          "001.jpg","002.jpg","010.jpg","057.jpg","063.jpg"]
    names=hard+sorted(bad)+controls
    paths=[base/"input"/name for name in hard]+[old/"input"/name for name in sorted(bad)+controls]
else:
    data=root/"八标照片集"
    manifest=json.loads((data/".download-state/manifest.json").read_text())
    original="/codex-download-1mQf6F6g-20261031/八标照片集/"
    seen=set(json.loads((base/"sources.json").read_text()))
    candidates=[data/r["path"].removeprefix(original) for r in manifest
                if Path(r["path"]).suffix.lower() in {".jpg",".jpeg",".png"}]
    for name in args.exclude_benchmarks:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+",name): parser.error("Invalid excluded benchmark name")
        seen.update(json.loads((root/name/"sources.json").read_text()))
    seen_hashes={hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in seen if Path(p).is_file()}
    random.Random(args.seed).shuffle(candidates)
    paths=[]
    for p in candidates:
        if str(p) in seen or not p.is_file() or not p.stat().st_size:
            continue
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if digest in seen_hashes: continue
        try:
            with Image.open(p) as im: im.load()
        except OSError:
            continue
        paths.append(p)
        seen_hashes.add(digest)
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
        "repair_blending":args.blending,"quality_check":True,"quality_batch_size":2,
        "quality_detail_patches":4,"quality_cross_review":args.cross_review,"overwrite":True}})["id"]
    last_progress=started
    while True:
        state=request(args.url,"/health");status=state["job"]
        assert status["id"]==job
        if status["state"] in {"completed","failed","cancelled"}: break
        if time.monotonic()-last_progress>=30:
            print(json.dumps({"profile":profile,"completed":status["completed"],"total":status["total"],
                "first_counts":status.get("first_pass_counts",{}),"elapsed":round(time.monotonic()-started,1)}),flush=True)
            last_progress=time.monotonic()
        if time.monotonic()-started>10800: raise TimeoutError(job)
        time.sleep(.5)
    rows=[json.loads(line) for line in (Path.home()/".local/state/watermark-studio"/job/"results.jsonl").read_text().splitlines()]
    counts=collections.Counter(r.get("quality",{}).get("status","none") for r in rows)
    record={"profile":profile,"tag":args.tag,"subset":args.subset,"images":len(paths),"job":job,
        "blending":args.blending,"seed":args.seed,"cross_review":args.cross_review,"code_revision":state.get("code_revision","unrecorded"),"seconds":round(time.monotonic()-started,3),"first_counts":dict(counts),
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
    print(json.dumps({k:v for k,v in record.items() if k!="failures"}|
        {"failure_examples":record["failures"][:2]},ensure_ascii=False),flush=True)
    if status["failed"]:raise RuntimeError("Processing failed")
