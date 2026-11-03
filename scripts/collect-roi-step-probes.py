"""Collect disjoint no-retry sampling shards without selecting between attempts."""
import argparse
import collections
import hashlib
import json
import os
import subprocess
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--work",type=Path,required=True)
a=p.parse_args();work=a.work.resolve(strict=True)
target=work/"uniform12-64";assert not target.exists()
tags=["uniform12-16",*[f"uniform12-16-shard{i}" for i in (1,2,3)]]
reports=[json.loads((work/tag/"report.json").read_text()) for tag in tags]
source_trees={subprocess.check_output(["git","rev-parse",r["revision"]+":src"],text=True).strip() for r in reports}
assert len(source_trees)==1,"Loaded inference/quality code must match in every shard"
settings=[]
for r in reports:
    assert r["images"]==16 and not r["errors"] and not r["bad_sizes"] and r["first_generation_only"]
    cfg={k:v for k,v in r["config"].items() if k!="gpus"}
    settings.append((cfg,r["options"]))
assert all(s==settings[0] for s in settings)
assert settings[0][0]["uniform_steps"]==12 and not settings[0][0]["prefill_overlays"]
assert not settings[0][1]["quality_auto_retry"]
raw={r["file"]:r for r in map(json.loads,(work/"raw/results.valid.jsonl").read_text().splitlines())}
combined={};origins={};hashes={}
for tag,report in zip(tags,reports,strict=True):
    rows=list(map(json.loads,(work/tag/"results.jsonl").read_text().splitlines()))
    assert len(rows)==16 and set(report["source_sha256"])=={r["file"] for r in rows}
    for r in rows:
        name=r["file"];assert name not in combined
        combined[name]=r;origins[name]=tag;hashes[name]=report["source_sha256"][name]
assert set(combined)==set(raw)
assert hashes==json.loads((work/"raw/report.json").read_text())["source_sha256"]
target.mkdir();out=target/"output";out.mkdir()
for name,tag in origins.items():
    src=work/tag/"output"/name;dst=out/name
    assert src.is_file();dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst)
ordered=[combined[name] for name in raw]
(target/"results.valid.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in ordered))
up=[name for name in raw if raw[name]["quality"]["status"]!="passed" and combined[name]["quality"]["status"]=="passed"]
down=[name for name in raw if raw[name]["quality"]["status"]=="passed" and combined[name]["quality"]["status"]!="passed"]
counts=dict(collections.Counter(r["quality"]["status"] for r in ordered))
report={"images":64,"counts":counts,"paired_raw_counts":dict(collections.Counter(r["quality"]["status"] for r in raw.values())),
    "recovered":len(up),"regressed":len(down),"recovered_files":up,"regressed_files":down,
    "same_detected_regions":sum(raw[n]["regions"]==combined[n]["regions"] for n in raw),
    "sum_single_gpu_processing_seconds":sum(r["seconds"] for r in reports),
    "shards":dict(zip(tags,[{"revision":r["revision"],"seconds":r["seconds"],"counts":r["counts"]} for r in reports],strict=True)),
    "source_tree":next(iter(source_trees)),"source_sha256":hashes,
    "selection_sha256":hashlib.sha256(json.dumps(list(raw),ensure_ascii=False).encode()).hexdigest(),
    "no_generation_retries":True,"error":0,"bad_sizes":0,"development_reuse":True,
    "performance_scope":"Sum of disjoint single-GPU probe processing time, not production service throughput"}
(target/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in {"recovered_files","regressed_files","source_sha256","shards"}}))
