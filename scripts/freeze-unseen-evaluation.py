"""Freeze new source-only evaluation inputs, excluding audited history and image aliases."""
import argparse
import collections
import hashlib
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from watermark_studio.source_fingerprints import FingerprintIndex, fingerprint_file

p=argparse.ArgumentParser(__doc__)
p.add_argument("--root",type=Path,required=True)
p.add_argument("--history-audit",type=Path,required=True)
p.add_argument("--extra-history",type=Path,required=True)
p.add_argument("--target",type=Path,required=True)
p.add_argument("--images",type=int,default=1000)
p.add_argument("--seed",type=int,default=2026110301)
p.add_argument("--additional-history",type=Path,action="append",default=[])
a=p.parse_args()
root=a.root.resolve(strict=True);data=(root/"八标照片集").resolve(strict=True)
audit=a.history_audit.resolve(strict=True);target=a.target.resolve()
assert audit.parent==root and target.parent==root and not target.exists()
assert 1000<=a.images<=4096 and a.extra_history.stat().st_size<=100000
report=json.loads((audit/"report.json").read_text())
assert report["state"]=="completed" and report["qualified"]
assert not any(report[k] for k in ("fingerprint_errors","missing_history_paths","unresolved_catalog_paths","known_history_gaps"))
extra=json.loads(a.extra_history.read_text());assert isinstance(extra,list) and 1<=len(extra)<=32
index=FingerprintIndex();seen=set()
for line in (audit/"fingerprints.jsonl").open():
    row=json.loads(line)
    if row["encoded_sha256"] not in seen:
        index.add(row);seen.add(row["encoded_sha256"])
for row in extra:
    if row["encoded_sha256"] not in seen:index.add(row);seen.add(row["encoded_sha256"])
additional={}
for path in a.additional_history:
    assert 0<path.stat().st_size<=50_000_000
    raw=path.read_bytes();additional[str(path.resolve())]=hashlib.sha256(raw).hexdigest()
    for line in raw.decode().splitlines():
        row=json.loads(line)
        if row["encoded_sha256"] not in seen:
            index.add(row);seen.add(row["encoded_sha256"])
manifest_path=data/".download-state/manifest.json"
manifest_raw=manifest_path.read_bytes();manifest=json.loads(manifest_raw)
prefix="/codex-download-1mQf6F6g-20261031/八标照片集/"
candidates=[]
for row in manifest:
    assert row["path"].startswith(prefix)
    rel=Path(row["path"].removeprefix(prefix))
    assert not rel.is_absolute() and ".." not in rel.parts
    if rel.suffix.lower() in {".jpg",".jpeg",".png"}:candidates.append(data/rel)
assert len(candidates)==len(set(candidates))
random.Random(a.seed).shuffle(candidates)
target.mkdir();inputs=target/"input";inputs.mkdir()
selected=[];skips=collections.Counter();counts=collections.Counter()
def inspect(path):
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(data):
        raise ValueError("Unavailable or unbounded source")
    return fingerprint_file(path)
with ThreadPoolExecutor(max_workers=2) as pool:
    for start in range(0,len(candidates),64):
        batch=candidates[start:start+64]
        futures=[pool.submit(inspect,path) for path in batch]
        for path,future in zip(batch,futures,strict=True):
            if len(selected)==a.images:break
            try:fp=future.result()
            except (OSError,ValueError):
                skips["invalid_source"]+=1;continue
            if index.matches(fp):
                skips["history_or_selected_alias"]+=1;continue
            rel=path.relative_to(data);dst=inputs/rel
            dst.parent.mkdir(parents=True,exist_ok=True);os.link(path,dst)
            selected.append(fp);index.add(fp);counts[rel.parts[0]]+=1
            if len(selected)%100==0:print(json.dumps({"selected":len(selected),"skips":dict(skips)}),flush=True)
        if len(selected)==a.images:break
assert len(selected)==a.images
selection={"sources":[row["path"] for row in selected],
    "source_sha256":{Path(row["path"]).relative_to(data).as_posix():row["encoded_sha256"] for row in selected},
    "seed":a.seed,"images":len(selected),"town_counts":dict(counts),"skips":dict(skips),
    "policy":"First fixed-seed shuffled source candidates passing source availability and prior/intra-set alias exclusion; no output or quality labels",
    "manifest_sha256":hashlib.sha256(manifest_raw).hexdigest(),
    "history_report_sha256":hashlib.sha256((audit/"report.json").read_bytes()).hexdigest(),
    "extra_history_sha256":hashlib.sha256(a.extra_history.read_bytes()).hexdigest(),
    "history_unique_encoded":len(seen),"additional_history_sha256":additional,"model_calls":0,"previews_created":0,"images_changed":0}
raw=json.dumps(selection,ensure_ascii=False,indent=2)
(target/"source-selection.json").write_text(raw)
(target/"fingerprints.jsonl").write_text("".join(json.dumps(fp,ensure_ascii=False)+"\n" for fp in selected))
print(json.dumps({"state":"frozen","images":len(selected),"selection_sha256":hashlib.sha256(raw.encode()).hexdigest(),"towns":len(counts),"skips":dict(skips)}),flush=True)
