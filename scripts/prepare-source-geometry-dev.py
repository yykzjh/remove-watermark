"""Prepare explicitly reused source-geometry development inputs on the data disk."""
import argparse
import json
import os
import shutil
from pathlib import Path

from watermark_studio.config import Region
from watermark_studio.frozen_candidates import bounded, digest, record_digest
from watermark_studio.source_geometry import broad_masks

p=argparse.ArgumentParser()
p.add_argument("--snapshot",type=Path,required=True)
p.add_argument("--work",type=Path,required=True)
a=p.parse_args()
if a.work.exists():
    p.error("Fresh development directory required")
summary=json.loads((a.snapshot/"summary.json").read_text())
manifest={r["file"]:r for r in json.loads((a.snapshot/"manifest.json").read_text())}
rows=list(map(json.loads,(a.snapshot/"results.jsonl").read_text().splitlines()))
selected=[r for r in rows if r.get("source_scene")=="outdoor" and
          broad_masks(([Region.model_validate(v) for v in r.get("repair_regions",[])],""))]
if not selected:
    p.error("No source-only broad-mask cases")
a.work.mkdir(parents=True)
inputs=a.work/"input";inputs.mkdir();hashes={}
for row in selected:
    name=row["file"];source=bounded(Path(summary["input_dir"]),name);item=manifest[name]
    assert digest(source)==item["source_sha256"] and record_digest(row)==item["record_sha256"]
    target=bounded(inputs,name);target.parent.mkdir(parents=True,exist_ok=True)
    try:os.link(source,target)
    except OSError:shutil.copyfile(source,target)
    assert digest(target)==item["source_sha256"]
    hashes[name]=item["source_sha256"]
selection={"scope":"reused-source geometry development; never a fresh holdout",
           "development_reuse":True,"source_job":summary["origin_job"],"source_snapshot":str(a.snapshot),
           "images":len(hashes),"unique_sha256":len(set(hashes.values())),
           "prior_sha256_overlap":len(hashes),"source_sha256":hashes}
(a.work/"source-selection.json").write_text(json.dumps(selection,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in selection.items() if k!="source_sha256"}))
