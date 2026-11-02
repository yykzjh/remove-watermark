"""Partition disjoint original candidates before parallel independent clean screening."""
from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

from watermark_studio.evaluation import exclusion_sources
from watermark_studio.trial_sources import unique_sources


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--target",type=Path,required=True)
    p.add_argument("--shards",type=int,default=3)
    p.add_argument("--per-shard",type=int,default=240)
    p.add_argument("--seed",type=int,default=2026110206)
    a=p.parse_args()
    root=a.root.resolve(strict=True);data=root/"八标照片集"
    if not 1<=a.shards<=8 or not 40<=a.per_shard<=512 or a.target.resolve().parent!=root:
        p.error("Bounded independent original pool under data root required")
    a.target.mkdir(exist_ok=False)
    exposed,_=exclusion_sources(root,root/"quality-benchmark-cross-review-v7-64",[])
    exposed={str(Path(v).resolve()) for v in exposed};forbidden=set()
    for file in root.glob("*/forbidden-sha256.json"):forbidden.update(json.loads(file.read_text()))
    for file in root.glob("*/source-selection.json"):
        forbidden.update(v for v in json.loads(file.read_text()).get("source_sha256",{}).values() if isinstance(v,str) and len(v)==64)
    cloud=json.loads((data/".download-state/manifest.json").read_text())
    prefix="/codex-download-1mQf6F6g-20261031/八标照片集/";groups=defaultdict(list)
    for record in cloud:
        if not record["path"].startswith(prefix):continue
        path=(data/record["path"].removeprefix(prefix)).resolve()
        if (path.is_relative_to(data.resolve()) and path.is_file() and str(path) not in exposed and
                path.suffix.lower() in {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"}):
            groups[path.relative_to(data).parts[0]].append(path)
    rng=random.Random(a.seed)
    for group in groups.values():rng.shuffle(group)
    ordered=(path for group in itertools.zip_longest(*[groups[k] for k in sorted(groups)]) for path in group if path)
    rows,hashed=unique_sources(ordered,forbidden,a.shards*a.per_shard);rng.shuffle(rows)
    files=[]
    for shard in range(a.shards):
        partition=rows[shard*a.per_shard:(shard+1)*a.per_shard]
        path=a.target/f"part-{shard}.json"
        path.write_text(json.dumps({"version":1,"sources":partition},ensure_ascii=False,indent=2))
        files.append(str(path.resolve()))
    report={"state":"completed","version":1,"shards":a.shards,"per_shard":a.per_shard,
        "unique_sha256":len({row["sha256"] for row in rows}),"paths_hashed":hashed,
        "forbidden_sha256":len(forbidden),"parts":files,
        "scope":"Hash/filename reservation only; actual screened sources immediately enter each preparation exposure ledger"}
    (a.target/"report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=="__main__":
    main()
