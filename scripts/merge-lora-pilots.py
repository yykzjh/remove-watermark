"""Merge disjoint screened pilot shards without losing their original-level splits."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from watermark_studio.training_features import digest, validate_pairs


def merge_pilots(pool,target,manifests):
    pool=Path(pool).resolve(strict=True);target=Path(target).resolve()
    if target.parent!=pool.parent or not 2<=len(manifests)<=8:
        raise ValueError("Bounded sibling pilot merge required")
    report=json.loads((pool/"report.json").read_text())
    forbidden_path=pool/"forbidden-sha256.json"
    forbidden=set(json.loads(forbidden_path.read_text()))
    if report.get("state")!="completed" or len(forbidden)!=report["forbidden_sha256"]:
        raise ValueError("Unchanged pre-experiment forbidden snapshot required")
    partitions={}
    all_sources=set()
    for path in report["parts"]:
        path=Path(path).resolve(strict=True)
        if not path.is_relative_to(pool):raise ValueError("Selector escaped reserved pool")
        rows=json.loads(path.read_text())["sources"];hashes={r["sha256"] for r in rows}
        if len(hashes)!=len(rows) or hashes&all_sources or hashes&forbidden:
            raise ValueError("Disjoint previously unexposed source reservations required")
        all_sources.update(hashes);partitions[digest(path)]=hashes
    datasets=[];selected=set();pairs=[]
    for path in manifests:
        path=Path(path).resolve(strict=True);data,_=validate_pairs(path)
        hashes={r["source_original_sha256"] for r in data["pairs"]}
        selector=data.get("candidate_selector_sha256")
        if (path.parent.parent!=pool.parent or not data.get("require_native_pixels")
                or selector not in partitions or not hashes<=partitions[selector]
                or hashes&selected):
            raise ValueError("Independent native-pixel pilot partitions required")
        selected.update(hashes);datasets.append((path,data))
        pairs.extend(data["pairs"])
    if len({r["id"] for r in pairs})!=len(pairs):
        raise ValueError("Duplicate pair IDs")
    if len(pairs)>640:raise ValueError("Bounded merged experiment required")
    for key in ("resolution","font_sha256"):
        if len({data.get(key) for _,data in datasets})!=1:
            raise ValueError("Common target resolution and rendering font required")
    target.mkdir(exist_ok=False)
    (target/"forbidden-sha256.json").write_text(json.dumps(sorted(forbidden)))
    merged=[]
    for row in pairs:
        row=dict(row)
        for key in ("input","target","mask"):
            source=Path(row[key]).resolve(strict=True)
            folder={"input":"pairs","target":"clean","mask":"masks"}[key]
            destination=target/folder/(row[key+"_sha256"]+source.suffix);destination.parent.mkdir(exist_ok=True)
            if not destination.exists():os.link(source,destination)
            if digest(destination)!=row[key+"_sha256"]:
                raise ValueError("Linked target file identity collision")
            row[key]=str(destination)
        merged.append(row)
    train={r["source_original_sha256"] for r in merged if r["split"]=="train"}
    validation={r["source_original_sha256"] for r in merged if r["split"]=="validation"}
    if train&validation or selected&forbidden:
        raise ValueError("Global source leakage")
    data={"version":1,"state":"completed","revision":subprocess.check_output(
        ["git","rev-parse","HEAD"],text=True).strip(),"source_identity":"sha256",
        "resolution":datasets[0][1]["resolution"],"require_native_pixels":True,
        "font_sha256":datasets[0][1].get("font_sha256"),
        "train_originals":len(train),"validation_originals":len(validation),
        "source_overlap":0,"previous_exposure_overlap":0,"pairs":merged,
        "joint_forbidden_snapshot":str(forbidden_path),"joint_forbidden_sha256":digest(forbidden_path),
        "source_manifests":[{"path":str(path),"sha256":digest(path)} for path,_ in datasets],
        "scope":"One pre-screening joint forbidden snapshot; preserve independently assigned source splits"}
    path=target/"manifest.json";path.write_text(json.dumps(data,ensure_ascii=False,indent=2))
    try:validate_pairs(path)
    except Exception:
        data["state"]="failed";path.write_text(json.dumps(data,ensure_ascii=False,indent=2));raise
    summary={"state":"completed","manifest":str(path),"manifest_sha256":digest(path),
        "originals":len(selected),"train_originals":len(train),"validation_originals":len(validation),
        "pairs":len(merged),"source_overlap":0,"previous_exposure_overlap":0}
    (target/"report.json").write_text(json.dumps(summary,indent=2))
    return summary


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--pool",type=Path,required=True)
    p.add_argument("--target",type=Path,required=True)
    p.add_argument("--manifests",type=Path,nargs="+",required=True)
    a=p.parse_args();print(json.dumps(merge_pilots(a.pool,a.target,a.manifests)))


if __name__=="__main__":main()
