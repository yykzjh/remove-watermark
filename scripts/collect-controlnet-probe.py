"""Collect disjoint GEN1 shards without generating or reinterpreting quality."""
import argparse
import hashlib
import json
import os
import re
import uuid
from pathlib import Path

from PIL import Image


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--jobs",nargs="+",required=True)
    p.add_argument("--tag",required=True)
    p.add_argument("--work-root",type=Path,default=Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031"))
    a=p.parse_args()
    if (Path(a.tag).name!=a.tag or not a.tag or a.tag in (".","..")
            or any(not re.fullmatch(r"[a-f0-9]{32}",job) for job in a.jobs)):
        p.error("Simple collection tag required")
    state=Path.home()/".local/state/watermark-studio"
    reports=[json.loads((state/job/"summary.json").read_text()) for job in a.jobs]
    first=reports[0]
    signature=("input_dir","source_job","code_revision","options","controlnet","controlnet_settings","server_isolation")
    count=first["probe_shard"]["count"]
    if (len(reports)!=count or {r["probe_shard"]["index"] for r in reports}!=set(range(count))
            or any(r["state"]!="completed" or r["quality_evaluated"] for r in reports)):
        raise ValueError("Exactly one complete generation-only report per shard required")
    for report in reports:
        for key in signature:
            if key=="controlnet":
                # Per-process artifact validation durations differ; weights cannot.
                keys=("sha256","tensors","layers","channels","kv_cache")
                if any(report[key][k]!=first[key][k] for k in keys):
                    raise ValueError("Control identity changed")
            elif report[key]!=first[key]:
                raise ValueError("Incompatible frozen probe shards: "+key)
        if report["probe_shard"]["all_files"]!=first["probe_shard"]["all_files"]:
            raise ValueError("Source selection changed")
    rows=[];hashes={};candidates={}
    for report in reports:
        shard_rows=list(map(json.loads,(state/report["job"]/"results.jsonl").read_text().splitlines()))
        if len(shard_rows)!=report["total"]:
            raise ValueError("Incomplete shard results")
        for row in shard_rows:
            name=Path(row["file"])
            if (name.is_absolute() or ".." in name.parts or str(name) in candidates or not row["ok"]
                    or row["quality_attempts"]!=1 or row["model_inferences"] not in (0,1)):
                raise ValueError("Duplicate, escaped or repeated-generation source")
            source=Path(report["input_dir"])/name
            candidate=Path(report["output_dir"])/name
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            if digest!=report["sources_sha256"][str(name)]:
                raise ValueError("Frozen original changed")
            with Image.open(candidate) as saved:
                if list(saved.size)!=row["output_size"]:
                    raise ValueError("Saved output dimensions changed")
                if not first["options"].get("output_width"):
                    with Image.open(source) as original:
                        if saved.size!=original.size:
                            raise ValueError("Original storage dimensions changed")
            if row["model_inferences"] and (row["controlnet_probe"]["denoising_steps"]!=8
                    or "harmonization" not in row):
                raise ValueError("Incomplete actual GEN1 provenance")
            rows.append(row);hashes[str(name)]=digest;candidates[str(name)]=candidate
    if set(candidates)!=set(first["probe_shard"]["all_files"]):
        raise ValueError("Merged sources are incomplete")
    work=a.work_root/a.tag
    work.mkdir(exist_ok=False);outputs=work/"output";outputs.mkdir()
    for name,source in candidates.items():
        destination=outputs/name;destination.parent.mkdir(exist_ok=True,parents=True)
        os.link(source,destination)
    job=uuid.uuid4().hex;folder=state/job;folder.mkdir(exist_ok=False)
    summary={k:first[k] for k in signature}
    summary.update(job=job,state="completed",input_dir=first["input_dir"],output_dir=str(outputs),
        scope="Collected disjoint development GEN1; no generation or quality approval in collection",
        total=len(rows),sources_sha256=hashes,quality_evaluated=False,
        generation_wall_seconds_max=max(r["seconds"] for r in reports),
        source_shard_jobs=a.jobs)
    (folder/"results.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in sorted(rows,key=lambda r:r["file"])))
    (folder/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (work/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (work/"sources.json").write_text(json.dumps([str(Path(first["input_dir"])/r["file"]) for r in rows],ensure_ascii=False))
    print(json.dumps({"job":job,"total":len(rows),"generation_wall_seconds_max":summary["generation_wall_seconds_max"]}),flush=True)


if __name__=="__main__":
    main()
