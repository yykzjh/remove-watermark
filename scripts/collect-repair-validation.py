"""Collect bounded, image-free metrics for the November repair investigation."""
import collections
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

root=Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031")
work=root/"review-investigation"
report={"date":"2026-11-01","code_revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "environment":{name:version(name) for name in ("torch","diffusers","transformers","torchao","numpy","pillow")},
        "diagnostic":{},"two_case":[],"cpu_ablation":[]}
for backend in ("resident","independent"):
    path=work/(backend+".json")
    if path.exists():
        report["diagnostic"][backend]=[
            {k:v for k,v in row.items() if k not in {"quality","explanation"}} |
            {"flags":{k:row["quality"].get(k) for k in
                       ("status","watermark_removed","natural","content_preserved","uncertain","reason")}}
            for row in json.loads(path.read_text()) if "quality" in row
        ]
for tag in ("repair-lighting","repair-lighting-v2","repair-lighting-final"):
    for strategy in ("feather","harmonic"):
        path=work/tag/(strategy+".json")
        if not path.exists():
            continue
        item=json.loads(path.read_text())
        report["two_case"].append({k:v for k,v in item.items() if k!="cases"} |
            {"tag":tag,"cases":[{
                "file":r["file"],"status":r["quality"]["status"],"reason":r["quality"]["reason"],
                "boundary":r["quality"].get("repair_boundary"),
                "texture":r["quality"].get("repair_texture"),
                "outside":r["quality"].get("outside_repair_change"),
                "size":r["output_size"],"peak_allocated_mib":r["peak_allocated_mib"],
                "harmonization":r.get("harmonization"),
            } for r in item["cases"]]})
for name in ("010.jpg","057.jpg"):
    for strategy in ("feather","harmonic"):
        path=work/"raw-cache"/f"replay-{strategy}-{name}.json"
        if path.exists():
            report["cpu_ablation"].append(json.loads(path.read_text()))
path=root/"quality-benchmark/subtle-controls-independent-combined-local-guard-repair/results.jsonl"
latest={r["case"]:r for r in map(json.loads,path.read_text().splitlines())}
rows=list(latest.values())
report["controls"]={
    "cases":len(rows),"bad":sum(r["expected"]=="review" for r in rows),
    "good_or_observed_plausible":sum(r["expected"]=="passed" for r in rows),
    "bad_false_pass":[r["case"] for r in rows if r["expected"]=="review" and r["status"]=="passed"],
    "good_false_review":[r["case"] for r in rows if r["expected"]=="passed" and r["status"]!="passed"],
    "errors":sum(r["status"]=="error" for r in rows),
    "review_versions":dict(collections.Counter(r["review_version"] for r in rows)),
    "selected":[{"case":r["case"],"status":r["status"],"reason":r["reason"],
                 "boundary_count":r.get("repair_boundary",{}).get("count"),
                 "texture_count":r.get("repair_texture",{}).get("count")}
                for r in rows if "inside_blur" in r["case"] or r["case"] in {"prior_dark_seam","prior_excavator_observed"}],
}
path=root/"quality-benchmark-lighting-final-64/results.jsonl"
report["performance"]=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
if Path(".run/deployment.json").exists():
    deployment=json.loads(Path(".run/deployment.json").read_text())
    report["deployment"]={k:deployment[k] for k in (
        "gpus","quality_backend","quality_verifier","batch_size","batch_window","detection_batch_size",
        "adaptive_dispatch","quantization","dtype","compile_scope","compile_dynamic","vae_tiling","cpu_threads")}
target=work/"validation-2026-11-01.json"
target.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False))
