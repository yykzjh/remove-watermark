"""Compact metadata report; never exports source images or credentials."""
import collections
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

root = Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031")
state_dir = root/"八标照片集/.download-state"
status = json.loads((state_dir/"status.json").read_text())
manifest = json.loads((state_dir/"manifest.json").read_text())
report = {"dataset": {"image_files": status["images"], "valid_decoded": status["valid"],
                      "source_zero_byte_files": sum(r["size"] == 0 and Path(r["path"]).suffix.lower()
                                                   in {".jpg", ".png"} for r in manifest),
                      "image_bytes": status["bytes"], "cloud_files": len(manifest),
                      "excluded_nonimages": len(manifest)-status["images"]},
          "performance": [], "controls": []}
for tag in ("decomposed-64", "independent-8", "stage-batch-64",
            "ablation-serial-64", "ablation-batched-64", "retry-64", "protected-retry-24", "retry-hints-64"):
    f = root/("quality-benchmark-"+tag)/"results.jsonl"
    if f.exists():
        for line in f.read_text().splitlines():
            row = json.loads(line)
            report["performance"].append({k: v for k,v in row.items()
                if k not in {"quality_errors"}})
for backend in ("resident", "independent"):
    f = root/"quality-benchmark"/("subtle-controls-"+backend+"-v2")/"results.jsonl"
    if not f.exists():
        continue
    latest = {(r["case"], r["details"]): r for r in map(json.loads, f.read_text().splitlines())}
    for detail in (0, 4):
        rows = [r for (_, d), r in latest.items() if d == detail]
        report["controls"].append({"backend": backend, "details": detail, "cases": len(rows),
            "expected_bad": sum(r["expected"] == "review" for r in rows),
            "bad_false_pass": [r["case"] for r in rows if r["expected"] == "review" and r["status"] == "passed"],
            "clean_status": dict(collections.Counter(r["status"] for r in rows if r["expected"] == "passed")),
            "errors": sum(r["status"] == "error" for r in rows)})
report["environment"] = {name: version(name) for name in ("torch", "diffusers", "transformers", "torchao")}
report["code_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
report["combined_controls"] = []
for backend in ("resident", "independent"):
    for suffix in ("combined", "combined-details", "combined-local-guard"):
        f = root/"quality-benchmark"/("subtle-controls-"+backend+"-"+suffix)/"results.jsonl"
        if not f.exists():
            continue
        rows = list({r["case"]: r for r in map(json.loads, f.read_text().splitlines())}.values())
        report["combined_controls"].append({
            "backend": backend, "mode": suffix, "cases": len(rows),
            "bad_false_pass": [r["case"] for r in rows if r["expected"] == "review" and r["status"] == "passed"],
            "clean_status": dict(collections.Counter(r["status"] for r in rows if r["expected"] == "passed")),
            "errors": sum(r["status"] == "error" for r in rows),
        })
sources = json.loads((root/"sample-review/sources.json").read_text())
report["sampling_design"] = {"samples": len(sources), "groups": len({r["group"] for r in sources}),
    "towns": len({Path(r["group"]).parts[0] for r in sources}), "seed": 20261031}
profile = root/"sample-review/profile-1024/summary.json"
if profile.exists():
    report["sample_profile"] = json.loads(profile.read_text())
    summary = report["sample_profile"]
    if summary["completed"] == summary["total"] and not summary["worker_failures"]:
        (profile.parent/"status.json").write_text(json.dumps({
            "completed": summary["completed"], "total": summary["total"], "phase": "completed",
        }))
readonly = root/"quality-readonly-8/result.json"
if readonly.exists():
    report["readonly"] = json.loads(readonly.read_text())
resume = root/"quality-readonly-8/resume-result.json"
if resume.exists():
    report["resume"] = json.loads(resume.read_text())
(root/"quality-validation-summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report, ensure_ascii=False))
