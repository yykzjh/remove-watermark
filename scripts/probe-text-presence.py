"""CPU-only veto on disputed frozen candidates; it cannot promote any image."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
from pathlib import Path

from watermark_studio.frozen_candidates import digest, record_digest
from watermark_studio.images import load_photo
from watermark_studio.text_presence import TargetTextPresence

p=argparse.ArgumentParser()
p.add_argument("--snapshot",type=Path,required=True)
p.add_argument("--proposed",type=Path,required=True,help="Read-only experimental checker results.jsonl")
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
if a.output.exists():
    p.error("Fresh report required")
summary=json.loads((a.snapshot/"summary.json").read_text())
manifest={r["file"]:r for r in json.loads((a.snapshot/"manifest.json").read_text())}
original={r["file"]:r for r in map(json.loads,(a.snapshot/"results.jsonl").read_text().splitlines())}
rows=list(map(json.loads,a.proposed.read_text().splitlines()))
selected=[r for r in rows if r["quality"].get("text_free_review",{}).get("status")=="corrected"]
if not selected:
    p.error("Explicit proposed experimental corrections required")
engine=TargetTextPresence();reports=[]
for row in selected:
    name=row["file"];source=Path(summary["input_dir"])/name;candidate=Path(summary["output_dir"])/name
    expected=manifest[name]
    assert digest(source)==expected["source_sha256"] and digest(candidate)==expected["candidate_sha256"]
    assert record_digest(original[name])==expected["record_sha256"]
    boxes=row["quality"]["text_free_review"].get("target_boxes") or [[0,0,1,1]]
    report=engine.inspect(load_photo(candidate)[0],boxes)
    reports.append({"file":name,"source_sha256":expected["source_sha256"],
                    "candidate_sha256":expected["candidate_sha256"],"text_presence":report})
    assert digest(source)==expected["source_sha256"] and digest(candidate)==expected["candidate_sha256"]
    print(json.dumps({"completed":len(reports),"total":len(selected),"status":report["status"]}),flush=True)
spec=importlib.util.find_spec("rapidocr_onnxruntime")
model_root=Path(next(iter(spec.submodule_search_locations)))/"models"
models={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in model_root.glob("*.onnx")}
report={"scope":"additional text absence veto only; no quality approvals",
        "code_revision":subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip(),
        "rapidocr_version":importlib.metadata.version("rapidocr-onnxruntime"),
        "onnxruntime_version":importlib.metadata.version("onnxruntime"),"model_sha256":models,
        "proposed":len(reports),"clear":sum(r["text_presence"]["status"]=="clear" for r in reports),
        "text_present":sum(r["text_presence"]["status"]=="text_present" for r in reports),
        "errors":sum(r["text_presence"]["status"]=="error" for r in reports),
        "seconds":round(sum(r["text_presence"]["seconds"] for r in reports),3),
        "sources_outputs_unchanged":True,"rows":reports}
a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in {"rows","model_sha256"}}),flush=True)
