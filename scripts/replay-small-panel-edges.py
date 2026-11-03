"""Read-only source geometry audit; no image generation or quality promotion."""
import argparse
import hashlib
import json
from pathlib import Path

from watermark_studio.config import Region
from watermark_studio.detection import complete_edge_strokes
from watermark_studio.images import load_photo

p=argparse.ArgumentParser(__doc__);p.add_argument("--source-job",required=True)
p.add_argument("--target",type=Path,required=True);a=p.parse_args()
assert not a.target.exists()
state=Path.home()/".local/state/watermark-studio"/a.source_job
summary=json.loads((state/"summary.json").read_text());assert summary["state"]=="completed"
root=Path(summary["input_dir"]).resolve();rows=list(map(json.loads,(state/"results.jsonl").read_text().splitlines()))
assert len(rows)==summary["total"] and len({r["file"] for r in rows})==len(rows)
changes=[];candidates=0
for row in rows:
    regions=[Region.model_validate(r) for r in row.get("regions",[])]
    if not any(r.kind=="panel" and (r.box[2]-r.box[0])*(r.box[3]-r.box[1])<.035 for r in regions):
        continue
    source=(root/row["file"]).resolve();assert source.is_relative_to(root)
    with source.open("rb") as f:sha=hashlib.file_digest(f,"sha256").hexdigest()
    image=load_photo(source)[0]
    control,_=complete_edge_strokes(image,regions)
    changed,proof=complete_edge_strokes(image,regions,include_small_panels=True)
    candidates+=1
    with source.open("rb") as f:assert hashlib.file_digest(f,"sha256").hexdigest()==sha
    if control!=changed:
        changes.append({"file":row["file"],"source_sha256":sha,
            "control_regions":[r.model_dump() for r in control],
            "proposed_regions":[r.model_dump() for r in changed],"source_evidence":proof,
            "baseline_status":row.get("quality",{}).get("status")})
a.target.mkdir(parents=True)
report={"scope":"Source-only geometry audit; no new image generation or quality permission",
    "source_job":a.source_job,"images":len(rows),"small_panel_images":candidates,"changed":len(changes),
    "changed_review_images":sum(r["baseline_status"]!="passed" for r in changes),"rows":changes}
(a.target/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
(a.target/"selection.json").write_text(json.dumps([r["file"] for r in changes],ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!="rows"}),flush=True)
