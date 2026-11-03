"""Read-only CPU candidate audit; unchanged printing is NOT itself a quality approval."""
import argparse
import hashlib
import json
from pathlib import Path

from watermark_studio.images import load_photo
from watermark_studio.physical_text_review import eligible
from watermark_studio.text_preservation import OutputTextPreservation

p=argparse.ArgumentParser(__doc__)
p.add_argument("--proposed",type=Path,required=True)
p.add_argument("--baseline-job",required=True)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--known-bad",type=Path)
a=p.parse_args()
assert not a.output.exists()
state=Path.home()/".local/state/watermark-studio"/a.baseline_job
original={r["file"]:r for r in map(json.loads,(state/"results.jsonl").read_text().splitlines())}
proposals=list(map(json.loads,a.proposed.read_text().splitlines()))
engine=OutputTextPreservation();reports=[]
def digest(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
for row in proposals:
    prior=original[row["file"]]
    context={"background_preserved":prior.get("background_preserved"),
             "source_scene":prior.get("source_scene")}
    if row["quality"]["status"]!="passed" or not eligible(prior.get("quality",{}),context):
        continue
    source,candidate=row["source"],row["output"]
    before_hash,after_hash=digest(source),digest(candidate)
    proof=engine.inspect(load_photo(source)[0],load_photo(candidate)[0])
    assert digest(source)==before_hash and digest(candidate)==after_hash
    reports.append({"file":row["file"],"source_sha256":before_hash,"candidate_sha256":after_hash,
        "proof":proof,"cohort":"native_candidate"})
if a.known_bad:
    manifest=json.loads(a.known_bad.read_text())
    for row in manifest["cases"]:
        source,candidate=row["source"],row["output"]
        reports.append({"file":row["file"],"source_sha256":digest(source),"candidate_sha256":digest(candidate),
            "proof":engine.inspect(load_photo(source)[0],load_photo(candidate)[0]),"cohort":"known_defect"})
report={"scope":"candidate text-preservation audit only; NOT final approvals","rows":reports,
    "proposed_rows_read":len(proposals),"candidate_witnesses":sum(r["cohort"]=="native_candidate" and r["proof"]["status"]=="unchanged" for r in reports),
    "candidate_attempts":sum(r["cohort"]=="native_candidate" for r in reports),
    "seconds":sum(r["proof"]["seconds"] for r in reports),"pixels_unchanged":True}
a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!="rows"}),flush=True)
