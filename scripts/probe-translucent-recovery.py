"""CPU-only exploratory recovery; never mutate frozen source/output candidates."""
import argparse
import json
from pathlib import Path

from watermark_studio.frozen_candidates import digest, record_digest
from watermark_studio.images import load_photo
from watermark_studio.translucent_recovery import recover_translucent_body

p=argparse.ArgumentParser()
p.add_argument("--snapshot",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--limit",type=int,default=8)
a=p.parse_args()
if a.output.exists() or not 1<=a.limit<=16:
    p.error("Fresh output and1..16 cases required")
summary=json.loads((a.snapshot/"summary.json").read_text())
rows=[json.loads(s) for s in (a.snapshot/"results.jsonl").read_text().splitlines()]
manifest={r["file"]:r for r in json.loads((a.snapshot/"manifest.json").read_text())}
chosen=[r for r in rows if "repair_texture_collapse" in r["quality"].get("guard_codes",[])
        and "source_physical_ambiguity" not in r["quality"].get("guard_codes",[])]
chosen.sort(key=lambda r:r["quality"].get("repair_texture",{}).get("count",0),reverse=True)
a.output.mkdir(parents=True)
reports=[]
for index,row in enumerate(chosen[:a.limit]):
    name=row["file"];source=Path(summary["input_dir"])/name;candidate=Path(summary["output_dir"])/name
    expected=manifest[name]
    assert digest(source)==expected["source_sha256"] and digest(candidate)==expected["candidate_sha256"]
    assert record_digest(row)==expected["record_sha256"]
    before=load_photo(source)[0];result=load_photo(candidate)[0]
    logs=[]
    for item in row.get("protected_core_boxes",[]):
        matching=next((r for r in row["repair_regions"] if r["box"]==item["mask"] and r["kind"]=="panel"),None)
        if matching:
            result,log=recover_translucent_body(before,result,item["core"]);logs.append(log)
    if any(log["applied"] for log in logs):
        result.save(a.output/f"{index:02d}.png")
    assert digest(source)==expected["source_sha256"] and digest(candidate)==expected["candidate_sha256"]
    reports.append({"file":name,"source_sha256":expected["source_sha256"],
                    "candidate_sha256":expected["candidate_sha256"],"recovery":logs})
(a.output/"report.json").write_text(json.dumps({"experimental":True,"not_quality_approved":True,"rows":reports},indent=2))
print(json.dumps({"cases":len(reports),"applied":sum(any(log["applied"] for log in r["recovery"]) for r in reports),
                  "sources_and_candidates_unchanged":True}),flush=True)
