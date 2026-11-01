"""Paired acceptance/output audit, never a substitute for ground-truth labels."""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--work",type=Path,required=True)
p.add_argument("--baseline",required=True)
p.add_argument("--candidate",required=True)
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
if a.output.exists():
    p.error("Fresh report required")
records=[json.loads(x) for x in (a.work/"results.jsonl").read_text().splitlines()]
records={r["profile"]:r for r in records}
left,right=records[a.baseline],records[a.candidate]
assert left["source_selection_sha256"]==right["source_selection_sha256"]
assert left["source_inputs_unchanged"] and right["source_inputs_unchanged"]
assert left["code_revision"]==right["code_revision"],"Different loaded runtimes confound the paired recipe"
def rows(record):
    assert re.fullmatch("[a-f0-9]{32}",record["job"])
    root=Path.home()/".local/state/watermark-studio"/record["job"]
    result=[json.loads(x) for x in (root/"results.jsonl").read_text().splitlines()]
    assert len(result)==record["images"]==len({r["file"] for r in result})
    return {r["file"]:r for r in result}
before,after=rows(left),rows(right)
assert set(before)==set(after)
transitions=collections.Counter()
details=[]
for name in sorted(before):
    old,new=before[name],after[name]
    x,y=old.get("quality",{}).get("status"),new.get("quality",{}).get("status")
    transitions[str(x)+"->"+str(y)]+=1
    def source_mask(row):
        return (row.get("quality",{}).get("source_regions") if row.get("processing_path")=="whole_direct"
                else row.get("repair_regions"))
    first_mask,second_mask=source_mask(old),source_mask(new)
    same_masks=isinstance(first_mask,list) and isinstance(second_mask,list) and first_mask==second_mask
    paths=[(a.work/profile/name).resolve() for profile in (a.baseline,a.candidate)]
    assert all(path.is_relative_to(a.work.resolve()) for path in paths)
    hashes=[hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None for path in paths]
    details.append({"file":name,"before":x,"after":y,
                    "model_before":old.get("quality",{}).get("model_status"),
                    "model_after":new.get("quality",{}).get("model_status"),"same_source_masks":same_masks,
                    "same_output":hashes[0] is not None and hashes[0]==hashes[1]})
summary={"scope":"acceptance transitions, not ground-truth accuracy",
         "runtime":left["code_revision"],"images":len(before),
         "baseline":a.baseline,"candidate":a.candidate,
         "seconds_before":left["seconds"],"seconds_after":right["seconds"],
         "sources_unchanged":True,"transitions":dict(transitions),
         "gained":[r["file"] for r in details if r["before"]!="passed" and r["after"]=="passed"],
         "lost":[r["file"] for r in details if r["before"]=="passed" and r["after"]!="passed"],
         "same_source_masks":sum(r["same_source_masks"] for r in details),
         "model_gained":[r["file"] for r in details if r["model_before"]!="passed" and r["model_after"]=="passed"],
         "model_lost":[r["file"] for r in details if r["model_before"]=="passed" and r["model_after"]!="passed"],
         "same_output_verdict_changed":[r["file"] for r in details if r["same_output"] and r["before"]!=r["after"]]}
a.output.write_text(json.dumps(summary|{"pairs":details},indent=2))
print(json.dumps(summary),flush=True)
