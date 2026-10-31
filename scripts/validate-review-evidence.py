"""Validate real rejected/pass evidence and a complete original-based retry cycle."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from watermark_studio.cli import request

parser=argparse.ArgumentParser()
parser.add_argument("--url",default="http://127.0.0.1:8765")
args=parser.parse_args()
root=Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031")
base=root/"quality-benchmark-protected-v3-64"
work=root/"review-investigation"/f"evidence-{time.time_ns()}"
inputs=work/"input"
outputs=work/"output"
inputs.mkdir(parents=True);outputs.mkdir()
for name in ("010.jpg","054.jpg","057.jpg"):
    os.link(base/"input"/name,inputs/name)
    previous=(base/"quality-768-b2"/name if name!="054.jpg" else
              root/"quality-benchmark-lighting-final-64/quality-768-b2"/name)
    os.link(previous,outputs/name)
original_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}

def run(retry):
    start=time.monotonic()
    job=request(args.url,"/jobs",{"input_dir":str(inputs),"output_dir":str(outputs),
        "options":{"concurrency":3,"preserve_background":True,"quality_check":True,
                   "quality_batch_size":2,"quality_auto_retry":retry,"overwrite":retry}})["id"]
    while True:
        status=request(args.url,"/health")["job"]
        assert status["id"]==job
        if status["state"] in {"completed","failed","cancelled"}:
            break
        if time.monotonic()-start>600:
            raise TimeoutError(job)
        time.sleep(.25)
    assert status["state"]=="completed" and not status["failed"],status
    rows=[json.loads(line) for line in
          (Path.home()/".local/state/watermark-studio"/job/"results.jsonl").read_text().splitlines()]
    result={"job":job,"retry":retry,"seconds":round(time.monotonic()-start,3),
            "quality_counts":status["quality_counts"],"cases":[]}
    for row in rows:
        q=row["quality"]
        report=q["model_explanation"]
        if not retry and q["status"]=="review":
            assert report["status"]=="ok",(row["file"],report)
            assert report.get("summary"),row["file"]
            assert q["retry_feedback"] and q["repair_advice"],row["file"]
        assert q["judgment_basis"]["model_checks"],row["file"]
        used=[a for a in row.get("attempt_history",[]) if a.get("repair_feedback_used")]
        if retry and row.get("quality_attempts",1)>1:
            assert used,row["file"]
        result["cases"].append({"file":row["file"],"status":q["status"],
            "model_explanation":report,"evidence":q["evidence"],"retry_feedback":q["retry_feedback"],
            "explanation_seconds":q.get("explanation_seconds",0),
            "quality_attempts":row.get("quality_attempts",1),"feedback_attempts":len(used),
            "feedback_codes_used":[f["issue"] for a in used for f in a["repair_feedback_used"]],
            "evidence_version":q["review_version"]})
    if not retry:
        by_name={r["file"]:r for r in rows}
        assert by_name["010.jpg"]["quality"]["status"]=="review"
        assert by_name["057.jpg"]["quality"]["status"]=="passed"
    (work/("retry.json" if retry else "inspect.json")).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!="cases"} |
        {"cases":[{k:r[k] for k in ("file","status","quality_attempts","feedback_attempts","feedback_codes_used")}
                  | {"explanation_status":r["model_explanation"]["status"],"evidence_count":len(r["evidence"])}
                 for r in result["cases"]]},ensure_ascii=False),flush=True)
    return result

records=[run(False),run(True)]
assert original_hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
Path(".run/evidence-validation-location.txt").write_text(str(work))
print("Evidence validation complete; input hashes unchanged",flush=True)
