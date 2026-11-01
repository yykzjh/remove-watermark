"""Source-SHA-proven card recovery diagnostic; original candidates remain immutable."""
from __future__ import annotations

import argparse
import json
import re
import runpy
import subprocess
import time
import uuid
from pathlib import Path

from watermark_studio.config import ProcessingConfig
from watermark_studio.images import load_photo, save_output
from watermark_studio.source_panel_rims import restore_proven_rims
from watermark_studio.translucent_recovery import recover_translucent_body


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--source-job",required=True)
    p.add_argument("--tag",required=True)
    p.add_argument("--native-padding-search",action="store_true")
    a=p.parse_args()
    if not re.fullmatch(r"[a-f0-9]{32}",a.source_job) or not re.fullmatch(r"[a-zA-Z0-9_-]+",a.tag):
        p.error("Explicit source job and new simple tag required")
    state=Path.home()/".local/state/watermark-studio"
    helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    cases=helper["frozen_job_cases"](state,a.source_job)
    if not 1<=len(cases)<=64:
        p.error("Only bounded previously exposed development cases allowed")
    hashes=helper["file_hashes"](cases)
    original=json.loads((state/a.source_job/"summary.json").read_text())
    rows={r["file"]:r for r in map(json.loads,(state/a.source_job/"results.jsonl").read_text().splitlines())}
    options=ProcessingConfig.model_validate(original["options"])
    revision=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    work=Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031")/a.tag
    work.mkdir(exist_ok=False);out=work/"output";out.mkdir()
    reports=[];changed=[];began=time.monotonic()
    for case in cases:
        before,exif,icc,alpha=load_photo(case["source"])
        candidate=load_photo(case["output"])[0]
        result=candidate;logs=[]
        for record in case.get("source_rim_protection",[]):
            if not isinstance(record,dict) or record.get("approved") is not True:
                continue
            _restored,count=restore_proven_rims(before,before,[record])
            if count!=1:
                raise ValueError("Source band/whole-photo SHA proof mismatch")
            if not any(r["kind"]=="panel" and r["box"]==record["mask"] for r in case["regions"]):
                raise ValueError("Proof must belong to the frozen original digital panel")
            result,log=recover_translucent_body(before,result,record["core"],
                _native_padding_search=a.native_padding_search)
            logs.append(log)
        report={"file":case["file"],"recovery":logs,
                "applied":any(log["applied"] for log in logs)}
        reports.append(report)
        if not report["applied"]:
            continue
        if alpha is not None:
            result.putalpha(alpha)
        destination=out/case["file"];destination.parent.mkdir(parents=True,exist_ok=True)
        size=save_output(result,case["source"],destination,exif,icc,options)
        row=rows[case["file"]]
        if size!=row["output_size"]:
            raise ValueError("Native output size changed")
        changed.append(row|{"quality":{},"harmonization":None,
            "harmonization_before_recovery":row.get("harmonization"),
            "source_transmission_recovery":logs,
            "candidate_before_recovery_sha256":hashes[case["output"]],
            "additional_generations":0})
    if hashes!=helper["file_hashes"](cases) or revision!=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip():
        raise RuntimeError("Immutable input/candidate/revision changed")
    job=None
    if changed:
        job=uuid.uuid4().hex;folder=state/job;folder.mkdir(exist_ok=False)
        (folder/"results.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in changed))
        summary=original|{"job":job,"source_job":a.source_job,"total":len(changed),
            "output_dir":str(out),"quality_evaluated":False,"additional_generations":0,
            "code_revision":revision,"state":"completed","scope":"Only physically supported source recovery candidates; not whole-cohort approval"}
        (folder/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    report={"job":job,"examined":len(cases),"applied":len(changed),
        "native_padding_search":a.native_padding_search,"seconds":round(time.monotonic()-began,3),
        "sources_and_candidates_unchanged":True,"quality_evaluated":False,"quality_permission":False,
        "revision":revision,"rows":reports,"file_sha256":hashes}
    (work/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (work/"sources.json").write_text(json.dumps([c["source"] for c in cases],ensure_ascii=False))
    print(json.dumps({k:v for k,v in report.items() if k not in {"rows","file_sha256"}}))


if __name__=="__main__":
    main()
