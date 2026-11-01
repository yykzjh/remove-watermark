"""Read-only source geometry proposals; no new approvals or image modifications."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import runpy
import subprocess
import time
from collections import Counter
from pathlib import Path

from watermark_studio.config import Region
from watermark_studio.images import load_photo
from watermark_studio.source_card_edges import propose_card_side_rims


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--source-job",required=True)
    p.add_argument("--tag",required=True)
    a=p.parse_args()
    if not re.fullmatch(r"[a-f0-9]{32}",a.source_job) or not re.fullmatch(r"[a-zA-Z0-9_-]+",a.tag):
        p.error("Exact development source and new simple tag required")
    h=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    state=Path.home()/".local/state/watermark-studio"
    cases=h["frozen_job_cases"](state,a.source_job)
    if not 1<=len(cases)<=64:
        p.error("Bounded previously exposed development source only")
    hashes=h["file_hashes"](cases);began=time.monotonic()
    revision=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    records=[]
    for c in cases:
        source=load_photo(c["source"])[0]
        proposals=(propose_card_side_rims(source,[Region.model_validate(r) for r in c["regions"]],
            c.get("source_rim_protection",[])) if c["scene"]!="document" else [])
        records.append({"file":c["file"],"source":c["source"],"source_file_sha256":hashes[c["source"]],
            "source_rgb_sha256":hashlib.sha256(source.convert("RGB").tobytes()).hexdigest(),
            "regions":c["regions"],"source_cores":c.get("source_cores",[]),
            "proposals":proposals,"approved":False})
    if hashes!=h["file_hashes"](cases) or revision!=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip():
        raise RuntimeError("Original, candidate or code changed during read-only proposal")
    work=Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031")/a.tag
    work.mkdir(exist_ok=False)
    report={"version":1,"source_job":a.source_job,"revision":revision,"images":len(records),
        "proposals":sum(len(r["proposals"]) for r in records),"sides":dict(Counter(
            p["side"] for r in records for p in r["proposals"])),"seconds":round(time.monotonic()-began,3),
        "scope":"CV proposals only; fresh independent VLM/OCR/full-source identity proof required",
        "sources_and_candidates_unchanged":True,"quality_permission":False,"cases":records,"file_sha256":hashes}
    (work/"proposals.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (work/"sources.json").write_text(json.dumps([r["source"] for r in records],ensure_ascii=False))
    print(json.dumps({k:v for k,v in report.items() if k not in {"cases","file_sha256"}},ensure_ascii=False))


if __name__=="__main__":
    main()
