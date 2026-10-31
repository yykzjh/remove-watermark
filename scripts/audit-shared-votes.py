"""Restore pre-adjudication verdicts for an audited identical-weight pair; preserve originals."""
import argparse
import collections
import hashlib
import json
import os
import re
from pathlib import Path

from watermark_studio.evidence import refresh_evidence


def audit_job(folder):
    folder=Path(folder)
    context=json.loads((folder/"weight-identity-audit.json").read_text())
    summary=json.loads((folder/"summary.json").read_text())
    identity=context["identity"]
    if (context["job"]!=summary["id"] or context["observed_revision"]!=summary["code_revision"]
            or identity["relation"]!="same_base_weights"
            or identity["editor_sha256"]!=identity["verifier_sha256"]):
        raise ValueError("A matching complete identical-weight audit is required")
    if summary["state"]!="completed":
        raise ValueError("Completed job required")
    source=folder/"results.jsonl"
    original=source.read_bytes()
    rows=[json.loads(line) for line in original.decode().splitlines()]
    if len(rows)!=summary["total"] or len({r["file"] for r in rows})!=len(rows):
        raise ValueError("Incomplete or repeated job results")
    raw_counts=collections.Counter(r.get("quality",{}).get("status","missing") for r in rows)
    excluded=0
    for row in rows:
        q=row.get("quality",{})
        cross=q.get("cross_review",{})
        if cross.get("status")!="corrected":
            continue
        prior=q.get("pre_cross_review")
        if not isinstance(prior,dict) or prior.get("status") not in {"review","error"}:
            raise ValueError("Cannot reconstruct the original refusal")
        q["invalidated_shared_vote"]=dict(cross)
        q.update(prior)
        q["guard_codes"]=q.get("guard_codes") or []
        q["deterministic_warnings"]=q.get("deterministic_warnings") or []
        q["cross_review"]=cross|{"status":"invalidated_same_base_weights"}
        q["same_weight_vote_invalidated"]=True
        refresh_evidence(q)
        excluded+=1
    counts=collections.Counter(r.get("quality",{}).get("status","missing") for r in rows)
    result={"job":summary["id"],"code_revision":summary["code_revision"],"images":len(rows),
        "raw_counts":dict(raw_counts),"strict_counts":dict(counts),"same_weight_votes_excluded":excluded,
        "identity":identity,"processing_failed":sum(not r.get("ok") for r in rows),
        "quality_errors":counts["error"],"dimension_mismatches":sum(
            r.get("quality",{}).get("dimensions_match") is False for r in rows),
        "generation_retries":sum(max(0,r.get("quality_attempts",1)-1) for r in rows),
        "actions":dict(collections.Counter(r.get("action","unknown") for r in rows)),
        "guards":dict(collections.Counter(g for r in rows for g in (r.get("quality",{}).get("guard_codes") or []))),
        "original_results_sha256":hashlib.sha256(original).hexdigest(),
        "original_log_unchanged":True}
    outputs={"results-without-shared-votes.jsonl":"".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),
             "summary-without-shared-votes.json":json.dumps(result,indent=2)}
    for name,content in outputs.items():
        destination=folder/name
        temp=folder/(name+f".{os.getpid()}.tmp")
        temp.write_text(content)
        temp.replace(destination)
    if source.read_bytes()!=original:
        raise RuntimeError("Original result evidence changed")
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--job",required=True)
    parser.add_argument("--state-dir",type=Path,default=Path.home()/".local/state/watermark-studio")
    args=parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{32}",args.job):
        parser.error("Exact job ID required")
    print(json.dumps(audit_job(args.state_dir/args.job)),flush=True)


if __name__=="__main__":
    main()
