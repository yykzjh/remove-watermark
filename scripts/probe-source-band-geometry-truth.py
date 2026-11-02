"""Exact-pixel truth test for proposed source bands; no pixel restoration."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np

from watermark_studio.config import Region
from watermark_studio.images import load_photo
from watermark_studio.source_card_edges import propose_card_side_rims
from watermark_studio.source_panel_rims import full_source_binding


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--fixture",type=Path,required=True)
    p.add_argument("--top-proofs",type=Path,nargs="+",required=True)
    p.add_argument("--target",type=Path,required=True)
    a=p.parse_args();data=json.loads(a.fixture.read_text())
    if data.get("version")!=1 or data.get("quality_permission") is not False:
        p.error("Diagnostic exact-source fixture required")
    fixture_root=a.fixture.resolve().parent
    hashes=dict(data["known_clean_source_sha256"])
    clean={Path(path).name.split("-")[0]:load_photo(path)[0] for path in hashes}
    for path,sha in hashes.items():
        if digest(path)!=sha:
            raise ValueError("Known clean source bytes changed")
    top={};identity=None
    for path in a.top_proofs:
        if path.stat().st_size>16*2**20:
            raise ValueError("Bounded independent identity reports required")
        report=json.loads(path.read_text());current=report["weight_identity"]
        if (report.get("mode")!="top" or report.get("sources_unchanged") is not True or
                report.get("false_approvals")!=0 or current["relation"]!="different_weights" or
                current["editor_sha256"]==current["verifier_sha256"]):
            raise ValueError("Fresh independent source identity required")
        if identity is not None and identity!=current:
            raise ValueError("Checkpoint identity differs across source reports")
        identity=current
        for row in report["rows"]:
            proof=row.get("proof",{})
            if proof.get("approved") is True:
                key=(str(Path(row["source"]).resolve()),tuple(proof["mask"]))
                if key in top:
                    raise ValueError("Duplicate identity proof")
                top[key]=proof
    unique={}
    for row in data["cases"]:
        source=Path(row["source"]).resolve(strict=True)
        if not source.is_relative_to(fixture_root) or digest(source)!=row["source_file_sha256"]:
            raise ValueError("Fixture source changed or escaped")
        hashes[str(source)]=row["source_file_sha256"]
        unique.setdefault(str(source),row)
    if len(unique)!=data["images"]:
        raise ValueError("Unique source coverage mismatch")
    revision=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    a.target.mkdir(exist_ok=False);began=time.monotonic();rows=[];counts={"header":Counter(),"body":Counter()}
    for source,row in unique.items():
        image=load_photo(source)[0];original=clean[Path(source).name.split("-")[0]]
        if image.size!=original.size:
            raise ValueError("Exact clean-image counterpart required")
        prop=row["proposal"];proof=top.get((source,tuple(prop["mask"])))
        record={"file":Path(source).name,"approved":False,"identity":proof is not None,"variants":{}}
        if proof is not None:
            if not full_source_binding(image,proof):
                raise ValueError("Fresh identity does not bind whole fixture RGB")
            for label,body in (("header",False),("body",True)):
                proposals=propose_card_side_rims(image,[Region(kind="panel",box=prop["mask"])],
                    [proof],verify_body_sides=body)
                values=[]
                for proposal in proposals:
                    box=tuple(proposal["restore_pixels"])
                    old=np.asarray(original.crop(box));overlay=np.asarray(image.crop(box))
                    changed=int(np.any(old!=overlay,axis=2).sum())
                    values.append({"side":proposal["side"],"pixels":list(box),
                        "changed_pixels":changed,"total_pixels":old.shape[0]*old.shape[1],"approved":False})
                    counts[label]["proposals"]+=1
                    counts[label]["contaminated"]+=bool(changed)
                    counts[label]["outside_only"]+=not bool(changed)
                counts[label]["photos_with_proposals"]+=bool(values)
                record["variants"][label]=values
        rows.append(record)
    if (any(digest(path)!=sha for path,sha in hashes.items()) or
            revision!=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()):
        raise ValueError("Source or code changed during geometry truth probe")
    result={"scope":"Exact source/known-clean pixel comparison only; no approvals or image writes",
        "revision":revision,"images":len(rows),"source_identities":sum(r["identity"] for r in rows),
        "results":{k:dict(v) for k,v in counts.items()},"seconds":round(time.monotonic()-began,3),
        "sources_unchanged":True,"rows":rows}
    (a.target/"report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!="rows"}))


if __name__=="__main__":
    main()
