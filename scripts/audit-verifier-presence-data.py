"""Audit exact synthetic label visibility; never filter generated-photo QA outcomes."""
from __future__ import annotations

import argparse
import collections
import copy
import json
from pathlib import Path

from watermark_studio.images import load_photo
from watermark_studio.quality_presence import measurable_overlay
from watermark_studio.training_features import digest


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    assert not a.output.exists()
    data=json.loads(a.manifest.read_text())
    assert data["state"]=="completed" and data["task"]=="output_only_digital_overlay_presence"
    clean_rows={row["source_original_sha256"]:row for row in data["rows"]
                if not row["overlay_present"] and not row["uncertain"]}
    rejected=set();reports=[];current=None;clean=None
    for row in data["rows"]:
        if not row["overlay_present"]:
            continue
        parent=row["source_original_sha256"]
        if parent!=current:
            original=clean_rows[parent]
            assert digest(original["after"])==original["after_sha256"]
            clean=load_photo(original["after"])[0];current=parent
        assert digest(row["after"])==row["after_sha256"]
        measured=measurable_overlay(clean,load_photo(row["after"])[0])
        reports.append({"id":row["id"],**measured})
        if not measured["usable"]:
            prefix=row["id"].removesuffix("-marked")
            rejected.update([prefix+"-marked",prefix+"-clean"])
    updated=copy.deepcopy(data)
    updated["rows"]=[row for row in data["rows"] if row["id"] not in rejected]
    updated["row_counts"]=dict(collections.Counter(row["split"] for row in updated["rows"]))
    updated["visibility_audit"]={"parent_manifest":str(a.manifest),"parent_manifest_sha256":digest(a.manifest),
        "rejected_synthetic_rows":sorted(rejected),"marked_cases":len(reports),
        "scope":"Pre-training synthetic label audit; no generated-photo outcomes or native quality thresholds"}
    report=a.output.with_suffix(".visibility-report.json")
    report.write_text(json.dumps({"cases":reports,**updated["visibility_audit"]},indent=2))
    a.output.write_text(json.dumps(updated,ensure_ascii=False,indent=2))
    print(json.dumps({"manifest":str(a.output),"marked_cases":len(reports),
                      "rejected_synthetic_rows":len(rejected),"row_counts":updated["row_counts"]}),flush=True)


if __name__=="__main__":
    main()
