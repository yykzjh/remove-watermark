"""Compare GEN1 synthetic controls to exact pre-overlay RGB; no QA overrides."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from watermark_studio.frozen_candidates import digest
from watermark_studio.images import load_photo


def known_body_metrics(truth, planted, output, geometry):
    if truth.size != planted.size or truth.size != output.size:
        raise ValueError("Native grids must match")
    l, _, r, b = geometry["panel"]
    top = geometry["header_bottom"]
    box = (l,top,r,b)
    clean = np.asarray(truth.crop(box),dtype=np.float32)
    marked = np.asarray(planted.crop(box),dtype=np.float32)
    candidate = np.asarray(output.crop(box),dtype=np.float32)
    predicted = clean*geometry["transmission"]+255*(1-geometry["transmission"])
    visible = np.max(np.abs(marked-predicted),axis=2) <= 1.01
    # The separately drawn footer is opaque even if it overlaps the body.
    fl,ft,fr,fb = geometry["footer"]
    visible[max(0,ft-top):max(0,fb-top+1),max(0,fl-l):max(0,fr-l+1)] = False
    if visible.sum() < 64:
        raise ValueError("No measurable transmitted background")
    error = np.abs(candidate-clean)
    x, y = candidate.mean(axis=2)[visible], clean.mean(axis=2)[visible]
    correlation = float(np.corrcoef(x,y)[0,1]) if min(x.std(),y.std())>1e-6 else None
    return {"transmitted_pixels":int(visible.sum()),"transmitted_mae":float(error[visible].mean()),
            "transmitted_p95":float(np.quantile(error[visible],.95)),
            "transmitted_luma_correlation":correlation,
            "planted_background_mae":float(np.abs(marked-clean)[visible].mean()),
            "scope":"known synthetic transmitted background only; not quality permission"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work",type=Path,required=True)
    p.add_argument("--jobs",nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--state-dir",type=Path,default=Path.home()/".local/state/watermark-studio")
    a = p.parse_args()
    if a.output.exists():
        p.error("Fresh diagnostic output required")
    fixtures = json.loads((a.work/"fixture.json").read_text())
    selection = json.loads((a.work/"source-selection.json").read_text())
    reports = []
    for job in a.jobs:
        summary = json.loads((a.state_dir/job/"summary.json").read_text())
        rows = {r["file"]:r for r in map(json.loads,(a.state_dir/job/"results.jsonl").read_text().splitlines())}
        if (summary["state"] != "completed" or Path(summary["input_dir"]).resolve() != (a.work/"input").resolve()
                or set(rows) != {f["file"] for f in fixtures}):
            raise ValueError("Complete matched control job required")
        scores = []
        for fixture in fixtures:
            name = fixture["file"]
            gt, source, output = a.work/"ground-truth"/fixture["ground_truth"], a.work/"input"/name, Path(summary["output_dir"])/name
            hashes = [digest(f) for f in (gt,source,output)]
            if hashes[0] != fixture["ground_truth_sha256"] or hashes[1] != selection["source_sha256"][name]:
                raise ValueError("Known truth/control changed")
            row = rows[name]
            if row.get("quality_attempts",1) != 1 or len(row.get("attempt_history",[]))>1:
                raise ValueError("GEN1 only; retries cannot be pooled")
            images = [load_photo(path)[0] for path in (gt,source,output)]
            metrics = known_body_metrics(*images,fixture["geometry"])
            # Small, bounded outside-region comparison: diagnostic, never a threshold waiver.
            mask = Image.new("L",images[0].size,255)
            draw = ImageDraw.Draw(mask)
            for key in ("panel","footer"):
                draw.rectangle(fixture["geometry"][key],fill=0)
            mask.thumbnail((512,512),Image.Resampling.NEAREST)
            outside = np.asarray(mask)>0
            reference, candidate = [np.asarray(im.resize(mask.size,Image.Resampling.LANCZOS),dtype=np.float32)
                                    for im in (images[0],images[2])]
            metrics["outside_known_overlay_mae_512"] = float(np.abs(candidate-reference)[outside].mean())
            scores.append({"file":name,"status":row.get("quality",{}).get("status"),
                           "model_inferences":row.get("model_inferences"),"metrics":metrics})
            assert hashes == [digest(f) for f in (gt,source,output)]
        reports.append({"job":job,"count":len(scores),"quality_counts":dict(collections.Counter(r["status"] for r in scores)),
                        "mean_transmitted_mae":float(np.mean([r["metrics"]["transmitted_mae"] for r in scores])),
                        "mean_outside_mae_512":float(np.mean([r["metrics"]["outside_known_overlay_mae_512"] for r in scores])),
                        "rows":scores})
    report = {"scope":"known synthetic background development; not human-labelled or unseen qualification",
              "files_unchanged":True,"quality_permission":False,"jobs":reports}
    a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({**report,"jobs":[{k:v for k,v in r.items() if k!="rows"} for r in reports]}),flush=True)


if __name__ == "__main__":
    main()
