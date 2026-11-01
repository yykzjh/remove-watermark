"""CPU-only source-feature diagnostic on a SHA-bound development snapshot."""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from watermark_studio.config import Region
from watermark_studio.frozen_candidates import bounded, digest, validate_first_snapshot
from watermark_studio.images import load_photo
from watermark_studio.source_complexity import source_resolution_features

p = argparse.ArgumentParser()
p.add_argument("--state-dir", type=Path, required=True)
p.add_argument("--source-job", required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
if not re.fullmatch(r"[a-zA-Z0-9_-]+", a.source_job) or a.output.exists():
    p.error("Exact immutable source snapshot and fresh output required")
folder = a.state_dir/a.source_job
summary = json.loads((folder/"summary.json").read_text())
validate_first_snapshot(folder, summary["quality_weight_identity"])
records = [json.loads(line) for line in (folder/"results.jsonl").read_text().splitlines()]
manifest = {r["file"]: r for r in json.loads((folder/"manifest.json").read_text())}
began = time.monotonic()
rows = []
targets = {"repair_texture_collapse", "repair_boundary_step", "repair_rectangle_phase"}
for index, record in enumerate(records):
    source = bounded(Path(summary["input_dir"]), record["file"])
    regions = [Region.model_validate(r) for r in record.get("repair_regions", [])]
    features = source_resolution_features(load_photo(source)[0], regions,
        record.get("protected_core_boxes", []), scene=record.get("source_scene"))
    assert digest(source) == manifest[record["file"]]["source_sha256"]
    rows.append({"file": record["file"], "features": features,
                 "frozen_numeric_rejection": bool(targets & set(record.get("quality", {}).get("guard_codes", [])))})
    if (index+1) % 128 == 0:
        print(json.dumps({"sources": index+1, "total": len(records)}), flush=True)
eligible = sorted([r for r in rows if r["features"]["eligible"]],
                  key=lambda r: (-r["features"]["score"], r["file"]))
positive = np.array([r["features"]["score"] for r in eligible if r["frozen_numeric_rejection"]])
negative = np.array([r["features"]["score"] for r in eligible if not r["frozen_numeric_rejection"]])
auc = (float((positive[:, None] > negative).mean()+.5*(positive[:, None] == negative).mean())
       if len(positive) and len(negative) else None)
tiers = []
for fraction in [.10, .20, .30, .40]:
    selected = eligible[:max(1, round(len(eligible)*fraction))]
    tiers.append({"eligible_fraction": fraction, "images": len(selected),
                  "all_images_fraction": round(len(selected)/len(rows), 4),
                  "frozen_numeric_rejections_captured": sum(r["frozen_numeric_rejection"] for r in selected),
                  "score_cutoff": selected[-1]["features"]["score"] if selected else None})
validate_first_snapshot(folder, summary["quality_weight_identity"])
report = {"scope": "source-only development risk diagnostic, not a repair, QA approval or holdout",
          "source_job": a.source_job, "images": len(rows), "eligible": len(eligible),
          "all_frozen_numeric_rejections": sum(r["frozen_numeric_rejection"] for r in rows),
          "eligible_numeric_rejections": len(positive), "fixed_score_auc": auc, "tiers": tiers,
          "seconds": round(time.monotonic()-began, 3), "source_and_candidate_unchanged": True,
          "rows": rows}
a.output.write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)
