"""Read-only local transmission diagnostics on approved source cards; no repair."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from watermark_studio.images import load_photo, pixel_box
from watermark_studio.translucent_hint import translucent_body_hint
from watermark_studio.translucent_recovery import local_transmission_support

p = argparse.ArgumentParser()
p.add_argument("--job", required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--state-dir", type=Path, default=Path.home()/".local/state/watermark-studio")
a = p.parse_args()
if a.output.exists():
    p.error("Fresh report required")
folder = a.state_dir/a.job
summary = json.loads((folder/"summary.json").read_text())
if summary["state"] != "completed":
    p.error("Completed development job required")
rows = [json.loads(line) for line in (folder/"results.jsonl").read_text().splitlines()]
started, reports = time.monotonic(), []
for row in rows:
    proofs = [r for r in row.get("source_rim_protection", []) if r.get("approved")]
    if not proofs or row.get("action") != "restored":
        continue
    source = Path(summary["input_dir"])/row["file"]
    output = Path(summary["output_dir"])/row["file"]
    hashes = [hashlib.sha256(f.read_bytes()).hexdigest() for f in (source, output)]
    before, after = load_photo(source)[0], load_photo(output)[0]
    if before.size != after.size:
        raise ValueError("Native source/candidate grid required")
    _, audits = translucent_body_hint(before, proofs)
    fits = []
    for proof, audit in zip(proofs, audits, strict=True):
        if not audit.get("applied"):
            fits.append({"supported":False, "reason":audit.get("skip_reason")})
            continue
        left, _, right, bottom = pixel_box(proof["core"], before.size)
        top = audit["body_pixels"][1]
        crop = (left+12, top+8, right-12, bottom-12)
        observed = before.crop(crop)
        observed.thumbnail((512,512), Image.Resampling.LANCZOS)
        predicted = after.crop(crop).resize(observed.size, Image.Resampling.LANCZOS)
        rgb = np.asarray(observed, dtype=np.float32)
        ink = Image.fromarray((rgb.mean(axis=2)<100).astype(np.uint8)*255)
        valid = np.asarray(ink.filter(ImageFilter.MaxFilter(9))) == 0
        valid &= (np.ptp(rgb,axis=2)<45) & (rgb.mean(axis=2)>125) & (rgb.mean(axis=2)<250)
        fit = local_transmission_support(
            np.asarray(observed.filter(ImageFilter.GaussianBlur(.6)),dtype=np.float32),
            np.asarray(predicted.filter(ImageFilter.GaussianBlur(.6)),dtype=np.float32), valid=valid)
        fits.append(fit | {"native_body":list(crop)})
    assert hashes == [hashlib.sha256(f.read_bytes()).hexdigest() for f in (source, output)]
    reports.append({"file":row["file"], "status":row["quality"]["status"],
                    "sha256":hashes, "fits":fits})
result = {"scope":"development physical-fit diagnostic only; no changes or quality approval",
          "job":a.job,"count":len(reports), "supported":sum(any(f.get("supported") for f in r["fits"]) for r in reports),
          "seconds":round(time.monotonic()-started,3), "source_and_candidate_unchanged":True,"rows":reports}
a.output.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!="rows"}),flush=True)
