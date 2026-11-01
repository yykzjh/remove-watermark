"""Replay two composition scopes on identical captured GEN1 pixels; no new generation.

This is a development diagnostic, never an automatic acceptance or holdout result.
It retains the original and tighter numeric checks and validates the current saved
candidate byte-for-byte before interpreting comparisons.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from PIL import Image

from watermark_studio.config import ProcessingConfig, Region
from watermark_studio.images import (
    composite_protected,
    load_photo,
    protect_edge_completion_context,
    save_output,
)
from watermark_studio.quality import repair_boundary_warnings, repair_texture_warnings
from watermark_studio.rectangle_phase import rectangle_phase
from watermark_studio.source_panel_rims import effective_rim_scope, restore_proven_rims


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def guards(before, after, regions, cores, records, feather):
    scopes = [(regions, cores)]
    if records:
        tighter, tighter_cores, invalid = effective_rim_scope(before, regions, cores, records)
        if invalid:
            raise ValueError("Stale source rim proof")
        scopes.append((tighter, tighter_cores))
    found = set()
    for masks, core in scopes:
        for code, result in (
            ("repair_boundary_step", repair_boundary_warnings(after, masks, feather, before=before)),
            ("repair_texture_collapse", repair_texture_warnings(
                after, masks, feather, before=before, core_boxes=core)),
            ("repair_rectangle_phase", rectangle_phase(before, after, masks)),
        ):
            if result["count"]:
                found.add(code)
    return sorted(found)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--job", required=True)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--state-dir", type=Path, default=Path.home()/".local/state/watermark-studio")
    a = p.parse_args()
    if a.output.exists():
        p.error("Fresh diagnostic output required")
    os.environ.pop("WATERMARK_REPAIR_CAPTURE", None)
    root = a.state_dir/a.job
    summary = json.loads((root/"summary.json").read_text())
    if summary["state"] != "completed":
        p.error("Completed source job required")
    rows = [json.loads(l) for l in (root/"results.jsonl").read_text().splitlines()]
    options = ProcessingConfig.model_validate(summary["options"])
    captures = {}
    for metadata in a.capture.glob("*/capture.json"):
        info = json.loads(metadata.read_text())
        key = info["original_rgb_sha256"]
        if key in captures:
            raise ValueError("Ambiguous duplicate raw generation")
        captures[key] = (metadata.parent, info)
    report, started = [], time.monotonic()
    with tempfile.TemporaryDirectory(prefix="composition-replay-") as temporary:
        for row in rows:
            records = [r for r in row.get("source_rim_protection", []) if r.get("approved")]
            if not records or row.get("action") != "restored":
                continue
            source = Path(summary["input_dir"])/row["file"]
            candidate = Path(summary["output_dir"])/row["file"]
            source_hash, candidate_hash = digest(source), digest(candidate)
            before, exif, icc, alpha = load_photo(source)
            rawdir, info = captures[hashlib.sha256(before.tobytes()).hexdigest()]
            raw = Image.open(rawdir/"generated.png").convert("RGB")
            if (list(raw.size) != info["native_size"]
                    or hashlib.sha256(raw.tobytes()).hexdigest() != info["generated_rgb_sha256"]):
                raise ValueError("Captured generation checksum mismatch")
            regions = [Region.model_validate(r) for r in row["repair_regions"]]
            cores = row["protected_core_boxes"]
            tighter, tighter_cores, invalid = effective_rim_scope(
                before, regions, cores, records, header_core=True)
            if invalid or [r.model_dump(mode="json") for r in tighter] != row["source_rim_composition"]["regions"]:
                raise ValueError("Current composition scope mismatch")
            item = {"file": row["file"], "source_sha256": source_hash,
                    "candidate_sha256": candidate_hash, "current_status": row["quality"]["status"]}
            for mode, masks, core in (("old", regions, cores), ("current", tighter, tighter_cores)):
                result = composite_protected(
                    before, raw, masks, options.feather, options.repair_blending,
                    core_boxes={tuple(c["mask"]): tuple(c["core"]) for c in core})
                result, _ = protect_edge_completion_context(
                    before, result, row.get("source_edge_completion", []))
                result, count = restore_proven_rims(before, result, records)
                if count != len(records):
                    raise ValueError("Invalid source proof")
                if alpha is not None:
                    result.putalpha(alpha)
                output = Path(temporary)/(mode+candidate.suffix)
                save_output(result, source, output, exif, icc, options)
                decoded = load_photo(output)[0]
                item[mode+"_guards"] = guards(before, decoded, regions, cores, records, options.feather)
                if mode == "current":
                    item["current_bytes_match"] = digest(output) == candidate_hash
                    if not item["current_bytes_match"]:
                        raise ValueError("Replayed candidate differs; no valid A/B claim")
            if digest(source) != source_hash or digest(candidate) != candidate_hash:
                raise ValueError("Read-only pair changed")
            report.append(item)
            if len(report) % 8 == 0:
                print(json.dumps({"replayed": len(report), "seconds": round(time.monotonic()-started, 2)}), flush=True)
    changed = [r for r in report if r["old_guards"] != r["current_guards"]]
    result = {"scope": "same raw generation CPU composition diagnostic; NOT first-pass acceptance",
              "job": a.job, "count": len(report), "seconds": round(time.monotonic()-started, 3),
              "all_current_bytes_match": all(r["current_bytes_match"] for r in report),
              "sources_candidates_unchanged": True,
              "old_guard_counts": dict(collections.Counter(g for r in report for g in r["old_guards"])),
              "current_guard_counts": dict(collections.Counter(g for r in report for g in r["current_guards"])),
              "guard_clear_gains": sum(bool(r["old_guards"]) and not r["current_guards"] for r in report),
              "guard_clear_losses": sum(not r["old_guards"] and bool(r["current_guards"]) for r in report),
              "changed": changed, "rows": report}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in {"rows", "changed"}}), flush=True)


if __name__ == "__main__":
    main()
