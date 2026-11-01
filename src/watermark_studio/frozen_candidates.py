"""Freeze original first candidates for read-only verifier comparisons."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bounded(root, value):
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Candidate name must be relative without parent traversal")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Candidate escapes its frozen root")
    return target


def first_snapshot(state_root, job, target_root, tag):
    """Link unchanged first outputs; never replace source images or job records."""
    for value in (job, tag):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", value):
            raise ValueError("Simple job and snapshot identifiers required")
    folder = (Path(state_root) / job).resolve()
    summary = json.loads((folder / "summary.json").read_text())
    if summary.get("state") != "completed":
        raise ValueError("A completed source job is required")
    final = [json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines()]
    if len(final) != summary["total"] or len({r["file"] for r in final}) != len(final):
        raise ValueError("Incomplete or repeated final records")
    initial = {r["file"]: r for r in final if r.get("quality_attempts", 1) == 1}
    events = folder / "retry-events.jsonl"
    if events.exists():
        for line in events.read_text().splitlines():
            row = json.loads(line)
            if row.get("quality_attempts", 1) == 1:
                if len(row.get("attempt_history", [])) > 1:
                    raise ValueError("First candidate has a later attempt history")
                initial[row["file"]] = row
    source_root = Path(summary["input_dir"]).resolve()
    output_root = Path(summary["output_dir"]).resolve()
    prepared = []
    for row in final:
        history = row.get("attempt_history") or [row]
        first = history[0]
        original = initial.get(row["file"])
        if original is None:
            raise ValueError("Missing full first-candidate provenance")
        if original.get("quality", {}).get("status") != first["quality"]["status"]:
            raise ValueError("First verdict differs from recorded attempt history")
        source = bounded(source_root, row["file"])
        first_status = first["quality"]["status"]
        if first_status == "passed":
            if len(history) != 1:
                raise ValueError("A first-pass acceptance must not have been regenerated")
            candidate = bounded(output_root, row["file"])
        else:
            retained = row.get("first_rejected_candidate")
            expected = row.get("first_rejected_sha256")
            if not retained or not expected:
                raise ValueError("Rejected first candidate was not retained")
            candidate = Path(retained).resolve()
            if not candidate.is_relative_to(folder) or digest(candidate) != expected:
                raise ValueError("Retained first candidate escaped or its SHA changed")
        if not source.is_file() or not candidate.is_file() or not original.get("ok"):
            raise ValueError("Missing or unsuccessful original candidate")
        # Source masks and harmonization come from the first event, never final retries.
        record = {k: original[k] for k in (
            "file", "ok", "repair_regions", "source_scene", "background_preserved",
            "protected_core_boxes", "harmonization", "processing_path",
            "source_rim_protection", "denoising_context",
        ) if k in original}
        record.update(quality=first["quality"], quality_attempts=1)
        prepared.append((source, candidate, record, digest(source), digest(candidate)))
    target = Path(target_root) / tag
    target.mkdir(parents=True, exist_ok=False)
    outputs = target / "outputs"
    outputs.mkdir()
    manifest, records = [], []
    for source, candidate, record, source_sha, candidate_sha in prepared:
        destination = bounded(outputs, record["file"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(candidate, destination)
        except OSError:
            shutil.copy2(candidate, destination)
        if digest(destination) != candidate_sha or digest(source) != source_sha:
            raise RuntimeError("Candidate or source changed while preparing snapshot")
        records.append(record)
        manifest.append({"file": record["file"], "source_sha256": source_sha,
                         "candidate_sha256": candidate_sha, "record_sha256": record_digest(record)})
    (target / "results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False))
    frozen = {"id": tag, "state": "completed", "total": len(records),
              "input_dir": str(source_root), "output_dir": str(outputs.resolve()),
              "origin_job": job, "origin_revision": summary.get("code_revision"),
              "snapshot_schema":2,"quality_weight_identity":summary.get("quality_weight_identity"),
              "scope": "immutable first-generation candidates; original vetoes retained",
              "first_counts": summary.get("first_pass_counts")}
    (target / "summary.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2))
    return frozen


def record_digest(record):
    content = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def validate_first_snapshot(root, identity):
    """Reject changed pixels, metadata or a different cached primary checkpoint."""
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    previous = summary.get("quality_weight_identity") or {}
    keys = ("editor_sha256", "verifier_sha256")
    if (summary.get("snapshot_schema") != 2 or previous.get("relation") != "different_weights"
            or identity.get("relation") != "different_weights"
            or not all(previous.get(k) and previous.get(k) == identity.get(k) for k in keys)):
        raise ValueError("Immutable first snapshot and identical canonical primary weights required")
    records = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
    manifest = json.loads((root / "manifest.json").read_text())
    expected = {r["file"]: r for r in manifest}
    if (len(expected) != len(manifest) or len(records) != summary["total"]
            or len({r["file"] for r in records}) != len(records)
            or {r["file"] for r in records} != set(expected)):
        raise ValueError("Snapshot manifest coverage differs")
    for row in records:
        item = expected[row["file"]]
        if record_digest(row) != item.get("record_sha256"):
            raise ValueError("Frozen first verdict or geometry metadata changed")
        for key, directory in (("source_sha256", summary["input_dir"]),
                               ("candidate_sha256", summary["output_dir"])):
            if digest(bounded(Path(directory), row["file"])) != item[key]:
                raise ValueError("Frozen source or first-candidate pixels changed")
    return summary


def validate_fresh_inspection(root, identity):
    """Re-inspect immutable pixels with a new independent checkpoint, never cached flags.

    Cached adjudication must still use validate_first_snapshot with the current
    identity. This separate path only attests original input/record integrity;
    the caller must compute all new model judgments from pixels.
    """
    root = Path(root)
    summary = json.loads((root/"summary.json").read_text())
    previous = summary.get("quality_weight_identity") or {}
    if (identity.get("relation") != "different_weights"
            or not identity.get("verifier_sha256")
            or identity.get("editor_sha256") != previous.get("editor_sha256")
            or identity.get("verifier_sha256") == identity.get("editor_sha256")):
        raise ValueError("Fresh inspection needs the same editor and independent verifier weights")
    validate_first_snapshot(root, previous)
    return {"cached_verdicts_reused": False,
            "original_verifier_sha256": previous["verifier_sha256"],
            "fresh_verifier_sha256": identity["verifier_sha256"],
            "original_editor_sha256": previous["editor_sha256"],
            "source_candidate_and_record_integrity": True}
