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
                         "candidate_sha256": candidate_sha})
    (target / "results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False))
    frozen = {"id": tag, "state": "completed", "total": len(records),
              "input_dir": str(source_root), "output_dir": str(outputs.resolve()),
              "origin_job": job, "origin_revision": summary.get("code_revision"),
              "scope": "immutable first-generation candidates; original vetoes retained",
              "first_counts": summary.get("first_pass_counts")}
    (target / "summary.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2))
    return frozen
