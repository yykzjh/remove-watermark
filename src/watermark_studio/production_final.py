"""Reconcile every final byte, original name, image dimension and immutable audit grade."""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path

from PIL import Image

from .codex_photo_audit import validate_verdict
from .dataset_transfer import beneath, write_json


def finalize(assembly, audit, repairs, output, state, workers=8):
    assembly, audit, repairs, output, state = map(Path, [assembly, audit, repairs, output, state])
    state.mkdir(parents=True, exist_ok=True)
    original = json.loads((assembly / "reports/cdu_dev1/global/original-manifest.json").read_text())
    assembly_report = json.loads((assembly / "report.json").read_text())
    if hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest() != assembly_report["manifest_sha256"]:
        raise ValueError("Original manifest changed after verified assembly")
    references = {r["path"]: r for r in original["files"]}
    baseline = {r["file"]: r for r in (json.loads(line) for line in
                (assembly / "assembled-images.jsonl").read_text().splitlines())}
    reviews = [json.loads(line) for line in (audit / "audit-results.jsonl").read_text().splitlines()]
    changes = [json.loads(line) for line in (repairs / "repair-results.jsonl").read_text().splitlines()]
    review_map, changed = {r["file"]: r for r in reviews}, {r["file"]: r for r in changes}
    if len(review_map) != len(reviews) or len(changed) != len(changes):
        raise ValueError("Duplicate final audit or repair records")
    for row in reviews:
        validate_verdict(row["verdict"])
    for row in changes:
        if row["status"] == "passed":
            validate_verdict(row["final_verdict"])
            if not row["final_verdict"]["acceptable"]:
                raise ValueError("Recorded repair pass contradicts its independent verdict")
        elif row["status"] == "accepted_by_user":
            proof = row["human_acceptance"]
            if (proof["decision"] != "accepted_by_user" or proof["file"] != row["file"]
                    or proof["candidate_sha256"] != row["final_sha256"]):
                raise ValueError("Human acceptance belongs to a different image")
    plan = json.loads((audit / "audit-plan.json").read_text())
    if set(review_map) != set(plan["sample"]) | set(plan["initially_rejected"]):
        raise ValueError("Not every required photo received its individual Codex review")
    rejected = {r["file"] for r in reviews if not r["verdict"]["acceptable"]}
    if set(changed) != rejected:
        raise ValueError("Not every rejected photo received an individual repair disposition")
    deferred = set(plan.get("deferred_server_review", []))
    if not deferred <= set(baseline):
        raise ValueError("Deferred review contains an unknown image")
    if plan.get("minimum_server_attempts", 0):
        server_bad = {name for name, row in baseline.items() if row.get("quality_status") != "passed"}
        if server_bad != set(plan["initially_rejected"]) | deferred:
            raise ValueError("Server failures were omitted from reviewed or deferred dispositions")
    actual_files, actual_dirs = set(), set()
    for folder, directories, names in os.walk(output):
        actual_dirs.add(Path(folder).relative_to(output).as_posix())
        for name in [*directories, *names]:
            if (Path(folder) / name).is_symlink():
                raise ValueError("Final tree contains a symbolic link")
        actual_files.update((Path(folder) / n).relative_to(output).as_posix() for n in names)
    if actual_files != set(references) or actual_dirs != set(original["directories"]):
        raise ValueError("Final tree differs from the complete original file/directory names")

    def verify(item):
        name, reference = item
        p = beneath(output, name)
        sha, md5 = hashlib.sha256(), hashlib.md5()
        with p.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                sha.update(block)
                md5.update(block)
        size = p.stat().st_size
        if name not in baseline:
            if size != reference["size"] or md5.hexdigest() != reference["md5"]:
                raise ValueError("Auxiliary original bytes changed: " + name)
            return {"file": name, "size": size, "sha256": sha.hexdigest(),
                    "status": "preserved_original_auxiliary"}
        initial = baseline[name]
        expected = changed[name]["final_sha256"] if name in changed else initial["sha256"]
        if sha.hexdigest() != expected:
            raise ValueError("Final image differs from its accepted or recorded best output: " + name)
        if name in changed and changed[name]["final_sha256"] != initial["sha256"]:
            with Image.open(p) as im:
                im.load()
                if list(im.size) != initial["original_size"]:
                    raise ValueError("Codex repair changed the original image dimensions")
        if name in changed:
            status = "codex_" + changed[name]["status"]
        elif name in review_map:
            status = "codex_verified"
        elif name in deferred:
            status = "server_needs_review_outside_local_scope"
        else:
            status = "self_hosted_verified_unsampled"
        return {"file": name, "size": size, "sha256": sha.hexdigest(), "status": status,
                "original_size": initial["original_size"], "initial_sha256": initial["sha256"]}

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(verify, sorted(references.items())))
    remaining = sorted(deferred | {r["file"] for r in changes if r["status"] not in {"passed", "accepted_by_user"}})
    sample = set(plan["sample"])
    initial_sample_passed = sum(review_map[name]["verdict"]["acceptable"] for name in sample)
    final_sample_passed = sum(review_map[name]["verdict"]["acceptable"] or
                              changed.get(name, {}).get("status") in {"passed", "accepted_by_user"}
                              for name in sample)
    report = {"state": "completed_with_review" if remaining else "completed",
              "files": len(rows), "directories_including_root": len(actual_dirs),
              "valid_images": len(baseline), "preserved_auxiliary_files": len(rows) - len(baseline),
              "individually_reviewed_by_codex": len(reviews), "initial_sample_size": len(sample),
              "initial_sample_passed": initial_sample_passed, "sample_passed_after_repairs": final_sample_passed,
              "sample_reviewed_while_processing": sum(
                  bool(review_map[name].get("reviewed_while_processing")) for name in sample),
              "codex_repair_dispositions": len(changes), "remaining_valid_images_needing_review": len(remaining),
              "source_empty_images": assembly_report["source_empty_images"],
              "server_review_deferred_by_scope": len(deferred),
              "previously_completed_preserved": len(plan.get("completed_before_scope_change", [])),
              "manifest_sha256": assembly_report["manifest_sha256"], "output": str(output)}
    write_json(state / "final-output-manifest.json", {"files": rows, "source_manifest_sha256": report["manifest_sha256"]})
    (state / "needs-review-relative-paths.txt").write_text("".join(name + "\n" for name in remaining))
    write_json(state / "report.json", report)
    return report
