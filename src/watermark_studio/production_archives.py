"""Package immutable ownership-filtered image shards and complete verification metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from .dataset_transfer import beneath, signature, write_json


def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def committed_records(state, unit_ids):
    records = {}
    for unit_id in unit_ids:
        receipt = state / (unit_id + "-committed.json")
        report = state / (unit_id + "-results.json")
        if not receipt.exists() or not report.exists():
            raise ValueError("Missing committed directory evidence: " + unit_id)
        for row in json.loads(report.read_text())["rows"]:
            if row["file"] in records:
                raise ValueError("Image appears in multiple committed directories")
            records[row["file"]] = row
    return records


def build_archives(output, controller_state, ownership, original, node, archive_dir,
                   target_bytes=3 * 1024**3):
    archive_dir.mkdir(parents=True, exist_ok=True)
    index_path = archive_dir / "index.json"
    owned = {key: unit for key, unit in ownership["units"].items() if unit["node"] == node}
    expected_images = {p for unit in owned.values() for p in unit["files"]}
    records = committed_records(controller_state, owned)
    if set(records) != expected_images:
        raise ValueError("Owned image inventory differs from committed results")
    auxiliary = {r["path"]: r for r in ownership["auxiliary"]} if node == "cdu_dev1" else {}
    expected = expected_images | set(auxiliary)
    if expected_images & set(auxiliary):
        raise ValueError("Image and auxiliary ownership overlap")
    source_signatures = {p: signature(beneath(output, p)) for p in sorted(expected)}
    evidence_sha = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    if index_path.exists():
        saved = json.loads(index_path.read_text())
        if (saved.get("manifest_sha256") == ownership["manifest_sha256"]
                and saved.get("evidence_sha256") == evidence_sha
                and saved.get("source_signatures") == source_signatures
                and all(Path(a["local_path"]).is_file()
                        and signature(Path(a["local_path"])) == a["signature"]
                        for a in saved["artifacts"])):
            return saved
    rows = []
    for name in sorted(expected):
        source = beneath(output, name)
        if source.is_symlink() or not source.is_file():
            raise ValueError("Output is missing or not a normal file: " + name)
        record = records.get(name, {})
        rows.append({"file": name, "size": source.stat().st_size,
                     "expected_sha256": record.get("sha256"),
                     "expected_md5": auxiliary.get(name, {}).get("md5"),
                     "quality_status": record.get("quality_status", "original_auxiliary"),
                     "original_size": record.get("original_size")})
    groups, batch, byte_count = [], [], 0
    for row in rows:
        if batch and byte_count + row["size"] > target_bytes:
            groups.append(batch)
            batch, byte_count = [], 0
        batch.append(row)
        byte_count += row["size"]
    if batch or not groups:
        groups.append(batch)
    artifacts, outputs = [], []
    for number, group in enumerate(groups, 1):
        temporary = archive_dir / f"{node}-part{number:04d}.zip.partial"
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED,
                             allowZip64=True, strict_timestamps=False) as z:
            if node == "cdu_dev1" and number == 1:
                for directory in sorted(ownership["directories"]):
                    if directory != ".":
                        info = zipfile.ZipInfo(directory.rstrip("/") + "/")
                        info.external_attr = (0o40755 << 16) | 0x10
                        z.writestr(info, b"")
            for row in group:
                source = beneath(output, row["file"])
                before = signature(source)
                sha, md5, count = hashlib.sha256(), hashlib.md5(), 0
                info = zipfile.ZipInfo.from_file(source, arcname=row["file"], strict_timestamps=False)
                info.compress_type = zipfile.ZIP_STORED
                with source.open("rb") as f, z.open(info, "w", force_zip64=True) as target:
                    for block in iter(lambda: f.read(1024 * 1024), b""):
                        target.write(block)
                        sha.update(block)
                        md5.update(block)
                        count += len(block)
                if (before != signature(source) or count != row["size"]
                        or row["expected_sha256"] and sha.hexdigest() != row["expected_sha256"]
                        or row["expected_md5"] and md5.hexdigest() != row["expected_md5"]):
                    raise ValueError("Output changed or differs from accepted evidence: " + row["file"])
                outputs.append({k: v for k, v in row.items()
                                if k not in {"expected_sha256", "expected_md5"}}
                               | {"sha256": sha.hexdigest(), "archive_part": number})
        with zipfile.ZipFile(temporary) as z:
            if z.testzip() is not None:
                raise ValueError("New result archive failed CRC verification")
        sha = file_sha256(temporary)
        final = archive_dir / f"{node}-part{number:04d}-{sha[:16]}.zip"
        temporary.replace(final)
        artifacts.append({"name": final.name, "local_path": str(final),
                          "size": final.stat().st_size, "sha256": sha,
                          "signature": signature(final), "kind": "images", "files": len(group)})
    output_manifest = archive_dir / "output-manifest.json"
    write_json(output_manifest, {"node": node, "files": outputs,
                                "manifest_sha256": ownership["manifest_sha256"]})
    ownership_file, original_file = archive_dir / "ownership.json", archive_dir / "original-manifest.json"
    write_json(ownership_file, ownership)
    write_json(original_file, original)
    temporary = archive_dir / (node + "-reports.zip.partial")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.write(output_manifest, "output-manifest.json")
        if node == "cdu_dev1":
            z.write(ownership_file, "global/ownership.json")
            z.write(original_file, "global/original-manifest.json")
        for name in ["needs-review-relative-paths.txt", "needs-review.jsonl", "quality-summary.json",
                     "source-invalid.json", "gpu-checks.jsonl"]:
            p = controller_state / name
            if p.exists():
                z.write(p, name)
        for unit_id in owned:
            p = controller_state / (unit_id + "-results.json")
            z.write(p, "directories/" + p.name)
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None:
            raise ValueError("New reports archive failed CRC verification")
    sha = file_sha256(temporary)
    final = archive_dir / f"{node}-reports-{sha[:16]}.zip"
    temporary.replace(final)
    artifacts.append({"name": final.name, "local_path": str(final), "size": final.stat().st_size,
                      "sha256": sha, "signature": signature(final), "kind": "reports"})
    index = {"schema": 1, "node": node, "state": "packaged",
             "manifest_sha256": ownership["manifest_sha256"],
             "files": len(expected), "image_files": len(expected_images),
             "auxiliary_files": len(auxiliary), "artifacts": artifacts,
             "source_signatures": source_signatures, "evidence_sha256": evidence_sha}
    write_json(index_path, index)
    return index


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--controller-state", type=Path, required=True)
    p.add_argument("--ownership", type=Path, required=True)
    p.add_argument("--original-manifest", type=Path, required=True)
    p.add_argument("--node", choices=["cdu_dev1", "cdu_dev2"], required=True)
    p.add_argument("--archives", type=Path, required=True)
    p.add_argument("--part-bytes", type=int, default=3 * 1024**3)
    a = p.parse_args()
    index = build_archives(a.output, a.controller_state, json.loads(a.ownership.read_text()),
                           json.loads(a.original_manifest.read_text()), a.node, a.archives, a.part_bytes)
    print(json.dumps({"node": a.node, "files": index["files"],
                      "artifacts": len(index["artifacts"])}))


if __name__ == "__main__":
    main()
