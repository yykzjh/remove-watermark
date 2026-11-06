"""Verify both immutable shards and rebuild exactly the original recursive tree."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import stat
import zipfile
from pathlib import Path

from .dataset_transfer import beneath, digest, relative_path, write_json
from .production_archives import file_sha256


def checked_entries(archive):
    seen = set()
    for entry in archive.infolist():
        name = relative_path(entry.filename.rstrip("/"))
        if name in seen or stat.S_ISLNK(entry.external_attr >> 16):
            raise ValueError("Duplicate or symbolic-link ZIP entry")
        seen.add(name)
        yield entry, name


def extract_one(archive, entry, target, expected_sha=None, expected_size=None, temporary_root=None):
    if target.is_symlink():
        raise ValueError("Refusing symbolic-link output")
    if expected_size is not None and entry.file_size != expected_size:
        raise ValueError("ZIP member size differs from output manifest")
    if (target.is_file() and target.stat().st_size == entry.file_size and expected_sha
            and file_sha256(target) == expected_sha):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = temporary_root or target.parent
    temporary_root.mkdir(parents=True, exist_ok=True)
    temp = temporary_root / (hashlib.sha256(str(target).encode()).hexdigest() + ".part")
    h, count = hashlib.sha256(), 0
    with archive.open(entry) as source, temp.open("wb") as output:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            output.write(block)
            h.update(block)
            count += len(block)
    # ZipExtFile checks the member CRC while reading to EOF.
    if count != entry.file_size or expected_sha and h.hexdigest() != expected_sha:
        raise ValueError("Extracted output SHA256 mismatch")
    temp.replace(target)


def assemble(index_paths, artifacts, destination, state, decode_workers=8):
    artifacts, destination, state = Path(artifacts), Path(destination), Path(state)
    state.mkdir(parents=True, exist_ok=True)
    indexes = [json.loads(Path(p).read_text()) for p in index_paths]
    if {i["node"] for i in indexes} != {"cdu_dev1", "cdu_dev2"} or len(indexes) != 2:
        raise ValueError("Exactly one final index from each authorized server is required")
    if len({i["manifest_sha256"] for i in indexes}) != 1:
        raise ValueError("Servers used different source snapshots")
    for index in indexes:
        for artifact in index["artifacts"]:
            if Path(artifact["name"]).name != artifact["name"]:
                raise ValueError("Unsafe artifact basename")
            p = artifacts / artifact["name"]
            if p.stat().st_size != artifact["size"] or file_sha256(p) != artifact["sha256"]:
                raise ValueError("Artifact SHA256 differs from pinned index")
            if artifact["kind"] != "reports":
                continue
            with zipfile.ZipFile(p) as z:
                for entry, name in checked_entries(z):
                    if entry.is_dir():
                        continue
                    extract_one(z, entry, beneath(state / "reports" / index["node"], name),
                                temporary_root=state / "partial-files")
    global_root = state / "reports/cdu_dev1/global"
    original = json.loads((global_root / "original-manifest.json").read_text())
    identity = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
    if identity != indexes[0]["manifest_sha256"]:
        raise ValueError("Original source manifest identity differs from pinned receipt")
    ownership = json.loads((global_root / "ownership.json").read_text())
    if ownership["manifest_sha256"] != identity:
        raise ValueError("Ownership manifest identity mismatch")
    originals = {r["path"]: r for r in original["files"]}
    if len(originals) != len(original["files"]):
        raise ValueError("Duplicate original source paths")
    directories = set(original["directories"])
    rows, node_rows = {}, {}
    for index in indexes:
        manifest = json.loads((state / "reports" / index["node"] / "output-manifest.json").read_text())
        if manifest["manifest_sha256"] != identity or manifest["node"] != index["node"]:
            raise ValueError("Output manifest identity mismatch")
        local = {}
        for row in manifest["files"]:
            name = relative_path(row["file"])
            if name in rows or name in local:
                raise ValueError("Output file appears on more than one shard")
            local[name] = row
        expected_images = {p for unit in ownership["units"].values()
                           if unit["node"] == index["node"] for p in unit["files"]}
        expected = expected_images | ({r["path"] for r in ownership["auxiliary"]}
                                      if index["node"] == "cdu_dev1" else set())
        if set(local) != expected:
            raise ValueError("Output shard differs from final directory ownership")
        rows.update(local)
        node_rows[index["node"]] = local
    if set(rows) != set(originals):
        raise ValueError("Final shard union differs from complete input inventory")
    destination.mkdir(parents=True, exist_ok=True)
    for directory in directories:
        if directory != ".":
            beneath(destination, directory).mkdir(parents=True, exist_ok=True)
    seen_files = set()
    for index in indexes:
        for artifact in index["artifacts"]:
            if artifact["kind"] != "images":
                continue
            with zipfile.ZipFile(artifacts / artifact["name"]) as z:
                for entry, name in checked_entries(z):
                    if entry.is_dir():
                        if name not in directories:
                            raise ValueError("Unexpected output directory in archive")
                        beneath(destination, name).mkdir(parents=True, exist_ok=True)
                        continue
                    if name not in node_rows[index["node"]] or name in seen_files:
                        raise ValueError("Unexpected or duplicate image archive member")
                    row = rows[name]
                    extract_one(z, entry, beneath(destination, name), row["sha256"], row["size"],
                                state / "partial-files")
                    seen_files.add(name)
            write_json(state / "status.json", {"state": "assembling", "files": len(seen_files),
                                               "total": len(originals)})
    actual_files, actual_dirs = set(), set()
    for folder, _, names in os.walk(destination):
        actual_dirs.add(Path(folder).relative_to(destination).as_posix())
        actual_files.update((Path(folder) / n).relative_to(destination).as_posix() for n in names)
    if seen_files != set(originals) or actual_files != set(originals) or actual_dirs != directories:
        raise ValueError("Final output file/directory tree differs from source")
    auxiliary = {r["path"]: r for r in ownership["auxiliary"]}
    for name, reference in auxiliary.items():
        p = beneath(destination, name)
        if p.stat().st_size != reference["size"] or digest(p) != reference["md5"]:
            raise ValueError("Nonimage or invalid-original bytes changed")

    def check_image(item):
        from PIL import Image
        name, row = item
        if name in auxiliary:
            return
        with Image.open(beneath(destination, name)) as image:
            image.load()
            if list(image.size) != row["original_size"]:
                raise ValueError("Final image dimensions differ from original: " + name)

    with cf.ThreadPoolExecutor(max_workers=decode_workers) as pool:
        for _ in pool.map(check_image, rows.items()):
            pass
    records = [row for name, row in sorted(rows.items()) if name not in auxiliary]
    (state / "assembled-images.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    issues = [r for r in records if r["quality_status"] != "passed"]
    (state / "needs-review-relative-paths.txt").write_text("".join(r["file"] + "\n" for r in issues))
    report = {"state": "assembled", "files": len(rows), "directories_including_root": len(directories),
              "valid_images": len(records), "auxiliary_files": len(auxiliary),
              "initially_passed": len(records) - len(issues), "needs_codex_repair": len(issues),
              "source_empty_images": [r["path"] for r in auxiliary.values()
                                      if r["size"] == 0 and Path(r["path"]).suffix.lower()
                                      in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}],
              "manifest_sha256": identity, "destination": str(destination)}
    write_json(state / "report.json", report)
    write_json(state / "status.json", report)
    return report


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--indexes", nargs="+", type=Path, required=True)
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--decode-workers", type=int, default=8)
    a = p.parse_args()
    print(json.dumps(assemble(a.indexes, a.artifacts, a.destination, a.state, a.decode_workers)))


if __name__ == "__main__":
    main()
