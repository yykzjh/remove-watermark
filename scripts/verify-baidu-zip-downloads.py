"""Verify every completed ZIP in a frozen download inventory; never extract archives."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import time
import zipfile
from pathlib import Path, PurePosixPath


def signature(path):
    st = path.stat()
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def verify_archive(path):
    before = signature(path)
    try:
        with zipfile.ZipFile(path) as archive:
            entries = len([i for i in archive.infolist() if not i.is_dir()])
            bad = archive.testzip()
        result = {"ok": bad is None, "bad_member": bad, "entries": entries, "error": None}
    except Exception as error:  # noqa: BLE001 - preserve per-archive failures
        result = {"ok": False, "bad_member": None, "entries": None,
                  "error": f"{type(error).__name__}: {error}"[:400]}
    after = signature(path)
    return result | {"signature": after, "stable": before == after}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--task-state", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    p.add_argument("--poll-seconds", type=int, choices=range(1, 61), default=15)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = a.task_state/"manifest.json"
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    source = json.loads((a.task_state/"source.json").read_text())
    root = Path(source["destination"]).resolve()
    targets = {}
    for row in json.loads(manifest.read_text()):
        if PurePosixPath(row["path"]).suffix.lower() != ".zip":
            continue
        rel = PurePosixPath(row["path"]).relative_to(source["cloud_root"])
        path = root.joinpath(*rel.parts)
        if not path.resolve().is_relative_to(root):
            raise ValueError("Unsafe archive path")
        targets[rel.as_posix()] = (path, row["size"])
    def save(name, value):
        temp = a.output/(name+".tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
        temp.replace(a.output/name)
    records = {}
    began = time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
        while len(records) < len(targets):
            ready = []
            for rel, (path, size) in targets.items():
                if (not path.is_file() or path.stat().st_size != size
                        or Path(str(path)+".BaiduPCS-Go-downloading").exists()):
                    records.pop(rel, None)
                    continue
                if rel in records and records[rel]["signature"] != signature(path):
                    records.pop(rel)
                if rel not in records:
                    ready.append((rel, path))
            futures = {pool.submit(verify_archive, path): rel for rel, path in ready}
            for future in cf.as_completed(futures):
                rel = futures[future]
                result = future.result()
                if result["stable"]:
                    records[rel] = result
                    save("records.json", records)
            save("status.json", {"phase": "checking", "archives": len(targets), "checked": len(records),
                                 "valid": sum(r["ok"] for r in records.values())})
            if len(records) < len(targets):
                time.sleep(a.poll_seconds)
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != sha:
        raise ValueError("Frozen inventory changed")
    report = {"state": "completed" if all(r["ok"] for r in records.values()) else "completed_with_errors",
              "archives": len(targets), "valid": sum(r["ok"] for r in records.values()),
              "invalid": [rel for rel, r in records.items() if not r["ok"]],
              "manifest_sha256": sha, "seconds": time.monotonic()-began,
              "method": "Full decompression and ZipFile.testzip CRC verification; no extraction"}
    save("report.json", report)
    save("status.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
