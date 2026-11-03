"""Compare downloaded bytes with decoded checksum fields in the frozen cloud manifest."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import re
import time
from pathlib import Path, PurePosixPath

CODEC = "baidu_web_xor_swap_v1"


def decode_metadata_md5(value):
    if not isinstance(value, str) or len(value) != 32:
        raise ValueError("Unknown checksum metadata length/type")
    value = value.lower()
    if re.fullmatch("[0-9a-f]{32}", value):
        return value
    chars = list(value)
    if chars[9] not in "ghijklmnopqrstuv":
        raise ValueError("Unknown checksum metadata marker")
    chars[9] = format(ord(chars[9])-ord("g"), "x")
    if not re.fullmatch("[0-9a-f]{32}", "".join(chars)):
        raise ValueError("Unknown checksum metadata characters")
    mixed = "".join(format(int(c, 16)^(i&15), "x") for i, c in enumerate(chars))
    return mixed[8:16]+mixed[:8]+mixed[24:]+mixed[16:24]


def signature(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def verify_payload(path, metadata):
    before = signature(path)
    try:
        expected = decode_metadata_md5(metadata)
    except ValueError as error:
        return {"status": "reference_unavailable", "error": str(error), "signature": before, "stable": True}
    with path.open("rb") as f:
        actual = hashlib.file_digest(f, "md5").hexdigest()
    after = signature(path)
    return {"status": "verified" if actual == expected else "mismatch", "expected": expected,
            "actual": actual, "signature": after, "stable": before == after}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--task-state", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    p.add_argument("--poll-seconds", type=int, choices=range(1, 61), default=15)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=a.resume)
    manifest = a.task_state/"manifest.json"
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    source = json.loads((a.task_state/"source.json").read_text())
    root = Path(source["destination"]).resolve()
    identity = {"manifest_sha256": sha, "codec": CODEC, "root": str(root)}
    identity_file = a.output/"identity.json"
    if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
        raise ValueError("Cannot reuse checksum records for another frozen source")
    identity_file.write_text(json.dumps(identity))
    targets = {}
    for row in json.loads(manifest.read_text()):
        relative = PurePosixPath(row["path"]).relative_to(source["cloud_root"])
        path = root.joinpath(*relative.parts)
        if not path.resolve().is_relative_to(root) or relative.as_posix() in targets:
            raise ValueError("Unsafe or duplicate payload path")
        targets[relative.as_posix()] = (path, row)
    results_path = a.output/"results.jsonl"
    records = {}
    if a.resume and results_path.exists():
        lines = results_path.read_text().splitlines()
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines)-1:
                    break
                raise
            records[row["file"]] = row
    def save(name, value):
        temp = a.output/(name+".tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
        temp.replace(a.output/name)
    began = time.monotonic()
    with results_path.open("a") as log, cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
        while True:
            ready = []
            for relative, (path, row) in targets.items():
                if (not path.is_file() or path.stat().st_size != row["size"]
                        or Path(str(path)+".BaiduPCS-Go-downloading").exists()):
                    records.pop(relative, None)
                    continue
                if relative in records and records[relative]["signature"] != signature(path):
                    records.pop(relative)
                if relative not in records:
                    ready.append((relative, path, row))
            futures = {pool.submit(verify_payload, path, row.get("md5")): relative
                       for relative, path, row in ready}
            for future in cf.as_completed(futures):
                relative = futures[future]
                record = future.result()
                path = targets[relative][0]
                if record["stable"] and not Path(str(path)+".BaiduPCS-Go-downloading").exists():
                    record["file"] = relative
                    records[relative] = record
                    log.write(json.dumps(record, ensure_ascii=False)+"\n")
            log.flush()
            counts = {status: sum(r["status"] == status for r in records.values())
                      for status in ("verified", "mismatch", "reference_unavailable")}
            save("status.json", {"phase": "checking", "files": len(targets),
                                  "checked": len(records), **counts})
            if len(records) == len(targets):
                break
            time.sleep(a.poll_seconds)
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != sha:
        raise ValueError("Frozen cloud inventory changed")
    report = {"state": "completed" if counts["verified"] == len(targets) else "completed_with_unverified_files",
              "files": len(targets), **counts, "manifest_sha256": sha, "codec": CODEC,
              "seconds": time.monotonic()-began,
              "scope": "Full file-byte MD5 versus decoded provider metadata, not image-format or ZIP-member validation"}
    save("report.json", report)
    save("status.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
