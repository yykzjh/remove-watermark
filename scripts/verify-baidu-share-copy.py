"""Compare a downloaded-task inventory to the original share, without transferring payloads."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath


def compare_entries(shared, cloud, shared_root, cloud_root):
    def mapped(rows, root):
        result = {}
        for row in rows:
            key = PurePosixPath(row["path"]).relative_to(PurePosixPath(root)).as_posix()
            if key in result:
                raise ValueError("Duplicate source path")
            result[key] = row
        return result
    left, right = mapped(shared, shared_root), mapped(cloud, cloud_root)
    common = left.keys() & right.keys()
    sizes = sorted(k for k in common if left[k]["size"] != right[k]["size"])
    hashes = sorted(k for k in common if left[k].get("md5") and right[k].get("md5")
                    and left[k]["md5"] != right[k]["md5"])
    missing, extra = sorted(left.keys()-right.keys()), sorted(right.keys()-left.keys())
    return {"shared_files": len(left), "cloud_files": len(right), "missing_from_cloud": missing,
            "extra_in_cloud": extra, "size_mismatches": sizes, "md5_mismatches": hashes,
            "md5_comparisons": sum(bool(left[k].get("md5") and right[k].get("md5")) for k in common),
            "files_match": not (missing or extra or sizes or hashes)}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--cookie-file", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--task-state", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(a.metadata.read_text())
    roots = metadata["file_list"]
    if isinstance(roots, dict):
        roots = roots["list"]
    root = next(r["path"] for r in roots if r["server_filename"] == "八标照片集" and r["isdir"])
    cookie = a.cookie_file.read_text().strip()
    manifest = a.task_state/"manifest.json"
    cloud_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    source = json.loads((a.task_state/"source.json").read_text())
    def save(name, data):
        tmp = a.output/(name+".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        tmp.replace(a.output/name)
    def listing(directory):
        found = []
        for page in range(1, 100000):
            query = urllib.parse.urlencode({"uk": metadata["share_uk"], "shareid": metadata["shareid"],
                "dir": directory, "root": 0, "page": page, "num": 100, "order": "name",
                "desc": 0, "web": 1, "app_id": 250528, "clienttype": 0})
            for attempt in range(6):
                try:
                    request = urllib.request.Request("https://pan.baidu.com/share/list?"+query,
                        headers={"Cookie": cookie, "User-Agent": "Mozilla/5.0",
                                 "Referer": "https://pan.baidu.com/s/1mQf6F6gtj083sUuMlvmSaw"})
                    with urllib.request.urlopen(request, timeout=30) as f:
                        data = json.load(f)
                    if data.get("errno") != 0:
                        raise RuntimeError("Original-share inventory errno="+str(data.get("errno")))
                    rows = data["list"]
                    break
                except Exception:
                    if attempt == 5:
                        raise
                    time.sleep(2**attempt)
            found.extend(rows)
            if not rows or data.get("has_more") in (0, False) or len(rows) < 100:
                return found
        raise RuntimeError("Share pagination limit")
    began = time.monotonic()
    pending, directories, files = [root], [root], []
    done = 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
        while pending:
            following = []
            for items in pool.map(listing, pending):
                done += 1
                for row in items:
                    PurePosixPath(row["path"]).relative_to(PurePosixPath(root))
                    if row["isdir"]:
                        following.append(row["path"])
                        directories.append(row["path"])
                    else:
                        files.append({k: row[k] for k in ("path", "size", "md5") if k in row})
            pending = following
            save("status.json", {"phase": "inventory", "directories": done, "files": len(files),
                                  "remaining_directories": len(pending)})
    save("manifest.json", files)
    save("directories.json", directories)
    report = compare_entries(files, json.loads(manifest.read_text()), root, source["cloud_root"])
    cloud_dirs = json.loads((a.task_state/"directories.json").read_text())
    shared_dirs = {PurePosixPath(d).relative_to(PurePosixPath(root)).as_posix() for d in directories}
    own_dirs = {PurePosixPath(d).relative_to(PurePosixPath(source["cloud_root"])).as_posix() for d in cloud_dirs}
    report.update(directories_match=shared_dirs == own_dirs, directories=len(shared_dirs),
                  manifest_sha256=cloud_sha, seconds=time.monotonic()-began,
                  source_link="https://pan.baidu.com/s/1mQf6F6gtj083sUuMlvmSaw")
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != cloud_sha:
        raise ValueError("Frozen cloud snapshot changed")
    report["verified"] = report["files_match"] and report["directories_match"]
    save("report.json", report)
    save("status.json", {"phase": "completed", "verified": report["verified"]})
    print(json.dumps({k: len(v) if isinstance(v, list) else v for k, v in report.items()}), flush=True)


if __name__ == "__main__":
    main()
