"""Download a Baidu cloud directory directly to a data disk and verify its manifest.

Requires an existing BaiduPCS-Go login. Cookies and download URLs stay on this host.
"""
import argparse
import concurrent.futures as cf
import json
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath


def complete(r):
    """A resumable transfer marker always means the file is still incomplete."""
    f = Path(r["local"])
    try:
        return (f.is_file() and f.stat().st_size == r["size"]
                and not Path(str(f)+".BaiduPCS-Go-downloading").exists())
    except OSError:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cloud-root", required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--download-index", type=int, default=0, help="BaiduPCS-Go alternative download link index")
    p.add_argument("--cookie-file", type=Path, help="Private authorization file; never printed or copied into state")
    p.add_argument("--state-dir", type=Path, help="Keep task metadata outside the dataset directory")
    p.add_argument("--inventory-only", action="store_true")
    p.add_argument("--include-non-images", action="store_true")
    args = p.parse_args()
    if args.download_index < 0:
        p.error("--download-index must be nonnegative")
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    state = (args.state_dir.resolve() if args.state_dir else root / ".download-state")
    state.mkdir(exist_ok=True, parents=True)
    if args.cookie_file:
        cookie = args.cookie_file.read_text().strip()
    else:
        config = json.loads((Path.home() / ".config/BaiduPCS-Go/pcs_config.json").read_text())
        account = next(u for u in config["baidu_user_list"] if u["uid"] == config["baidu_active_uid"])
        cookie = account["cookies"]
    if not isinstance(cookie, str):
        raise TypeError("Expected cookie string")
    tool = str(Path.home() / ".local/bin/BaiduPCS-Go")


    def save(name, data):
        temp = state / (name + ".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False))
        temp.replace(state / name)


    def listing(directory):
        found = []
        for page in range(1, 100000):
            query = urllib.parse.urlencode({"dir": directory, "num": 1000, "page": page,
                                               "order": "name", "desc": 0, "web": 1})
            for attempt in range(6):
                try:
                    req = urllib.request.Request("https://pan.baidu.com/api/list?" + query,
                        headers={"Cookie": cookie, "User-Agent": "Mozilla/5.0",
                                 "Referer": "https://pan.baidu.com/disk/main"})
                    with urllib.request.urlopen(req, timeout=45) as response:
                        data = json.load(response)
                    if data.get("errno") != 0:
                        raise RuntimeError("Cloud listing errno=" + str(data.get("errno")))
                    items = data["list"]
                    break
                except Exception:
                    if attempt == 5:
                        raise
                    time.sleep(2 ** attempt)
            found.extend(items)
            if len(items) < 1000:
                return found
        raise RuntimeError("Pagination limit")


    source = state / "source.json"
    source_identity = {"cloud_root": args.cloud_root, "destination": str(root)}
    if source.exists() and json.loads(source.read_text()) != source_identity:
        raise ValueError("Existing task state belongs to another cloud root or destination")
    save("source.json", source_identity)
    manifest = state / "manifest.json"
    if manifest.exists():
        rows = json.loads(manifest.read_text())
    else:
        rows, pending, directories = [], [args.cloud_root], 0
        cloud_directories = [args.cloud_root]
        with cf.ThreadPoolExecutor(max_workers=8) as pool:
            while pending:
                next_dirs = []
                for items in pool.map(listing, pending):
                    directories += 1
                    for item in items:
                        if item["isdir"]:
                            next_dirs.append(item["path"])
                            cloud_directories.append(item["path"])
                        else:
                            rows.append({k: item[k] for k in ("path", "size", "md5") if k in item})
                pending = next_dirs
                save("status.json", {"phase": "inventory", "directories": directories,
                                        "discovered": len(rows), "remaining_directories": len(pending)})
        if len({r["path"] for r in rows}) != len(rows):
            raise RuntimeError("Duplicate manifest paths")
        save("manifest.json", rows)
        save("directories.json", cloud_directories)

    base = PurePosixPath(args.cloud_root)
    for r in rows:
        rel = PurePosixPath(r["path"]).relative_to(base)
        local = root.joinpath(*rel.parts)
        if not local.resolve().is_relative_to(root):
            raise ValueError("Unsafe manifest path")
        r["local"] = str(local)

    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif", ".avif", ".gif"}
    images = [r for r in rows if Path(r["path"]).suffix.lower() in image_exts]
    selected = rows if args.include_non_images else images
    save("inventory-summary.json", {"files": len(rows), "images": len(images),
        "image_bytes": sum(r["size"] for r in images), "non_images": len(rows)-len(images),
        "selected_files": len(selected), "selected_bytes": sum(r["size"] for r in selected)})
    if args.inventory_only:
        save("status.json", {"phase": "inventory_complete", "files": len(rows), "images": len(images)})
        raise SystemExit(0)
    # Preserve even empty source directories when the whole dataset was requested.
    directory_manifest = state / "directories.json"
    if args.include_non_images and directory_manifest.exists():
        for directory in json.loads(directory_manifest.read_text()):
            rel = PurePosixPath(directory).relative_to(base)
            target = root.joinpath(*rel.parts)
            if not target.resolve().is_relative_to(root):
                raise ValueError("Unsafe source directory")
            target.mkdir(parents=True, exist_ok=True)


    def download(batch):
        parent = Path(batch[0]["local"]).parent
        parent.mkdir(parents=True, exist_ok=True)
        # Every batch shares one parent, so --saveto preserves hierarchy without collisions.
        log = state / ("batch-" + str(batch[0]["index"]) + ".log")
        command = [tool, "download", "-p", "16", "-l", "4", "--retry", "5",
                   "--dindex", str(args.download_index),
                   "--saveto", str(parent), *[r["path"] for r in batch]]
        with log.open("a") as output:
            try:
                subprocess.run(command, stdout=output, stderr=output, timeout=3600, check=False)
            except subprocess.TimeoutExpired:
                return False
        return all(complete(r) for r in batch)


    for i, r in enumerate(selected):
        r["index"] = i
    for attempt in range(1, 7):
        missing = [r for r in selected if not complete(r)]
        if not missing:
            break
        groups = {}
        for r in missing:
            groups.setdefault(str(Path(r["local"]).parent), []).append(r)
        batches = [group[i:i+16] for group in groups.values() for i in range(0, len(group), 16)]
        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(download, b) for b in batches]
            while futures:
                _, unfinished = cf.wait(futures, timeout=20, return_when=cf.FIRST_COMPLETED)
                futures = list(unfinished)
                done = [r for r in selected if complete(r)]
                present = sum(min(Path(r["local"]).stat().st_size, r["size"])
                              for r in selected if Path(r["local"]).is_file())
                save("status.json", {"phase": "download", "attempt": attempt, "total": len(selected), "images": len(images),
                    "complete": len(done), "bytes": present, "total_bytes": sum(r["size"] for r in selected),
                    "remaining_batches": len(futures)})
    if any(not complete(r) for r in selected):
        save("status.json", {"phase": "failed", "missing": sum(not complete(r) for r in selected)})
        raise RuntimeError("Incomplete image downloads; manifest retained for resume")

    # Full decoding, not filename extensions alone, determines valid images.
    from PIL import Image


    def check(r):
        try:
            with Image.open(r["local"]) as im:
                im.load()
            return None
        except Exception as exc:  # noqa: BLE001 - record every per-image decode failure
            return {"path": r["path"], "size": r["size"], "error": type(exc).__name__}

    failures = []
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        for count, result in enumerate(pool.map(check, images), 1):
            if result:
                failures.append(result)
            if count % 500 == 0:
                save("status.json", {"phase": "validate", "checked": count, "total": len(images),
                                        "invalid": len(failures)})
    save("validation.json", {"total": len(images), "valid": len(images)-len(failures),
                                 "invalid": failures})
    save("status.json", {"phase": "completed", "images": len(images), "valid": len(images)-len(failures),
                            "invalid": len(failures), "files": len(selected), "bytes": sum(r["size"] for r in selected)})


if __name__ == "__main__":
    main()
