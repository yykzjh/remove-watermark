"""Range-download official ModelScope model shards and verify published SHA-256."""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path


def download_small(address, size, expected_sha, attempts=6):
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(address,timeout=120) as response:
                data=response.read(size+1)
            if len(data)!=size or hashlib.sha256(data).hexdigest()!=expected_sha:
                raise ValueError("Model asset size/SHA256 mismatch")
            return data
        except Exception:
            if attempt+1==attempts:
                raise
            time.sleep(min(2**attempt,16))
    raise RuntimeError("No verified model asset")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--workers",type=int,choices=range(1,17),default=8)
    args=parser.parse_args()
    repo=args.model
    if len(repo.split("/"))!=2 or any(not part.replace("-","").replace("_","").replace(".","").isalnum() for part in repo.split("/")):
        parser.error("Model must be owner/repository")
    root=args.root or Path.home()/".cache/watermark-models/models"/repo.replace("/","--")/"snapshots/master"
    root.mkdir(parents=True, exist_ok=True)
    url = f"https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true"
    with urllib.request.urlopen(url, timeout=30) as response:
        files = []
        for row in json.load(response)["Data"]["Files"]:
            relative = Path(row.get("Path", row["Name"]))
            if row.get("Type") == "tree" or row["Size"] == 0 and not row.get("Sha256"):
                continue
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe model asset path")
            if relative.parts[0] == "assets" or relative.suffix.lower() in {".md", ".png", ".jpg"}:
                continue
            files.append(row | {"Name": relative.as_posix()})
        files.sort(key=lambda f:(f["Name"].endswith(".safetensors"),f["Name"]))
    chunk = 32 * 1024 * 1024
    for row in files:
        name, size = row["Name"], row["Size"]
        dest = root/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_file() and dest.stat().st_size == size:
            with dest.open("rb") as f:
                if hashlib.file_digest(f, "sha256").hexdigest() == row["Sha256"]:
                    continue
        if size<=chunk:
            address=f"https://modelscope.cn/models/{repo}/resolve/master/{name}"
            data=download_small(address,size,row["Sha256"])
            temporary=root/(name+".download")
            temporary.write_bytes(data);temporary.replace(dest)
            print(json.dumps({"file":name,"sha256_verified":True}),flush=True)
            continue
        partial = root/(name+".ranged")
        bitmap = root/(name+".ranges.json")
        completed = set(json.loads(bitmap.read_text())) if bitmap.exists() and partial.exists() else set()
        fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o600)
        os.ftruncate(fd, size)

        def download(index, size=size, name=name, fd=fd):
            start = index*chunk
            end = min(size, start+chunk)-1
            address = f"https://modelscope.cn/models/{repo}/resolve/master/{name}"
            for retry in range(6):
                try:
                    request = urllib.request.Request(address, headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(request, timeout=45) as response:
                        expected = f"bytes {start}-{end}/{size}"
                        if response.status != 206 or response.headers.get("Content-Range") != expected:
                            raise ValueError("Range response mismatch")
                        offset = start
                        while data := response.read(1024*1024):
                            written = os.pwrite(fd, data, offset)
                            if written != len(data):
                                raise OSError("Short write")
                            offset += written
                        if offset != end+1:
                            raise OSError("Incomplete range")
                    return index
                except Exception:
                    if retry == 5:
                        raise
                    time.sleep(min(2**retry, 16))
        try:
            with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(download, i) for i in range((size+chunk-1)//chunk) if i not in completed]
                last = 0
                for future in cf.as_completed(futures):
                    completed.add(future.result())
                    bitmap.write_text(json.dumps(sorted(completed)))
                    if time.monotonic()-last > 20:
                        print(json.dumps({"file": name, "completed_bytes": min(size, len(completed)*chunk),
                                          "total_bytes": size}), flush=True)
                        last = time.monotonic()
            os.fsync(fd)
        finally:
            os.close(fd)
        with partial.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        if digest != row["Sha256"]:
            raise ValueError("Published SHA256 does not match downloaded shard")
        partial.replace(dest)
        print(json.dumps({"file": name, "sha256_verified": True}), flush=True)
    (root/"verified-files.json").write_text(json.dumps(files))
    print("All model shards verified", flush=True)


if __name__ == "__main__":
    main()
