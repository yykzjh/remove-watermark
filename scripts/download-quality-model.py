"""Range-download official ModelScope model shards and verify published SHA-256."""
import concurrent.futures as cf
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

repo = "Qwen/Qwen3-VL-4B-Instruct"
root = Path.home()/".cache/watermark-models/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master"
root.mkdir(parents=True, exist_ok=True)
url = f"https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true"
with urllib.request.urlopen(url, timeout=30) as response:
    files = [f for f in json.load(response)["Data"]["Files"] if f["Name"].endswith(".safetensors")]
chunk = 32 * 1024 * 1024
for row in files:
    name, size = row["Name"], row["Size"]
    dest = root/name
    if dest.is_file() and dest.stat().st_size == size:
        with dest.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() == row["Sha256"]:
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
        with cf.ThreadPoolExecutor(max_workers=8) as pool:
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
(root/"verified-shards.json").write_text(json.dumps(files))
print("All model shards verified", flush=True)
