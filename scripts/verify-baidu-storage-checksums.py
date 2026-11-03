"""Resolve mismatching web metadata against the authenticated storage object's checksum."""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import hashlib
import json
import re
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

DOMAINS = ("baidu.com", "baidupcs.com", "bcebos.com")


def secure_storage_url(value):
    u = urllib.parse.urlparse(value)
    if (u.scheme not in ("http", "https") or not u.hostname or u.username or u.password
            or not any(u.hostname == d or u.hostname.endswith("."+d) for d in DOMAINS)
            or u.port not in (None, 80, 443)):
        raise ValueError("Unexpected storage URL")
    return urllib.parse.urlunparse(u._replace(scheme="https", netloc=u.hostname))


class StorageRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        secured = secure_storage_url(newurl)
        result = super().redirect_request(request, fp, code, message, headers, secured)
        if result is not None:
            result.method = "HEAD"
        return result


def storage_reference(headers, expected_size):
    headers = {k.lower(): v for k, v in headers.items()}
    if int(headers.get("content-length", "-1")) != expected_size:
        raise ValueError("Storage object size differs from the frozen inventory")
    value = headers.get("content-md5", "").strip().strip('"')
    if re.fullmatch("[0-9a-fA-F]{32}", value):
        return value.lower()
    try:
        raw = base64.b64decode(value, validate=True)
    except ValueError as error:
        raise ValueError("Missing or invalid whole-object Content-MD5") from error
    if len(raw) != 16:
        raise ValueError("Missing or invalid whole-object Content-MD5")
    return raw.hex()


def signature(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def read_records(path):
    rows = {}
    if not path.exists():
        return rows
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines)-1:
                break
            raise
        rows[row["file"]] = row
    return rows


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--task-state", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    state = a.task_state
    raw = (state/"manifest.json").read_bytes()
    manifest_sha = hashlib.sha256(raw).hexdigest()
    source = json.loads((state/"source.json").read_text())
    root = Path(source["destination"]).resolve()
    manifest = {str(Path(row["path"]).relative_to(source["cloud_root"])): row for row in json.loads(raw)}
    home = Path.home()
    tool = home/".local/bin/BaiduPCS-Go"
    user_agent = json.loads((home/".config/BaiduPCS-Go/pcs_config.json").read_text()).get("user_agent", "netdisk")
    def inspect(item):
        relative, baseline = item
        path = root/relative
        try:
            before = signature(path)
            if before != baseline["signature"] or Path(str(path)+".BaiduPCS-Go-downloading").exists():
                return {"file": relative, "status": "changed", "signature": before}
            # Signed URLs are kept only in memory. Never write locator stdout to logs.
            location = source["cloud_root"].rstrip("/")+"/"+relative
            output = subprocess.run([str(tool), "locate", location], check=False, capture_output=True, text=True, timeout=25)
            if output.returncode:
                raise RuntimeError("Storage locator failed")
            plain = re.sub(r"\x1b\[[0-9;]*m", "", output.stdout)
            urls = re.findall(r"https?://[^\s\x1b]+", plain)
            digest = None
            last_error = "No storage candidates"
            for url in urls[:3]:
                try:
                    url = secure_storage_url(url)
                    opener = urllib.request.build_opener(StorageRedirect())
                    req = urllib.request.Request(url, method="HEAD",
                        headers={"User-Agent": user_agent, "Accept-Encoding": "identity"})
                    with opener.open(req, timeout=10) as response:
                        secure_storage_url(response.geturl())
                        digest = storage_reference(dict(response.headers), manifest[relative]["size"])
                    break
                except Exception as error:  # noqa: BLE001 - no URL or cookie in diagnostics
                    last_error = type(error).__name__
            if digest is None:
                raise RuntimeError("Storage checksum unavailable: "+last_error)
            actual = baseline.get("actual")
            if actual is None:
                with path.open("rb") as f:
                    actual = hashlib.file_digest(f, "md5").hexdigest()
            after = signature(path)
            if before != after or Path(str(path)+".BaiduPCS-Go-downloading").exists():
                return {"file": relative, "status": "changed", "signature": after}
            return {"file": relative, "status": "verified" if actual == digest else "mismatch",
                    "actual": actual, "storage_md5": digest, "signature": after,
                    "reference": "HTTPS storage Content-MD5 with exact whole-object length"}
        except Exception as error:  # noqa: BLE001 - an unavailable reference never grants approval
            return {"file": relative, "status": "unavailable", "error": type(error).__name__,
                    "signature": baseline.get("signature")}
    def save(name, value):
        temp = a.output/(name+".tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
        temp.replace(a.output/name)
    records = {}
    began = time.monotonic()
    with (a.output/"results.jsonl").open("x") as log, cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
        while True:
            base = read_records(state/"file-content-validation/results.jsonl")
            candidates = {rel: row for rel, row in base.items()
                          if row["status"] in ("mismatch", "reference_unavailable")}
            needed = []
            for rel, row in candidates.items():
                previous = records.get(rel)
                if not previous or previous.get("status") == "changed" or previous.get("signature") != row["signature"]:
                    needed.append((rel, row))
            for result in pool.map(inspect, needed):
                records[result["file"]] = result
                log.write(json.dumps(result, ensure_ascii=False)+"\n")
            log.flush()
            counts = {status: sum(r["status"] == status for r in records.values())
                      for status in ("verified", "mismatch", "unavailable", "changed")}
            save("status.json", {"phase": "checking", "candidates": len(candidates), **counts})
            if (state/"file-content-validation/report.json").exists() and not any(r["status"] == "changed" for r in records.values()):
                break
            time.sleep(15)
    if hashlib.sha256((state/"manifest.json").read_bytes()).hexdigest() != manifest_sha:
        raise ValueError("Frozen inventory changed")
    report = {"state": "completed" if counts["verified"] == len(candidates) else "completed_with_unverified_files",
              "candidates": len(candidates), **counts, "manifest_sha256": manifest_sha,
              "seconds": time.monotonic()-began,
              "scope": "Independent storage reference; original web-metadata mismatch records retained unchanged"}
    save("report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
