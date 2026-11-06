"""Authenticated, manifest-verified direct transfer between Linux data disks."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import http.server
import json
import os
import random
import secrets
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath


def relative_path(value, allow_root=False):
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or (str(p) == "." and not allow_root):
        raise ValueError("Unsafe manifest path")
    return p.as_posix()


def beneath(root, value):
    p = root / relative_path(value)
    p.resolve().relative_to(root.resolve())
    return p


def signature(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def digest(path):
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def frozen_inventory(root, state):
    source = json.loads((state / "source.json").read_text())
    if root.resolve() != Path(source["destination"]).resolve():
        raise ValueError("Inventory belongs to another dataset")
    checksums = {}
    for line in (state / "file-content-validation/results.jsonl").read_text().splitlines():
        row = json.loads(line)
        checksums[row["file"]] = row
    files = []
    for row in json.loads((state / "manifest.json").read_text()):
        rel = relative_path(PurePosixPath(row["path"]).relative_to(source["cloud_root"]).as_posix())
        p = beneath(root, rel)
        reference = checksums[rel]
        if p.is_symlink() or signature(p) != reference["signature"] or p.stat().st_size != row["size"]:
            raise ValueError("Source changed since full checksum verification: " + rel)
        if Path(str(p) + ".BaiduPCS-Go-downloading").exists():
            raise ValueError("Source transfer is incomplete")
        files.append({"path": rel, "size": row["size"], "md5": reference["actual"]})
    directories = [relative_path(PurePosixPath(d).relative_to(source["cloud_root"]).as_posix(), True)
                   for d in json.loads((state / "directories.json").read_text())]
    if len({f["path"] for f in files}) != len(files):
        raise ValueError("Duplicate manifest path")
    return {"schema": 1, "files": files, "directories": directories}, checksums


def partition_manifest(manifest, weights, depth=2):
    if not weights or any(n <= 0 for n in weights.values()):
        raise ValueError("Positive GPU counts are required")
    groups = {}
    for row in manifest["files"]:
        parent = PurePosixPath(row["path"]).parts[:-1]
        key = "/".join(parent[:depth]) or "."
        groups.setdefault(key, {"files": [], "directories": []})["files"].append(row)
    for directory in manifest["directories"]:
        key = "/".join(PurePosixPath(directory).parts[:depth]) or "."
        groups.setdefault(key, {"files": [], "directories": []})["directories"].append(directory)
    shards = {host: {"schema": 1, "name": host, "files": [], "directories": {"."},
                     "image_count": 0, "gpu_count": n} for host, n in weights.items()}
    loads = {host: [0, 0] for host in weights}

    def image_count(group):
        return sum(r["size"] > 0 and PurePosixPath(r["path"]).suffix.lower()
                   in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
                   for r in group["files"])

    for _, group in sorted(groups.items(), key=lambda kv: (-image_count(kv[1]),
                            -sum(r["size"] for r in kv[1]["files"]), kv[0])):
        host = min(weights, key=lambda h: (loads[h][0] / weights[h],
                                          loads[h][1] / weights[h], h))
        shard = shards[host]
        shard["files"].extend(group["files"])
        count = image_count(group)
        shard["image_count"] += count
        loads[host][0] += count
        loads[host][1] += sum(r["size"] for r in group["files"])
        for value in group["directories"] + [str(PurePosixPath(r["path"]).parent)
                                             for r in group["files"]]:
            current = PurePosixPath(value)
            shard["directories"].add(current.as_posix())
            shard["directories"].update(p.as_posix() for p in current.parents)
    for shard in shards.values():
        shard["directories"] = sorted(shard["directories"])
    paths = [r["path"] for shard in shards.values() for r in shard["files"]]
    if len(paths) != len(set(paths)) or set(paths) != {r["path"] for r in manifest["files"]}:
        raise ValueError("Partition lost or duplicated files")
    if set().union(*(set(s["directories"]) for s in shards.values())) != set(manifest["directories"]):
        raise ValueError("Partition changed directory inventory")
    return {"schema": 1, "shards": shards}



def sampled_manifest(manifest, count, seed):
    eligible = [row for row in manifest["files"] if row["size"] > 0
                and PurePosixPath(row["path"]).suffix.lower() in {".jpg", ".jpeg", ".png"}]
    chosen = random.Random(seed).sample(eligible, min(count, len(eligible)))
    directories = {"."}
    for row in chosen:
        directories.update(p.as_posix() for p in PurePosixPath(row["path"]).parents)
    return {"schema": 1, "files": chosen, "directories": sorted(directories)}


def read_token(path):
    value = path.read_text().strip()
    if len(value) < 32 or not value.isascii() or any(c.isspace() for c in value):
        raise ValueError("A nonempty random transfer token is required")
    return value


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_):
        raise ValueError("Transfer redirects are not permitted")


def serve(args):
    root = args.root.resolve()
    manifest, checksums = frozen_inventory(root, args.inventory)
    manifests = {"": manifest}
    if args.plan_file:
        plan = json.loads(args.plan_file.read_text())
        manifests.update(plan["shards"])
    encoded = {name: json.dumps(value, ensure_ascii=False).encode()
               for name, value in manifests.items()}
    allowed = set(args.allowed_peers)
    token = read_token(args.token_file)

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def do_GET(self):
            if (self.client_address[0] not in allowed or not secrets.compare_digest(
                    self.headers.get("Authorization", ""), "Bearer " + token)):
                self.send_error(403)
                return
            parsed = urllib.parse.urlsplit(self.path)
            path = parsed.path
            if path == "/manifest":
                shard = urllib.parse.parse_qs(parsed.query).get("shard", [""])[0]
                if shard not in encoded:
                    self.send_error(404)
                    return
                body = encoded[shard]
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
                return
            try:
                if not path.startswith("/file/"):
                    raise ValueError("Unknown endpoint")
                rel = relative_path(urllib.parse.unquote(path[6:], errors="strict"))
                reference = checksums[rel]
                p = beneath(root, rel)
                if p.is_symlink() or signature(p) != reference["signature"]:
                    raise ValueError("Source changed")
            except (ValueError, KeyError, OSError):
                self.send_error(404)
                return
            try:
                with p.open("rb") as source:
                    self.send_response(200)
                    self.send_header("Content-Length", str(p.stat().st_size))
                    self.send_header("Content-Type", "application/octet-stream")
                    self.end_headers()
                    for block in iter(lambda: source.read(2 * 1024 * 1024), b""):
                        self.wfile.write(block)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(args.certificate, args.private_key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    print(json.dumps({"state": "ready", "files": len(manifest["files"]),
                      "directories": len(manifest["directories"])}), flush=True)
    server.serve_forever()


def transfer_file(root, row, opener, base, token, temporary_root):
    rel = relative_path(row["path"])
    destination = beneath(root, rel)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError("Destination is a symbolic link")
    if (destination.is_file() and destination.stat().st_size == row["size"]
            and digest(destination) == row["md5"]):
        return "reused"
    temporary_root.mkdir(parents=True, exist_ok=True)
    temp = temporary_root / (hashlib.sha256(rel.encode()).hexdigest() + ".part")
    if temp.is_symlink():
        raise ValueError("Temporary destination is a symbolic link")
    for attempt in range(3):
        try:
            request = urllib.request.Request(base + "/file/" + urllib.parse.quote(rel, safe="/"),
                                             headers={"Authorization": "Bearer " + token})
            h = hashlib.md5()
            count = 0
            with opener.open(request, timeout=120) as response, temp.open("wb") as output:
                if int(response.headers.get("Content-Length", -1)) != row["size"]:
                    raise ValueError("Source size differs from inventory")
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block)
                    h.update(block)
                    count += len(block)
            if count != row["size"] or h.hexdigest() != row["md5"]:
                raise ValueError("Transferred file checksum mismatch")
            temp.replace(destination)
            return "copied"
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError("Transfer did not complete")


def fetch(args):
    if urllib.parse.urlsplit(args.url).scheme != "https":
        raise ValueError("A verified HTTPS source is required")
    context = ssl.create_default_context(cafile=str(args.certificate))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect(),
                                         urllib.request.HTTPSHandler(context=context))
    token = read_token(args.token_file)
    endpoint = args.url.rstrip("/") + "/manifest"
    if args.shard:
        endpoint += "?" + urllib.parse.urlencode({"shard": args.shard})
    req = urllib.request.Request(endpoint,
                                 headers={"Authorization": "Bearer " + token})
    with opener.open(req, timeout=60) as response:
        manifest = json.load(response)
    if args.sample:
        manifest = sampled_manifest(manifest, args.sample, args.seed)
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    files = manifest["files"]
    expected = {relative_path(r["path"]): r for r in files}
    if len(expected) != len(files):
        raise ValueError("Duplicate manifest path")
    directories = {relative_path(d, True) for d in manifest["directories"]}
    for directory in sorted(directories):
        if directory == ".":
            continue
        beneath(root, directory).mkdir(parents=True, exist_ok=True)
    write_json(args.state_dir / "manifest.json", manifest)
    completed, byte_count, failures = 0, 0, []
    began = last_report = time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(transfer_file, root, row, opener, args.url.rstrip("/"), token,
                            args.state_dir / "partial-files"): row
                for row in files}
        for job in cf.as_completed(jobs):
            row = jobs[job]
            try:
                job.result()
                completed += 1
                byte_count += row["size"]
            except Exception as error:  # noqa: BLE001 - keep every failed relative path
                failures.append({"file": row["path"], "error": type(error).__name__})
            if time.monotonic() - last_report >= 1:
                write_json(args.state_dir / "status.json", {"state": "transferring",
                           "complete": completed, "total": len(files), "bytes": byte_count,
                           "total_bytes": sum(r["size"] for r in files), "failures": len(failures)})
                last_report = time.monotonic()
    actual_files, actual_dirs = set(), set()
    for folder, _, names in os.walk(root):
        actual_dirs.add(Path(folder).relative_to(root).as_posix())
        actual_files.update((Path(folder) / n).relative_to(root).as_posix() for n in names)
    report = {"state": "completed" if not failures and actual_files == set(expected)
              and actual_dirs == directories else "failed", "files": len(actual_files),
              "directories_including_root": len(actual_dirs), "verified_files": completed,
              "bytes": byte_count, "failures": failures,
              "missing_files": sorted(set(expected) - actual_files),
              "extra_files": sorted(actual_files - set(expected)),
              "missing_directories": sorted(directories - actual_dirs),
              "extra_directories": sorted(actual_dirs - directories),
              "seconds": time.monotonic() - began}
    write_json(args.state_dir / "report.json", report)
    write_json(args.state_dir / "status.json", report)
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, list)}), flush=True)
    if report["state"] != "completed":
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(__doc__)
    commands = p.add_subparsers(dest="mode", required=True)
    server = commands.add_parser("serve")
    server.add_argument("--root", type=Path, required=True)
    server.add_argument("--inventory", type=Path, required=True)
    server.add_argument("--host", required=True)
    server.add_argument("--port", type=int, required=True)
    server.add_argument("--allowed-peers", nargs="+", required=True)
    server.add_argument("--token-file", type=Path, required=True)
    server.add_argument("--certificate", type=Path, required=True)
    server.add_argument("--private-key", type=Path, required=True)
    server.add_argument("--plan-file", type=Path)
    client = commands.add_parser("fetch")
    client.add_argument("--url", required=True)
    client.add_argument("--shard")
    client.add_argument("--sample", type=int, default=0)
    client.add_argument("--seed", type=int, default=42)
    client.add_argument("--destination", type=Path, required=True)
    client.add_argument("--state-dir", type=Path, required=True)
    client.add_argument("--workers", type=int, choices=range(1, 33), default=8)
    client.add_argument("--token-file", type=Path, required=True)
    client.add_argument("--certificate", type=Path, required=True)
    planner = commands.add_parser("plan")
    planner.add_argument("--root", type=Path, required=True)
    planner.add_argument("--inventory", type=Path, required=True)
    planner.add_argument("--weights", required=True, help="JSON host to free GPU count")
    planner.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.mode == "plan":
        manifest, _ = frozen_inventory(args.root, args.inventory)
        plan = partition_manifest(manifest, json.loads(args.weights))
        write_json(args.output, plan)
        print(json.dumps({h: {"files": len(v["files"]), "images": v["image_count"],
                              "directories": len(v["directories"])}
                          for h, v in plan["shards"].items()}))
    else:
        (serve if args.mode == "serve" else fetch)(args)


if __name__ == "__main__":
    main()
