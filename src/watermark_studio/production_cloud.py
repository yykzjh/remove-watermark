"""Deliver immutable result artifacts through the owner's Baidu account."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

from .dataset_transfer import write_json
from .production_archives import file_sha256


def cloud_path(value):
    p = PurePosixPath(value)
    if not p.is_absolute() or ".." in p.parts:
        raise ValueError("Cloud destination must be an absolute confined path")
    return p.as_posix()


class BaiduDelivery:
    def __init__(self, config, state, binary=None, upload_parallel=4):
        if type(upload_parallel) is not int or not 1 <= upload_parallel <= 64:
            raise ValueError("Upload parallelism must be an integer from 1 to 64")
        self.upload_parallel = upload_parallel
        self.config = Path(config)
        self.state = Path(state)
        self.state.mkdir(parents=True, exist_ok=True)
        self.state.chmod(0o700)
        self.binary = str(binary or Path.home() / ".local/bin/BaiduPCS-Go")
        self.env = dict(os.environ, BAIDUPCS_GO_CONFIG_DIR=str(self.config))
        if not (self.config / "pcs_config.json").is_file():
            raise ValueError("Missing task-scoped Baidu login")
        self.cookie = (self.config / "baidu.cookie").read_text().strip()

    def command(self, args, label):
        # CLI output may contain account identifiers or download URLs: keep it private.
        log = self.state / (label + ".log")
        with log.open("ab") as output:
            log.chmod(0o600)
            p = subprocess.run([self.binary, *args], check=False, env=self.env, stdin=subprocess.DEVNULL,
                               stdout=output, stderr=subprocess.STDOUT)
        return p.returncode

    def listing(self, directory):
        found = []
        for page in range(1, 10000):
            query = urllib.parse.urlencode({"dir": cloud_path(directory), "num": 1000,
                                                 "page": page, "order": "name", "desc": 0, "web": 1})
            for attempt in range(6):
                try:
                    req = urllib.request.Request("https://pan.baidu.com/api/list?" + query,
                        headers={"Cookie": self.cookie, "User-Agent": "Mozilla/5.0",
                                 "Referer": "https://pan.baidu.com/disk/main"})
                    with urllib.request.urlopen(req, timeout=45) as response:
                        data = json.load(response)
                    if data.get("errno") != 0:
                        raise RuntimeError("Baidu listing failed, errno=" + str(data.get("errno")))
                    break
                except (OSError, ValueError, RuntimeError):
                    if attempt == 5:
                        raise
                    time.sleep(2 ** attempt)
            found.extend(data["list"])
            if len(data["list"]) < 1000:
                return {r["server_filename"]: r for r in found}
        raise RuntimeError("Baidu listing pagination exceeded")

    def upload(self, source, directory, expected_sha):
        source = Path(source)
        if file_sha256(source) != expected_sha:
            raise ValueError("Artifact changed before upload")
        directory = cloud_path(directory)
        for attempt in range(4):
            remote = self.listing(directory).get(source.name)
            if remote and not remote["isdir"] and remote["size"] == source.stat().st_size:
                return
            self.command(["upload", "-p", str(self.upload_parallel), "-l", "1", "--retry", "3",
                          "--policy", "overwrite", str(source), directory], "upload-" + source.name)
            remote = self.listing(directory).get(source.name)
            if remote and not remote["isdir"] and remote["size"] == source.stat().st_size:
                return
            time.sleep(2 ** attempt)
        raise RuntimeError("Artifact not present at expected cloud size: " + source.name)

    def download(self, directory, artifact, destination):
        name = artifact["name"]
        if PurePosixPath(name).name != name or name in {"", ".", ".."}:
            raise ValueError("Unsafe artifact name")
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / name
        if (target.is_file() and target.stat().st_size == artifact["size"]
                and file_sha256(target) == artifact["sha256"]):
            return target
        partial = self.state / "downloads" / name
        partial.mkdir(parents=True, exist_ok=True)
        for attempt in range(4):
            self.command(["download", "--saveto", str(partial), "-p", "8", "-l", "1",
                          "--retry", "3", cloud_path(directory).rstrip("/") + "/" + name],
                         "download-" + name)
            source = partial / name
            marker = Path(str(source) + ".BaiduPCS-Go-downloading")
            if (source.is_file() and not marker.exists()
                    and source.stat().st_size == artifact["size"]
                    and file_sha256(source) == artifact["sha256"]):
                source.replace(target)
                return target
            # A failed complete hash must be downloaded again, not silently skipped.
            if source.is_file() and not marker.exists():
                source.unlink()
            time.sleep(2 ** attempt)
        raise RuntimeError("Artifact download failed size/SHA256 verification: " + name)


def public_index(index):
    public = {key: index[key] for key in ["schema", "node", "manifest_sha256",
                                         "files", "image_files", "auxiliary_files"]}
    public["artifacts"] = [{k: a[k] for k in ["name", "size", "sha256", "kind"]}
                           for a in index["artifacts"]]
    return public


def publish_preview(index_path, cloud_root, config, state, binary=None):
    """Publish only the sealed artifact table; this is never a completion receipt."""
    state = Path(state)
    index = json.loads(Path(index_path).read_text())
    client = BaiduDelivery(config, state, binary)
    client.command(["mkdir", cloud_path(cloud_root)], "mkdir")
    client.listing(cloud_root)
    temporary = state / "public-index.json"
    write_json(temporary, public_index(index))
    sha = file_sha256(temporary)
    final = state / (index["node"] + "-preview-index-" + sha[:16] + ".json")
    temporary.replace(final)
    client.upload(final, cloud_root, sha)
    return {"state": "publishing", "node": index["node"], "cloud_root": cloud_path(cloud_root),
            "manifest_sha256": index["manifest_sha256"],
            "index": {"name": final.name, "size": final.stat().st_size, "sha256": sha}}


def delivery_identity(delivery):
    # Preview and final metadata use separate names but must contain identical bytes.
    return {k: delivery[k] for k in ["node", "cloud_root", "manifest_sha256"]} | {
        "index": {k: delivery["index"][k] for k in ["size", "sha256"]}}


def retrieve_available(delivery, available, config, state, destination, previous=None, binary=None):
    """One owner receives only a checksum-pinned uploaded prefix; do not assemble it."""
    if delivery["state"] not in {"publishing", "published"}:
        raise ValueError("Invalid streaming delivery state")
    client = BaiduDelivery(config, state, binary)
    identity = delivery_identity(delivery)
    if previous is None:
        path = client.download(delivery["cloud_root"], delivery["index"], destination)
        index = json.loads(path.read_text())
        if index["node"] != delivery["node"] or index["manifest_sha256"] != delivery["manifest_sha256"]:
            raise ValueError("Cloud index identity differs from trusted delivery receipt")
        names = [a["name"] for a in index["artifacts"]]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate streaming artifact name")
        progress = {"identity": identity, "index_path": str(path),
                    "artifacts": index["artifacts"], "verified": 0}
    else:
        if previous["identity"] != identity:
            raise ValueError("Streaming delivery receipt changed")
        progress = previous
    total = len(progress["artifacts"])
    if available is None:
        if delivery["state"] != "published":
            raise ValueError("Unpublished delivery requires a bounded available count")
        available = total
    if type(available) is not int or not 0 <= available <= total:
        raise ValueError("Invalid uploaded artifact prefix")
    for position in range(progress["verified"], available):
        client.download(delivery["cloud_root"], progress["artifacts"][position], destination)
        progress["verified"] = position + 1
        write_json(Path(state) / "status.json", {"state": "downloading", "node": delivery["node"],
                   "complete": progress["verified"], "available": available, "total": total})
    if delivery["state"] == "published":
        # Revalidate the tiny pinned index before final assembly, even after a long prefetch.
        path = client.download(delivery["cloud_root"], delivery["index"], destination)
        progress["index_path"] = str(path)
    return progress


def publish(index_path, cloud_root, config, state, binary=None, upload_parallel=4):
    index_path, state = Path(index_path), Path(state)
    index = json.loads(index_path.read_text())
    client = BaiduDelivery(config, state, binary, upload_parallel=upload_parallel)
    client.command(["mkdir", cloud_path(cloud_root)], "mkdir")
    client.listing(cloud_root)  # exit status alone never proves directory creation.
    public = public_index(index)
    for number, artifact in enumerate(index["artifacts"], 1):
        write_json(state / "status.json", {"state": "uploading", "complete": number - 1,
                                         "total": len(index["artifacts"]), "node": index["node"]})
        client.upload(artifact["local_path"], cloud_root, artifact["sha256"])
    temporary = state / "public-index.json"
    write_json(temporary, public)
    sha = file_sha256(temporary)
    final = state / (index["node"] + "-index-" + sha[:16] + ".json")
    temporary.replace(final)
    client.upload(final, cloud_root, sha)
    delivery = {"state": "published", "node": index["node"], "cloud_root": cloud_path(cloud_root),
                "manifest_sha256": index["manifest_sha256"],
                "index": {"name": final.name, "size": final.stat().st_size, "sha256": sha}}
    write_json(state / "delivery.json", delivery)
    write_json(state / "status.json", delivery)
    return delivery


def retrieve(deliveries, config, state, destination, binary=None):
    client = BaiduDelivery(config, state, binary)
    indexes = []
    for delivery in deliveries:
        index_path = client.download(delivery["cloud_root"], delivery["index"], destination)
        index = json.loads(index_path.read_text())
        if index["node"] != delivery["node"] or index["manifest_sha256"] != delivery["manifest_sha256"]:
            raise ValueError("Cloud index identity differs from trusted delivery receipt")
        for number, artifact in enumerate(index["artifacts"], 1):
            client.download(delivery["cloud_root"], artifact, destination)
            write_json(Path(state) / "status.json", {"state": "downloading", "node": index["node"],
                                                   "complete": number, "total": len(index["artifacts"])})
        indexes.append(str(index_path))
    write_json(Path(state) / "status.json", {"state": "downloaded", "indexes": indexes})
    return indexes


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("mode", choices=["publish", "retrieve"])
    p.add_argument("--config-dir", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--index", type=Path)
    p.add_argument("--cloud-root")
    p.add_argument("--deliveries", type=Path)
    p.add_argument("--destination", type=Path)
    a = p.parse_args()
    if a.mode == "publish":
        if not a.index or not a.cloud_root:
            p.error("publish requires --index and --cloud-root")
        print(json.dumps(publish(a.index, a.cloud_root, a.config_dir, a.state)))
    else:
        if not a.deliveries or not a.destination:
            p.error("retrieve requires --deliveries and --destination")
        print(json.dumps(retrieve(json.loads(a.deliveries.read_text()), a.config_dir,
                                  a.state, a.destination)))


if __name__ == "__main__":
    main()
