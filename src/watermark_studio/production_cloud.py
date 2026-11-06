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
    def __init__(self, config, state, binary=None):
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
            self.command(["upload", "-p", "4", "-l", "1", "--retry", "3",
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


def publish(index_path, cloud_root, config, state, binary=None):
    index_path, state = Path(index_path), Path(state)
    index = json.loads(index_path.read_text())
    client = BaiduDelivery(config, state, binary)
    client.command(["mkdir", cloud_path(cloud_root)], "mkdir")
    client.listing(cloud_root)  # exit status alone never proves directory creation.
    public = {key: index[key] for key in ["schema", "node", "manifest_sha256",
                                         "files", "image_files", "auxiliary_files"]}
    public["artifacts"] = [{k: a[k] for k in ["name", "size", "sha256", "kind"]}
                           for a in index["artifacts"]]
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
