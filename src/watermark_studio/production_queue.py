"""Durable exclusive directory leases for the two production machines."""
from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from pathlib import PurePosixPath

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def directory_units(manifest, target=256):
    images = [r for r in manifest["files"] if r["size"] > 0
              and PurePosixPath(r["path"]).suffix.lower() in IMAGE_SUFFIXES]
    units = []

    def visit(prefix, rows):
        direct, children = [], {}
        depth = len(PurePosixPath(prefix).parts)
        for row in rows:
            parts = PurePosixPath(row["path"]).parts
            if len(parts) == depth + 1:
                direct.append(row)
            else:
                child = "/".join(parts[:depth + 1])
                children.setdefault(child, []).append(row)
        if len(rows) <= target or not children:
            add(prefix, rows, "subtree")
            return
        if direct:
            add(prefix, direct, "direct")
        for child, selected in sorted(children.items()):
            visit(child, selected)

    def add(prefix, rows, mode):
        token = hashlib.sha256((prefix + ":" + mode).encode()).hexdigest()
        units.append({"id": token, "prefix": prefix, "mode": mode, "files": rows,
                      "images": len(rows)})

    if images:
        visit(".", images)
    units.sort(key=lambda u: (-u["images"], u["prefix"], u["mode"]))
    paths = [r["path"] for unit in units for r in unit["files"]]
    if len(paths) != len(set(paths)) or set(paths) != {r["path"] for r in images}:
        raise ValueError("Directory work plan lost or duplicated images")
    return units


class WorkBroker:
    def __init__(self, manifest, state_dir, nodes, clock=time.time, lease_seconds=1800):
        self.manifest = manifest
        self.root = state_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o700)
        self.nodes = set(nodes)
        self.clock = clock
        self.lease_seconds = lease_seconds
        self.lock = threading.RLock()
        self.units = directory_units(manifest)
        self.by_id = {u["id"]: u for u in self.units}
        self.identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        path = self.root / "leases.json"
        if path.exists():
            self.state = json.loads(path.read_text())
            if self.state["manifest_sha256"] != self.identity:
                raise ValueError("Work queue belongs to a different input snapshot")
        else:
            self.state = {"manifest_sha256": self.identity, "leases": {}, "workers": {}}
            self.save()
        self._write("original-manifest.json", manifest)

    def _write(self, name, value):
        p = self.root / name
        temporary = p.with_name(p.name + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
        temporary.replace(p)

    def save(self):
        self._write("leases.json", self.state)

    def claim(self, node):
        with self.lock:
            if node not in self.nodes:
                raise ValueError("Machine is outside the authorized pool")
            now = self.clock()
            leases = self.state["leases"]
            for unit in self.units:
                current = leases.get(unit["id"])
                if current and current["state"] == "done":
                    continue
                if current and now - current["heartbeat"] < self.lease_seconds:
                    continue
                lease = {"node": node, "lease": secrets.token_hex(16),
                         "state": "running", "heartbeat": now}
                leases[unit["id"]] = lease
                self.save()
                return {"unit": unit, "lease": lease["lease"]}
            return {"finished": all(leases.get(u["id"], {}).get("state") == "done"
                                    for u in self.units), "wait": True}

    def owned(self, node, unit_id, token):
        lease = self.state["leases"].get(unit_id)
        if (node not in self.nodes or not lease or lease["node"] != node
                or not secrets.compare_digest(lease["lease"], token)):
            raise ValueError("Stale or foreign directory lease")
        return lease

    def heartbeat(self, node, unit_id, token, progress=None):
        with self.lock:
            lease = self.owned(node, unit_id, token)
            lease["heartbeat"] = self.clock()
            if progress:
                lease["progress"] = progress
            self.save()
            return {"state": lease["state"]}

    def complete(self, node, unit_id, token, rows):
        with self.lock:
            lease = self.owned(node, unit_id, token)
            expected = {r["path"] for r in self.by_id[unit_id]["files"]}
            paths = [r["file"] for r in rows]
            if len(paths) != len(set(paths)) or set(paths) != expected:
                raise ValueError("Directory completion does not cover its exact input files")
            for row in rows:
                sha = row.get("sha256", "")
                if (row.get("quality_status") not in {"passed", "review", "error", "processing_failed"}
                        or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)
                        or row.get("size", 0) <= 0):
                    raise ValueError("Missing output or quality evidence")
            self._write(unit_id + ".json", {"node": node, "unit": unit_id, "rows": rows})
            lease.update(state="done", heartbeat=self.clock(), completed=len(rows),
                         passed=sum(r["quality_status"] == "passed" for r in rows))
            self.save()
            return {"state": "done"}

    def worker(self, node, info):
        with self.lock:
            if node not in self.nodes:
                raise ValueError("Machine is outside the authorized pool")
            self.state["workers"][node] = info | {"updated_at": self.clock()}
            self.save()
            return {"state": "recorded"}

    def status(self):
        with self.lock:
            done = [v for v in self.state["leases"].values() if v["state"] == "done"]
            return {"state": "completed" if len(done) == len(self.units) else "running",
                    "directories_completed": len(done), "directories_total": len(self.units),
                    "images_completed": sum(v.get("completed", 0) for v in done),
                    "images_total": sum(u["images"] for u in self.units),
                    "passed": sum(v.get("passed", 0) for v in done),
                    "workers": self.state["workers"]}

    def ownership(self):
        with self.lock:
            if self.status()["state"] != "completed":
                raise ValueError("Production processing is not finished")
            rows = {}
            for unit in self.units:
                lease = self.state["leases"][unit["id"]]
                rows[unit["id"]] = {"node": lease["node"], "prefix": unit["prefix"],
                                   "files": [r["path"] for r in unit["files"]]}
            return {"manifest_sha256": self.identity, "units": rows,
                    "directories": self.manifest["directories"],
                    "auxiliary": [r for r in self.manifest["files"] if not (
                        r["size"] > 0 and PurePosixPath(r["path"]).suffix.lower() in IMAGE_SUFFIXES)]}
