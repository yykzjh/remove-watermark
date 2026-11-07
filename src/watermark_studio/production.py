"""Process exclusively leased directories; expand GPUs only at directory boundaries."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import io
import json
import os
import signal
import ssl
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

from .dataset_transfer import NoRedirect, beneath, digest, read_token, transfer_file, write_json
from .production_queue import IMAGE_SUFFIXES


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def process_start(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def group_members(group):
    members = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            if os.getpgid(int(p.name)) == group and p.joinpath("cmdline").read_bytes():
                if p.stat().st_uid != os.getuid():
                    raise RuntimeError("Service group contains a foreign owner")
                members.append(int(p.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return members


def stop_service(plan):
    if not plan:
        return
    pid = plan["pid"]
    p = Path(f"/proc/{pid}")
    if p.exists() and p.joinpath("cmdline").read_bytes() and process_start(pid) != plan["start"]:
        raise RuntimeError("Service PID was reused; refusing to stop another process")
    if not group_members(pid):
        return
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + 20
    while group_members(pid) and time.monotonic() < deadline:
        time.sleep(.2)
    if group_members(pid):
        os.killpg(pid, signal.SIGKILL)
        time.sleep(1)
    if group_members(pid):
        raise RuntimeError("Previous service did not stop")


def gpu_inventory(service_group=None):
    def query(kind, fields):
        text = subprocess.check_output(
            ["nvidia-smi", "--query-" + kind + "=" + fields,
             "--format=csv,noheader,nounits"], text=True, timeout=15)
        return list(csv.reader(io.StringIO(text)))
    owned = set(group_members(service_group)) if service_group else set()
    own_memory = {}
    for uuid, pid, used in query("compute-apps", "gpu_uuid,pid,used_memory"):
        if int(pid.strip()) in owned and used.strip().isdigit():
            own_memory[uuid.strip()] = own_memory.get(uuid.strip(), 0) + int(used.strip())
    rows = []
    for uuid, index, total, used in query("gpu", "uuid,index,memory.total,memory.used"):
        own = own_memory.get(uuid.strip(), 0)
        rows.append({"index": int(index), "total": int(total), "used": int(used),
                     "own_used": own, "available": int(total) - max(0, int(used) - own)})
    return rows


def eligible_gpus(rows, required, allowed):
    return sorted(r["index"] for r in rows if r["index"] in allowed and r["available"] >= required)


class DeploymentCapacityError(RuntimeError):
    """Another workload claimed GPU memory while our models were loading."""


class ProductionRunner:
    def __init__(self, config):
        self.c = config
        self.node = config["node"]
        self.repo = Path(config["repo"])
        self.source = Path(config["input_root"]).resolve()
        self.output = Path(config["output_root"]).resolve()
        self.state = Path(config["state_dir"])
        self.state.mkdir(parents=True, exist_ok=True)
        self.state.chmod(0o700)
        self.output.mkdir(parents=True, exist_ok=True)
        self.source.mkdir(parents=True, exist_ok=True)
        cfg = Path(config["transfer_config"])
        self.token = read_token(cfg / "token")
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), NoRedirect(),
            urllib.request.HTTPSHandler(context=ssl.create_default_context(
                cafile=str(cfg / "certificate.pem"))))
        self.service = None
        control = self.state / "service-process.json"
        if control.exists():
            self.service = json.loads(control.read_text())
        self.gpus = list(self.service.get("gpus", [])) if self.service else []
        self.current = None
        current = self.state / "current-directory.json"
        if current.exists():
            saved = json.loads(current.read_text())
            if saved.get("state") == "running":
                self.current = saved
        self.last_heartbeat = 0

    def broker(self, path, payload=None):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
        request = urllib.request.Request(self.c["broker_url"].rstrip("/") + path, data=data,
            headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"})
        with self.opener.open(request, timeout=60) as response:
            return json.load(response)

    def api(self, path, payload=None):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
        request = urllib.request.Request(self.c["api_url"].rstrip("/") + path, data=data,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    def status(self, phase, **extra):
        value = {"node": self.node, "phase": phase, "gpus": self.gpus, **extra}
        write_json(self.state / "status.json", value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def heartbeat(self, progress=None):
        if not self.current or time.monotonic() - self.last_heartbeat < 30:
            return
        try:
            self.broker("/broker/heartbeat", {"node": self.node,
                        "unit_id": self.current["unit"]["id"], "lease": self.current["lease"],
                        "progress": progress or {}})
        except urllib.error.HTTPError as error:
            if error.code != 409 or (progress or {}).get("phase") not in {
                "waiting_for_gpu", "deploying",
            }:
                raise
            # No image work is in flight during resource waits/deployment.
            # A recovered controller must not revive an expired directory lease.
            write_json(self.state / "current-directory.json",
                       self.current | {"state": "lease_lost"})
            self.current = None
        self.last_heartbeat = time.monotonic()

    def resources(self):
        rows = gpu_inventory(self.service["pid"] if self.service else None)
        selected = eligible_gpus(rows, self.c["required_free_mib"], self.c.get("allowed_gpus", list(range(8))))
        with (self.state / "gpu-checks.jsonl").open("a") as f:
            f.write(json.dumps({"time": time.time(), "previous": self.gpus,
                               "eligible": selected, "required_free_mib": self.c["required_free_mib"],
                               "rows": rows}) + "\n")
        return selected

    def deploy(self, selected):
        if self.service:
            stop_service(self.service)
        self.service = None
        self.gpus = selected
        command = [str(arg).replace("{gpus}", ",".join(map(str, selected)))
                   .replace("{gpu_count}", str(len(selected))) for arg in self.c["service_command"]]
        log = (self.state / "service.log").open("a")
        env = dict(os.environ, UV_PROJECT_ENVIRONMENT=self.c["environment"], PYTHONUNBUFFERED="1")
        child = subprocess.Popen(command, cwd=self.repo, env=env, stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.service = {"pid": child.pid, "start": process_start(child.pid), "gpus": selected,
                        "command": command, "started_at": time.time()}
        write_json(self.state / "service-process.json", self.service)
        self.status("deploying")
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            self.heartbeat({"phase": "deploying"})
            if child.poll() is not None:
                raise RuntimeError("Model service exited during deployment")
            try:
                health = self.api("/health")
                if health["stage"] == "failed":
                    error = health.get("error", "")
                    if "cuda out of memory" in error.lower():
                        raise DeploymentCapacityError("Model deployment failed: " + error)
                    raise RuntimeError("Model deployment failed: " + error)
                if health["stage"] == "ready":
                    if health.get("quality_weight_identity", {}).get("relation") == "same_weights":
                        raise RuntimeError("Editor and verifier must use different weights")
                    self.broker("/broker/worker", {"node": self.node,
                                "info": {"gpus": selected, "phase": "ready"}})
                    self.status("ready")
                    return
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(2)
        raise TimeoutError("Model deployment timed out")

    def resumable_directory(self):
        current = getattr(self, "current", None)
        if not current or not current.get("unit") or not self.service:
            return False
        if not group_members(self.service["pid"]):
            return False
        health = self.api("/health")
        job = health.get("job") or {}
        if health.get("stage") != "ready" or job.get("state") not in {
            "scanning", "running", "paused", "cancelling",
        }:
            return False
        unit = current["unit"]
        input_dir = (self.state / "direct-input" / unit["id"]
                     if unit["mode"] == "direct" else self.source / unit["prefix"])
        options = dict(self.c["options"],
                       concurrency=min(4 * len(self.gpus), self.c.get("max_concurrency", 64)))
        if (job.get("input_dir") != str(input_dir)
                or job.get("output_dir") != str(self.output / unit["prefix"])
                or any(job.get("options", {}).get(k) != v for k, v in options.items())):
            raise RuntimeError("Unrecognized active service job; refusing resource redeployment")
        return True

    def ensure_service(self):
        # Controller recovery must reattach an owned active directory, not restart
        # its healthy model workers merely because another GPU became available.
        if self.resumable_directory():
            return
        while True:
            selected = self.resources()
            if not selected:
                if self.service:
                    stop_service(self.service)
                    self.service = None
                    self.gpus = []
                if not self.current and self.broker("/broker/status")["state"] == "completed":
                    return
                self.status("waiting_for_gpu")
                self.heartbeat({"phase": "waiting_for_gpu"})
                time.sleep(30)
                continue
            alive = self.service and bool(group_members(self.service["pid"]))
            if selected == self.gpus and alive:
                return
            try:
                self.deploy(selected)
                return
            except DeploymentCapacityError as error:
                # Eligibility is a snapshot, not a reservation. Stop only our
                # verified process group, then resample without disturbing peers.
                stop_service(self.service)
                self.service = None
                self.gpus = []
                self.status("waiting_for_gpu", reason="deployment_capacity_changed",
                            message=str(error)[:400])
                self.heartbeat({"phase": "waiting_for_gpu"})
                time.sleep(30)

    def preserve_auxiliary(self, manifest):
        if not self.c.get("preserve_auxiliary", False):
            return
        from .images import atomic_copy
        for directory in manifest["directories"]:
            if directory != ".":
                beneath(self.output, directory).mkdir(parents=True, exist_ok=True)
        invalid = []
        for row in manifest["files"]:
            if row["size"] > 0 and PurePosixPath(row["path"]).suffix.lower() in IMAGE_SUFFIXES:
                continue
            source, destination = beneath(self.source, row["path"]), beneath(self.output, row["path"])
            if not (destination.is_file() and destination.stat().st_size == row["size"]
                    and digest(destination) == row["md5"]):
                atomic_copy(source, destination, True, preserve_metadata=True)
            if digest(destination) != row["md5"]:
                raise ValueError("Auxiliary file differs from input: " + row["path"])
            if PurePosixPath(row["path"]).suffix.lower() in IMAGE_SUFFIXES:
                invalid.append({"file": row["path"], "reason": "empty_original_file"})
        write_json(self.state / "source-invalid.json", invalid)

    def prepare(self, unit):
        if not self.c.get("source_is_complete", False):
            with cf.ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(transfer_file, self.source, row, self.opener,
                           self.c["broker_url"], self.token, self.state / "input-partials")
                           for row in unit["files"]]
                while futures:
                    _, pending = cf.wait(futures, timeout=5, return_when=cf.FIRST_COMPLETED)
                    for future in set(futures) - pending:
                        future.result()
                    futures = list(pending)
                    self.heartbeat({"phase": "input_transfer"})
        if unit["mode"] != "direct":
            return self.source / unit["prefix"]
        from .images import atomic_copy
        isolated = self.state / "direct-input" / unit["id"]
        isolated.mkdir(parents=True, exist_ok=True)
        for row in unit["files"]:
            atomic_copy(beneath(self.source, row["path"]),
                        isolated / PurePosixPath(row["path"]).name, True, preserve_metadata=True)
        return isolated

    def run_directory(self, unit):
        input_dir = self.prepare(unit)
        destination = self.output / unit["prefix"]
        options = dict(self.c["options"])
        options["concurrency"] = min(4 * len(self.gpus), self.c.get("max_concurrency", 64))
        existing = self.api("/health").get("job")
        compatible = (existing and existing.get("state") not in {"failed", "interrupted", "cancelled"}
                      and existing.get("input_dir") == str(input_dir)
                      and existing.get("output_dir") == str(destination)
                      and all(existing.get("options", {}).get(k) == v
                              for k, v in options.items()))
        if compatible:
            job_id = existing["id"]
        else:
            if existing and existing.get("state") in {"scanning", "running", "paused", "cancelling"}:
                raise RuntimeError("Unrecognized active service job; refusing duplicate submission")
            submitted = self.api("/jobs", {"input_dir": str(input_dir),
                             "output_dir": str(destination), "options": options})
            job_id = submitted.get("id") or submitted.get("job", {}).get("id")
        self.current["job"] = job_id
        write_json(self.state / "current-directory.json", self.current)
        while True:
            health = self.api("/health")
            job = health.get("job")
            if not job or (job_id and job["id"] != job_id):
                raise RuntimeError("Service job identity changed")
            job_id = job["id"]
            self.heartbeat({"job": job_id, "completed": job["completed"], "total": job["total"]})
            self.status("processing", directory=unit["prefix"], job=job_id,
                        complete=job["completed"], total=job["total"])
            if job["state"] in {"completed", "failed", "interrupted", "cancelled"}:
                break
            time.sleep(5)
        if job["state"] != "completed":
            raise RuntimeError("Directory inference did not complete; retain lease for recovery")
        report_root = Path(self.c["server_state_dir"]) / job_id
        records = {}
        results = report_root / "results.jsonl"
        if results.exists():
            for line in results.read_text().splitlines():
                result = json.loads(line)
                path = (PurePosixPath(unit["prefix"]) / result["file"]).as_posix()
                records[path] = result | {"file": path}
        capacity_log = self.state / (unit["id"] + "-capacity-replays.jsonl")
        exhausted = [r for r in records.values()
                     if not r.get("ok") and "cuda out of memory" in r.get("error", "").lower()]
        if exhausted:
            with capacity_log.open("a") as stream:
                stream.write(json.dumps({"job": job_id, "time": time.time(),
                             "reason": "cuda_out_of_memory",
                             "failed": [{"file": r["file"], "error": r.get("error", "")[:1200]}
                                        for r in exhausted]}, ensure_ascii=False) + "\n")
            raise DeploymentCapacityError(
                "Directory contains CUDA memory errors; retain lease and reassess GPU capacity")
        rows = [self.validate_output(row, records.get(row["path"]), job_id) for row in unit["files"]]
        if capacity_log.exists():
            history = [json.loads(line) for line in capacity_log.read_text().splitlines()]
            for row in rows:
                row["capacity_replays"] = [
                    {"job": attempt["job"], "reason": attempt["reason"],
                     "failed_this_file": any(r["file"] == row["file"] for r in attempt["failed"])}
                    for attempt in history]
        write_json(self.state / (unit["id"] + "-results.json"),
                   {"unit": unit["id"], "job": job_id, "rows": rows})
        return rows

    def validate_output(self, row, result, job_id):
        from PIL import Image

        from .images import atomic_copy
        path = row["path"]
        source, output = beneath(self.source, path), beneath(self.output, path)
        result = dict(result or {"file": path, "ok": False, "error": "No processing result recorded"})
        original_size = None
        try:
            with Image.open(source) as image:
                original_size = image.size
            with Image.open(output) as image:
                image.load()
                if image.size != original_size:
                    raise ValueError("Output dimensions differ from input")
        except (OSError, ValueError):
            atomic_copy(source, output, True, preserve_metadata=True)
            result.update(ok=False, fallback_original=True,
                          error=result.get("error") or "No valid original-size output")
        quality = result.get("quality", {})
        status = quality.get("status", "error") if result.get("ok") else "processing_failed"
        if status not in {"passed", "review", "error", "processing_failed"}:
            status = "error"
        result.update(file=path, job=job_id, quality_status=status, size=output.stat().st_size,
                      sha256=sha256(output), node=self.node,
                      original_size=list(original_size) if original_size else None)
        return result

    def publish_local_reports(self):
        records = {}
        for receipt in self.state.glob("*-committed.json"):
            unit_id = receipt.name.removesuffix("-committed.json")
            p = self.state / (unit_id + "-results.json")
            if p.exists():
                for row in json.loads(p.read_text())["rows"]:
                    records[row["file"]] = row
        issues = [r for _, r in sorted(records.items()) if r["quality_status"] != "passed"]
        (self.state / "needs-review-relative-paths.txt").write_text(
            "".join(r["file"] + "\n" for r in issues))
        (self.state / "needs-review.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in issues))
        write_json(self.state / "quality-summary.json",
                   {"images": len(records), "passed": len(records) - len(issues),
                    "needs_review": len(issues)})

    def run(self):
        manifest = self.broker("/manifest")
        self.preserve_auxiliary(manifest)
        if self.broker("/broker/status")["state"] == "completed":
            stop_service(self.service)
            self.service = None
            self.gpus = []
            self.publish_local_reports()
            self.status("processing_completed")
            return
        self.ensure_service()
        while True:
            if self.current:
                try:
                    response = self.broker("/broker/heartbeat", {"node": self.node,
                        "unit_id": self.current["unit"]["id"], "lease": self.current["lease"]})
                    if response["state"] == "done":
                        write_json(self.state / (self.current["unit"]["id"] + "-committed.json"),
                                   {"node": self.node, "lease": self.current["lease"]})
                        self.current = None
                except urllib.error.HTTPError as error:
                    if error.code != 409:
                        raise
                    self.current = None
            if not self.current:
                claim = self.broker("/broker/claim", {"node": self.node})
                if claim.get("finished"):
                    break
                if "unit" not in claim:
                    self.status("waiting_for_remaining_directories")
                    time.sleep(10)
                    continue
                self.current = claim | {"state": "running"}
                write_json(self.state / "current-directory.json", self.current)
            self.ensure_service()
            rows = self.run_directory(self.current["unit"])
            compact = [{k: row[k] for k in ("file", "quality_status", "sha256", "size")}
                       for row in rows]
            self.broker("/broker/complete", {"node": self.node,
                        "unit_id": self.current["unit"]["id"], "lease": self.current["lease"],
                        "rows": compact})
            write_json(self.state / (self.current["unit"]["id"] + "-committed.json"),
                       {"node": self.node, "lease": self.current["lease"]})
            self.current["state"] = "done"
            write_json(self.state / "current-directory.json", self.current)
            self.publish_local_reports()
            self.resources()  # Required GPU availability check after every completed directory.
            self.current = None
        stop_service(self.service)
        self.service = None
        self.gpus = []
        self.publish_local_reports()
        self.status("processing_completed")



def monitor_memory(group, output):
    peaks = {}
    while group_members(group):
        try:
            rows = gpu_inventory(group)
            for row in rows:
                key = str(row["index"])
                peaks[key] = max(peaks.get(key, 0), row["own_used"])
            write_json(output, {"state": "sampling", "peak_process_mib": peaks,
                                "latest": rows, "time": time.time()})
        except (OSError, subprocess.SubprocessError):
            pass
        time.sleep(1)
    write_json(output, {"state": "service_stopped", "peak_process_mib": peaks,
                        "time": time.time()})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--memory-process", type=int, default=0)
    parser.add_argument("--memory-output", type=Path)
    args = parser.parse_args()
    if args.memory_process:
        if not args.memory_output:
            parser.error("--memory-output is required for memory sampling")
        monitor_memory(args.memory_process, args.memory_output)
        return
    if not args.config:
        parser.error("--config is required for production")
    runner = ProductionRunner(json.loads(args.config.read_text()))
    try:
        runner.run()
    except Exception as error:
        runner.status("error", error=type(error).__name__, message=str(error)[:500])
        stop_service(runner.service)
        raise


if __name__ == "__main__":
    main()
