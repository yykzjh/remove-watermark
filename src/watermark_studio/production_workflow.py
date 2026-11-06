"""Run a real calibration pilot, then production, packaging and cloud publication."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
import urllib.request
from pathlib import Path

from .dataset_transfer import write_json
from .production import ProductionRunner, gpu_inventory, process_start, stop_service
from .production_archives import build_archives
from .production_cloud import publish


def request(url, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url.rstrip("/") + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=45) as response:
        return json.load(response)


def pilot(c):
    state = Path(c["pilot_state"])
    state.mkdir(parents=True, exist_ok=True)
    model_ready = Path(c["image_model"]) / "verified-files.json"
    while not model_ready.is_file():
        write_json(state / "status.json", {"phase": "waiting_for_verified_model"})
        time.sleep(10)
    free = [r["index"] for r in gpu_inventory() if r["available"] >= 44000
            and r["index"] in c.get("allowed_gpus", list(range(8)))]
    while not free:
        write_json(state / "status.json", {"phase": "waiting_for_calibration_gpu"})
        time.sleep(30)
        free = [r["index"] for r in gpu_inventory() if r["available"] >= 44000
                and r["index"] in c.get("allowed_gpus", list(range(8)))]
    command = json.loads((state / "service-command.json").read_text())
    command[command.index("--gpus") + 1] = str(free[0])
    control = state / "service-process.json"
    if control.exists():
        previous = json.loads(control.read_text())
        stop_service(previous)  # process identity and ownership are checked before signalling.
    env = dict(os.environ, UV_PROJECT_ENVIRONMENT=c["environment"], PYTHONUNBUFFERED="1")
    with (state / "service.log").open("ab") as log:
        p = subprocess.Popen(command, cwd=c["repo"], env=env, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    plan = {"pid": p.pid, "start": process_start(p.pid), "command": command, "gpus": [free[0]]}
    write_json(control, plan)
    with (state / "memory-monitor.log").open("ab") as log:
        monitor = subprocess.Popen([str(Path(c["environment"]) / "bin/python"), "-u", "-m",
                     "watermark_studio.production", "--memory-process", str(p.pid),
                     "--memory-output", str(state / "memory.json")], cwd=c["repo"], env=env,
                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True)
    try:
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            write_json(state / "status.json", {"phase": "loading", "gpu": free[0]})
            if p.poll() is not None:
                raise RuntimeError("Pilot service exited during model deployment")
            try:
                health = request(c["pilot_url"], "/health")
            except OSError:
                time.sleep(2)
                continue
            if health["stage"] == "failed":
                raise RuntimeError("Pilot model deployment failed: " + health.get("error", ""))
            if health["stage"] == "ready":
                break
            time.sleep(2)
        else:
            raise TimeoutError("Pilot model deployment exceeded time limit")
        if health.get("quality_weight_identity", {}).get("relation") == "same_weights":
            raise ValueError("Editor and verifier weights must remain independent")
        options = dict(c["options"], concurrency=4)
        submitted = request(c["pilot_url"], "/jobs", {
            "input_dir": c["pilot_input"], "output_dir": c["pilot_output"], "options": options})
        began = time.monotonic()
        job_id = submitted["id"]
        write_json(state / "job.json", {"id": job_id, "options": options})
        deadline = time.monotonic() + 7200
        while time.monotonic() < deadline:
            health = request(c["pilot_url"], "/health")
            job = health.get("job")
            if not job or job["id"] != job_id or p.poll() is not None:
                raise RuntimeError("Pilot inference job identity lost")
            write_json(state / "status.json", {"phase": "processing", "complete": job["completed"],
                                              "total": job["total"], "job": job_id})
            if job["state"] in {"completed", "failed", "interrupted", "cancelled"}:
                break
            time.sleep(5)
        else:
            raise TimeoutError("Pilot inference exceeded time limit")
        if job["state"] != "completed":
            raise RuntimeError("Pilot job failed before all images completed")
        report = Path(c["pilot_server_state"]) / job_id / "results.jsonl"
        rows = [json.loads(line) for line in report.read_text().splitlines()]
        from PIL import Image
        for row in rows:
            if not row.get("ok"):
                raise ValueError("Pilot contains processing errors")
            with Image.open(Path(c["pilot_input"]) / row["file"]) as original:
                size = original.size
            with Image.open(Path(c["pilot_output"]) / row["file"]) as result:
                result.load()
                if result.size != size:
                    raise ValueError("Pilot output did not retain original dimensions")
        if len(rows) != 32:
            raise ValueError("Pilot did not process the complete 32-image frozen sample")
        memory = json.loads((state / "memory.json").read_text())
        measured = max([*memory["peak_process_mib"].values(),
                        *(r.get("peak_allocated_mib", 0) for r in rows)])
        seconds = time.monotonic() - began
        summary = {"phase": "pilot_completed", "images": len(rows), "seconds": seconds,
                   "images_per_second": len(rows) / seconds, "peak_process_mib": measured,
                   "required_free_mib": math.ceil(measured + 2048),
                   "passed": sum(r.get("quality", {}).get("status") == "passed" for r in rows),
                   "review": sum(r.get("quality", {}).get("status") != "passed" for r in rows),
                   "batch_size": command[command.index("--batch-size") + 1], "job": job_id,
                   "dimensions_verified": True}
        write_json(state / "report.json", summary)
        write_json(state / "status.json", summary)
        return summary
    finally:
        stop_service(plan)
        try:
            monitor.wait(timeout=10)
        except subprocess.TimeoutExpired:
            monitor.terminate()


def remote(c):
    state = Path(c["workflow_state"])
    state.mkdir(parents=True, exist_ok=True)
    source_report = c.get("full_source_report")
    if source_report:
        while not Path(source_report).is_file():
            write_json(state / "status.json", {"phase": "waiting_for_full_verified_dataset"})
            time.sleep(10)
        data = json.loads(Path(source_report).read_text())
        if data["state"] != "completed" or data["verified_files"] != 39155 or data["directories_including_root"] != 5593:
            raise ValueError("Full source copy has not passed exact inventory and byte verification")
    # Errors retain directory leases and evidence. Retry infrastructure failures without
    # converting an entire failed GPU batch into allegedly completed production work.
    for attempt in range(4):
        runner = ProductionRunner(c["production"])
        write_json(state / "status.json", {"phase": "processing", "controller_attempt": attempt + 1})
        try:
            runner.run()
            break
        except Exception as error:
            stop_service(runner.service)
            write_json(state / "status.json", {"phase": "recovering", "attempt": attempt + 1,
                       "error": type(error).__name__, "message": str(error)[:400]})
            if attempt == 3:
                raise
            time.sleep(15 * (attempt + 1))
    ownership = runner.broker("/broker/ownership")
    original = runner.broker("/manifest")
    write_json(state / "ownership.json", ownership)
    write_json(state / "original-manifest.json", original)
    write_json(state / "status.json", {"phase": "packaging"})
    index = build_archives(Path(c["production"]["output_root"]),
              Path(c["production"]["state_dir"]), ownership, original,
              c["production"]["node"], Path(c["archive_dir"]))
    write_json(state / "status.json", {"phase": "uploading", "artifacts": len(index["artifacts"])})
    receipt = publish(Path(c["archive_dir"]) / "index.json", c["cloud_root"],
                      Path(c["baidu_config"]), state / "cloud")
    write_json(state / "delivery.json", receipt)
    write_json(state / "status.json", {"phase": "published", "delivery": receipt})
    return receipt


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("mode", choices=["pilot", "remote"])
    p.add_argument("--config", type=Path, required=True)
    a = p.parse_args()
    c = json.loads(a.config.read_text())
    try:
        result = pilot(c) if a.mode == "pilot" else remote(c)
    except Exception as error:
        state = Path(c["pilot_state"] if a.mode == "pilot" else c["workflow_state"])
        write_json(state / "status.json", {"phase": "error", "error": type(error).__name__,
                                           "message": str(error)[:500]})
        raise
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
