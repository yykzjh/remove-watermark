"""Compare startup load parallelism; only run with an idle stopped service."""
import json
import subprocess
import time
from pathlib import Path

from watermark_studio.cli import request

root = Path(".run/tuning")
root.mkdir(parents=True, exist_ok=True)
try:
    existing = request("http://127.0.0.1:8765", "/health")
except OSError:
    existing = None
if existing:
    raise SystemExit("Stop the existing service after its jobs finish before measuring startup")
for parallel in (1, 2):
    start = time.monotonic()
    subprocess.run(["bash", "scripts/launch-server.sh", "--load-parallelism", str(parallel)],
                   check=True)
    deadline = start+600
    while time.monotonic() < deadline:
        try:
            state = request("http://127.0.0.1:8765", "/health")
        except OSError:
            time.sleep(1)
            continue
        if state["stage"] == "failed":
            raise RuntimeError(state["error"])
        if state["stage"] == "ready":
            break
        time.sleep(1)
    else:
        raise RuntimeError("Startup exceeded 600 seconds")
    report = {"load_parallelism":parallel, "ready_seconds":round(time.monotonic()-start,3),
              "worker_count":state["worker_count"]}
    with (root/"startup.jsonl").open("a") as stream:
        stream.write(json.dumps(report)+"\n")
    print(json.dumps(report),flush=True)
    if parallel == 1:
        subprocess.run(["bash","scripts/stop-server.sh"],check=True)
