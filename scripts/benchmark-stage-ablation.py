"""Controlled restart/ablation of the test service only (port 18770)."""
import argparse
import json
import subprocess
import time
from pathlib import Path

from watermark_studio.cli import request

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=["serial", "batched"], required=True)
parser.add_argument("--profiles", nargs="+", default=["baseline-hot", "quality-768-b2"])
args = parser.parse_args()
config = json.loads(Path(".run/deployment.json").read_text())
if config["port"] != 18770:
    raise ValueError("This script only restarts the dedicated test service on port 18770")
url = "http://127.0.0.1:18770"
state = request(url, "/health")
if state.get("job") and state["job"]["state"] in {"scanning", "running", "paused", "cancelling"}:
    raise ValueError("Wait for the current test to finish")
config.update(adaptive_dispatch=args.variant == "batched",
              detection_batch_size=2 if args.variant == "batched" else 1)
subprocess.run(["bash", "scripts/stop-server.sh"], check=True)
flags = []
for key, value in config.items():
    flag = "--" + key.replace("_", "-")
    if isinstance(value, bool):
        if value:
            flags.append(flag)
    elif key == "gpus":
        flags.extend([flag, ",".join(map(str, value))])
    elif isinstance(value, list):
        flags.extend([flag, *map(str, value)])
    else:
        flags.extend([flag, str(value)])
subprocess.run(["bash", "scripts/start-server.sh", *flags], check=True)
deadline = time.monotonic() + 600
while time.monotonic() < deadline:
    try:
        state = request(url, "/health")
        if state["stage"] == "ready":
            break
        if state["stage"] == "failed":
            raise RuntimeError(state["error"])
    except (OSError, RuntimeError):
        if time.monotonic() + 5 > deadline:
            raise
    time.sleep(3)
else:
    raise TimeoutError("Model load")
Path(f".run/ablation-{args.variant}-deployment.json").write_text(json.dumps(config, indent=2))
subprocess.run([
    ".venv/bin/python", "scripts/benchmark-quality.py",
    "--tag", "ablation-" + args.variant + "-64", "--images", "64",
    "--profiles", *args.profiles,
], check=True)
