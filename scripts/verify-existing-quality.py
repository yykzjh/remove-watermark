"""Real GPU integration: review existing outputs read-only; never rewrite them."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from watermark_studio.cli import request

parser = argparse.ArgumentParser()
parser.add_argument("--repair-existing", action="store_true")
args = parser.parse_args()

root = Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031")
case = root/"quality-readonly-8"
inputs, outputs = case/"input", case/"output"
inputs.mkdir(parents=True, exist_ok=True)
outputs.mkdir(parents=True, exist_ok=True)
base = root/"quality-benchmark-stage-batch-64"
for source in sorted((base/"input").iterdir())[:8]:
    for original, target in ((source, inputs/source.name),
                             (base/"quality-768-b2"/source.name, outputs/source.name)):
        if not target.exists():
            os.link(original, target)
def hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs.iterdir()}
before = hashes()
source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
job = request("http://127.0.0.1:18770", "/jobs", {
    "input_dir": str(inputs), "output_dir": str(outputs),
    "options": {"concurrency": 20, "quality_check": True, "quality_auto_retry": True,
                "quality_batch_size": 2, "overwrite": args.repair_existing,
                "quality_max_attempts": 2},
})["id"]
started = time.monotonic()
while True:
    state = request("http://127.0.0.1:18770", "/health")["job"]
    if state["state"] in {"completed", "failed", "cancelled"}:
        break
    if time.monotonic()-started > 300:
        raise TimeoutError(job)
    time.sleep(.3)
assert state["state"] == "completed" and state["failed"] == 0
assert source_hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
if args.repair_existing:
    rows = [json.loads(line) for line in (Path.home()/".local/state/watermark-studio"/job/"results.jsonl").read_text().splitlines()]
    for row in rows:
        if row["attempt_history"][0]["quality"]["status"] == "passed":
            assert hashes()[row["file"]] == before[row["file"]]
        else:
            assert row["quality_attempts"] == 2
else:
    assert state["existing_outputs"] == 8 and hashes() == before
    assert state.get("retry_attempts", 0) == 0
record = {"test": "existing_outputs_resume" if args.repair_existing else "existing_outputs_readonly", "images": 8,
          "seconds": round(time.monotonic()-started, 3), "unchanged": hashes() == before,
          "source_unchanged": True, "retry_attempts": state.get("retry_attempts", 0),
          "quality_counts": state["quality_counts"], "job": job}
(case/("resume-result.json" if args.repair_existing else "result.json")).write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
