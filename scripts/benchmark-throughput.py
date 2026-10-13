"""Controlled eight-GPU service experiments with fixed inputs and full output verification."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps

from watermark_studio.cli import request

parser = argparse.ArgumentParser()
parser.add_argument("--profiles", nargs="+", default=["baseline", "compiled", "fp8", "fp8-projections"])
parser.add_argument("--images", type=int, default=128)
parser.add_argument("--pattern", choices=["uniform", "mixed"], default="uniform")
parser.add_argument("--repeats", type=int, default=2)
parser.add_argument("--concurrency", type=int, default=32)
args = parser.parse_args()
root = Path(".run/throughput").resolve()
root.mkdir(exist_ok=True)
fixtures = root / "fixtures"
fixtures.mkdir(exist_ok=True)
originals = sorted(Path(".run/user-samples/input").glob("*.jpg"))
assert len(originals) == 2
for i, source in enumerate(originals):
    raw = Image.open(source).convert("RGB")
    target = fixtures / f"original-{i}.jpg"
    if not target.exists():
        os.link(source, target)
    for ratio, size in [("portrait", (1200, 1600)), ("square", (1400, 1400)), ("wide", (1600, 900))]:
        target = fixtures / f"{ratio}-{i}.jpg"
        if not target.exists():
            ImageOps.pad(raw, size, color=(110, 120, 105)).save(target, quality=97, subsampling=0)
source = root / f"input-{args.pattern}-{args.images}"
source.mkdir(exist_ok=True)
pool = sorted(fixtures.glob("original-*.jpg") if args.pattern == "uniform" else fixtures.glob("*.jpg"))
for index in range(args.images):
    path = source / f"folder-{index % 7}" / f"{index:05}.jpg"
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        os.link(pool[index % len(pool)], path)
(source / "not-image.txt").write_text("ignored")
url = "http://127.0.0.1:18770"
common = [
    "--config", "config/deployment.toml", "--port", "18770", "--gpus", "0,1,2,3,4,5,6,7",
    "--load-parallelism", "8", "--batch-size", "2", "--batch-window", "4",
    "--kv-cache", "--vae-tiling", "--vae-slicing", "--tf32", "--batch-bucketing",
    "--oom-split-batch", "--cpu-threads", "1",
]
profiles = {
    "baseline": [],
    "compiled": ["--prompt-encoder", "lean", "--encoder-batch-size", "1",
                 "--compile", "--compile-scope", "blocks", "--compile-dynamic",
                 "--expandable-segments", "--shape-sort", "--report-interval", "64"],
    "fp8": ["--prompt-encoder", "lean", "--encoder-batch-size", "2",
            "--quantization", "float8", "--compile", "--compile-scope", "blocks",
            "--expandable-segments", "--shape-sort", "--report-interval", "64"],
    "fp8-projections": ["--prompt-encoder", "lean", "--encoder-batch-size", "2",
                        "--quantization", "float8", "--compile", "--compile-scope", "projections",
                        "--compile-dynamic", "--expandable-segments", "--shape-sort", "--report-interval", "64"],
    "fp8-b1": ["--prompt-encoder", "lean", "--quantization", "float8", "--compile",
               "--compile-scope", "blocks", "--batch-size", "1", "--batch-window", "4",
               "--expandable-segments", "--shape-sort", "--report-interval", "64"],
    "fp8-b4": ["--prompt-encoder", "lean", "--encoder-batch-size", "2",
               "--quantization", "float8", "--compile", "--compile-scope", "blocks",
               "--batch-size", "4", "--expandable-segments", "--shape-sort", "--report-interval", "64"],
    "eager-b1": ["--batch-size", "1", "--expandable-segments", "--shape-sort", "--report-interval", "64"],
}
records = []
for profile in args.profiles:
    command = [sys.executable, "-m", "watermark_studio.server", *common, *profiles[profile]]
    log_path = root / f"server-{profile}-{args.pattern}.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
        try:
            boot = time.monotonic()
            for _ in range(300):
                try:
                    state = request(url, "/health")
                    if state["stage"] == "failed":
                        raise RuntimeError(state["error"])
                    if state["stage"] == "ready":
                        break
                except (OSError, RuntimeError):
                    if "state" in locals() and state.get("stage") == "failed":
                        raise RuntimeError(state.get("error", "Deployment failed")) from None
                    if process.poll() is not None:
                        raise RuntimeError(f"Service exited; see {log_path}") from None
                time.sleep(1)
            else:
                raise TimeoutError("Service startup timeout")
            boot_s = time.monotonic() - boot
            for repeat in range(args.repeats):
                target = root / f"output-{profile}-{args.pattern}"
                start = time.monotonic()
                job_id = request(url, "/jobs", {
                    "input_dir": str(source), "output_dir": str(target),
                    "options": {"concurrency": min(args.concurrency, state["capacity"]),
                                "resolution": 1024, "overwrite": True},
                })["id"]
                samples = []
                max_inflight = 0
                for tick in range(28800):
                    state = request(url, "/health")
                    job = state["job"]
                    if job["id"] != job_id:
                        raise RuntimeError("Benchmark job was replaced")
                    max_inflight = max(max_inflight, state["inflight_images"])
                    if tick % 40 == 0:
                        samples.append({"seconds": round(time.monotonic() - start, 2),
                                        "completed": job["completed"]})
                    if job["state"] in {"completed", "failed", "cancelled"}:
                        break
                    time.sleep(.25)
                else:
                    raise TimeoutError("Job exceeded two hours")
                elapsed = time.monotonic() - start
                rows = [json.loads(line) for line in
                        (Path.home() / ".local/state/watermark-studio" / job_id / "results.jsonl")
                        .read_text().splitlines()]
                bad_sizes = 0
                for row in rows:
                    if not row["ok"]:
                        continue
                    with Image.open(source / row["file"]) as before, Image.open(target / row["file"]) as after:
                        bad_sizes += before.size != after.size
                record = {
                    "profile": profile, "pattern": args.pattern, "images": args.images,
                    "repeat": repeat, "job": job_id, "boot_s": round(boot_s, 3),
                    "seconds": round(elapsed, 3), "images_per_s": round(job["succeeded"] / elapsed, 4),
                    "success": job["succeeded"], "failed": job["failed"], "bad_sizes": bad_sizes,
                    "workers": sorted({row["worker"] for row in rows}), "max_inflight": max_inflight,
                    "batch_sizes": sorted({n for row in rows for n in row.get("batch_sizes", [])}),
                    "oom_retries": max((row["oom_retries"] for row in rows), default=0),
                    "peak_allocated_mib": max((row["peak_allocated_mib"] for row in rows), default=0),
                    "progress_samples": samples,
                }
                by_worker = {}
                for row in rows:
                    by_worker.setdefault(row["worker"], []).append(row.get("peak_allocated_mib", 0))
                record["memory_by_worker"] = {
                    str(worker): {"first": values[0], "last": values[-1], "peak": max(values)}
                    for worker, values in by_worker.items()
                }
                records.append(record)
                with (root / "results.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(json.dumps({k: v for k, v in record.items() if k != "progress_samples"}), flush=True)
                assert job["state"] == "completed" and job["succeeded"] == args.images and bad_sizes == 0, record
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # Only the isolated process group created by this script belongs to us.
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
