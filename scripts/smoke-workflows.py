"""Run on the development machine; only print compact verification metadata."""
import argparse
import json
import shutil
import time
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image

from watermark_studio.cli import request

parser = argparse.ArgumentParser()
parser.add_argument("--only-auto", action="store_true")
args = parser.parse_args()

url = "http://127.0.0.1:8765"
root = Path(".run/user-samples").resolve()
samples = sorted((root / "input").glob("*.jpg"))
assert len(samples) == 2, "Upload the two authorized user samples first"
batch = root / "batch-input"
if not args.only_auto:
    for index in range(16):
        destination = batch / f"group-{index % 2}" / f"{index:02d}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(samples[index % 2], destination)
    (batch / "ignore.txt").write_text("not an image")
records = []


def run(label, payload):
    started = time.monotonic()
    response = request(url, "/jobs", payload)
    for _ in range(300):
        state = request(url, "/health")
        job = state["job"]
        if job["state"] in {"completed", "failed", "cancelled"}:
            assert job["failed"] == 0 and job["state"] == "completed", job
            results = [json.loads(line) for line in
                       (Path.home() / ".local/state/watermark-studio" / response["id"] / "results.jsonl")
                       .read_text().splitlines()]
            record = {"label": label, "job": response["id"], "succeeded": job["succeeded"],
                      "seconds": round(time.monotonic() - started, 2),
                      "workers": sorted({r["worker"] for r in results}),
                      "batch_sizes": sorted({b for r in results for b in r.get("batch_sizes", [])}),
                      "sizes": sorted({tuple(r["output_size"]) for r in results})}
            records.append(record)
            print(json.dumps(record), flush=True)
            return results
        time.sleep(2)
    raise RuntimeError("Timed out")


if not args.only_auto:
    run("full-image-eight-gpu", {
        "input_dir": str(batch), "output_dir": str(root / "batch-output"),
        "options": {"concurrency": 32, "resolution": 1024, "overwrite": True},
    })
    for source in batch.rglob("*.jpg"):
        with Image.open(source) as a, Image.open(root / "batch-output" / source.relative_to(batch)) as b:
            assert a.size == b.size

    preview = request(url, "/preview?" + urlencode({"path": str(samples[0])}))
    assert preview["bytes"] <= 65536
    run("single-custom-resized", {
        "mode": "single", "input_dir": str(root / "input"), "input_file": str(samples[0]),
        "preview_token": preview["token"], "output_dir": str(root / "custom-output"),
        "options": {"concurrency": 1, "resolution": 1024, "overwrite": True,
                    "preset": "custom", "regions": [
                        {"box": [.012, .727, .45, .988]}, {"box": [.864, .89, 1, 1]}],
                    "output_width": 1920, "output_height": 1440},
    })
run("single-auto-original", {
    "mode": "single", "input_dir": str(root / "input"), "input_file": str(samples[1]),
    "output_dir": str(root / "auto-output"),
    "options": {"concurrency": 1, "resolution": 1024, "overwrite": True},
})
(root / ("verification-auto.json" if args.only_auto else "verification.json")).write_text(json.dumps(records, indent=2))
