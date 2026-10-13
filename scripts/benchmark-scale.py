"""10k-file scan and image I/O benchmarks; no model/embedding cache."""
import json
import os
import time
from pathlib import Path

from PIL import Image

from watermark_studio.images import scan_images

root = Path(".run/deep-tuning/scale")
source = root / "input"
source.mkdir(parents=True, exist_ok=True)
samples = sorted(Path(".run/user-samples/input").glob("*.jpg"))
for index in range(10000):
    path = source / f"folder-{index % 100:03}" / f"{index:05}.jpg"
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        os.link(samples[index % 2], path)
records = []
for workers in [1, 4, 8, 1]:
    start = time.monotonic()
    tasks, skipped, invalid = scan_images(source.resolve(), (root / "output").resolve(),
                                         False, workers=workers)
    record = {"kind": "scan", "workers": workers, "images": len(tasks),
              "seconds": round(time.monotonic() - start, 3), "invalid": len(invalid)}
    assert len(tasks) == 10000 and not skipped and not invalid
    records.append(record)
    print(json.dumps(record), flush=True)
for sample in samples:
    start = time.monotonic()
    with Image.open(sample) as raw:
        image = raw.convert("RGB")
    decode_s = time.monotonic() - start
    for quality in [97, 95, 90]:
        timings = []
        destination = root / f"jpeg-{quality}.jpg"
        for _ in range(3):
            start = time.monotonic()
            image.save(destination, quality=quality, subsampling=0)
            timings.append(time.monotonic() - start)
        record = {"kind": "jpeg", "size": image.size, "quality": quality,
                  "decode_s": round(decode_s, 3), "save_s": [round(s, 3) for s in timings],
                  "bytes": destination.stat().st_size}
        records.append(record)
        print(json.dumps(record), flush=True)
(root / "results.json").write_text(json.dumps(records, indent=2))
