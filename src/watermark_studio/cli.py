import argparse
import json
import time
import urllib.error
import urllib.request


def request(url, path, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url.rstrip("/") + path, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode()[:1000]) from error


def main():
    parser = argparse.ArgumentParser(description="Batch images already on the service machine")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--input-resolution", "--resolution", dest="resolution", type=int, choices=[512, 768, 1024, 1280, 1536, 2048], default=1024)
    parser.add_argument("--output-width", type=int, default=0)
    parser.add_argument("--output-height", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quality-check", action="store_true")
    parser.add_argument("--quality-resolution", type=int, choices=[512, 768, 1024, 1280], default=768)
    parser.add_argument("--quality-batch-size", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--quality-max-tokens", type=int, default=128)
    parser.add_argument("--quality-detail-patches", type=int, choices=[0, 2, 4, 6, 8], default=4)
    args = parser.parse_args()
    request(
        args.url,
        "/jobs",
        {
            "input_dir": args.input_dir,
            "output_dir": args.output_dir,
            "options": {
                "concurrency": args.concurrency,
                "resolution": args.resolution,
                "output_width": args.output_width,
                "output_height": args.output_height,
                "overwrite": args.overwrite,
                "quality_check": args.quality_check,
                "quality_resolution": args.quality_resolution,
                "quality_batch_size": args.quality_batch_size,
                "quality_max_tokens": args.quality_max_tokens,
                "quality_detail_patches": args.quality_detail_patches,

            },
        },
    )
    previous = None
    while True:
        job = request(args.url, "/health")["job"]
        update = (job["state"], job["completed"], job["total"])
        if update != previous:
            print(
                f"{job['state']}: {job['completed']}/{job['total']}; "
                f"success={job['succeeded']}, no_watermark={job.get('no_watermark', 0)}, "
                f"failed={job['failed']}, existing_review={job.get('existing_outputs', 0)}, quality={job.get('quality_counts', {})}",
                flush=True,
            )
            previous = update
        if job["state"] in {"completed", "failed", "cancelled", "interrupted"}:
            if job["state"] != "completed" or job["failed"]:
                raise SystemExit(1)
            counts = job.get("quality_counts", {})
            raise SystemExit(2 if counts.get("review", 0) or counts.get("error", 0) else 0)
        time.sleep(2)


if __name__ == "__main__":
    main()
