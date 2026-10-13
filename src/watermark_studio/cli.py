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
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--preset", choices=["auto", "gray", "blue"], default="auto")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--detection-resolution", type=int, default=1280)
    parser.add_argument("--remove-logo", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    request(
        args.url,
        "/jobs",
        {
            "input_dir": args.input_dir,
            "output_dir": args.output_dir,
            "options": {
                "concurrency": args.concurrency,
                "preset": args.preset,
                "resolution": args.resolution,
                "detection_resolution": args.detection_resolution,
                "overwrite": args.overwrite,
                "remove_logo": args.remove_logo,
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
                f"failed={job['failed']}",
                flush=True,
            )
            previous = update
        if job["state"] in {"completed", "failed", "cancelled", "interrupted"}:
            raise SystemExit(0 if job["state"] == "completed" and not job["failed"] else 1)
        time.sleep(2)


if __name__ == "__main__":
    main()
