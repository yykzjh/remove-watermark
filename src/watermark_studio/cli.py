import argparse
import json
import time
import urllib.error
import urllib.request

from .config import ProcessingConfig, processing_route


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
    parser.add_argument("--remove-logo", action="store_true", help="Also process detected app logos and their small attached codes")
    parser.add_argument("--quality-check", action="store_true")
    parser.add_argument("--quality-cross-review", action="store_true")
    parser.add_argument("--quality-auto-retry", action="store_true")
    parser.add_argument("--quality-max-attempts", type=int, default=3, help="0: until pass or cancel")
    parser.add_argument("--quality-retry-resolution", type=int, default=1024)
    parser.add_argument("--preserve-background", action="store_true", help="Enable original protection for fine crop strategies or whole-image composition")
    parser.add_argument("--roi-grouping", choices=["separate","adaptive"], default="separate", help="adaptive: combine readable nearby ROI references; composite only original masks")
    parser.add_argument("--context",type=float,default=.6,help="Reconstruction context factor, 0.1..2")
    parser.add_argument("--first-pass-mode",choices=["regions","whole","focused","thorough","complete"],default="regions")
    parser.add_argument("--whole-prompt-profile",choices=["baseline","fidelity","scene","located","concise","neutral","clean-photo","clean-photo-detailed"],default=ProcessingConfig.model_fields["whole_prompt_profile"].default,help="neutral: source-material fidelity; fidelity/baseline: historical comparisons; concise: material-mismatch retry; scene/located/clean-photo/clean-photo-detailed: experimental")
    parser.add_argument("--repair-blending", choices=["feather","harmonic","harmonic-fine"], default="harmonic")
    parser.add_argument("--quality-resolution", type=int, choices=[512, 768, 1024, 1280], default=768)
    parser.add_argument("--quality-batch-size", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--quality-max-tokens", type=int, default=128)
    parser.add_argument("--quality-detail-patches", type=int, choices=range(9), default=4)
    args = parser.parse_args()
    if not .1<=args.context<=2:
        parser.error("--context must be finite and in 0.1..2")
    scope,first_mode=processing_route(args.first_pass_mode)
    request(
        args.url,
        "/jobs",
        {
            "input_dir": args.input_dir,
            "output_dir": args.output_dir,
            "options": {
                "scope": scope,
                "remove_logo": args.remove_logo,
                "concurrency": args.concurrency,
                "resolution": args.resolution,
                "context":args.context,
                "roi_grouping":args.roi_grouping,
                "output_width": args.output_width,
                "output_height": args.output_height,
                "overwrite": args.overwrite,
                "quality_check": args.quality_check,
                "quality_cross_review": args.quality_cross_review,
                "quality_auto_retry": args.quality_auto_retry,
                "quality_max_attempts": args.quality_max_attempts,
                "quality_retry_resolution": args.quality_retry_resolution,
                "preserve_background": args.preserve_background,
                "first_pass_mode":first_mode,
                "whole_prompt_profile":args.whole_prompt_profile,
                "repair_blending": args.repair_blending,
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
        update = (job["state"], job["completed"], job["total"],
                  job.get("mirror", {}).get("copied_files", 0),
                  job.get("mirror", {}).get("copy_failures", 0))
        if update != previous:
            print(
                f"{job['state']}: {job['completed']}/{job['total']}; "
                f"success={job['succeeded']}, no_watermark={job.get('no_watermark', 0)}, "
                f"failed={job['failed']}, existing_review={job.get('existing_outputs', 0)}, quality={job.get('quality_counts', {})}; "
                f"mirror={job.get('mirror', {})}",
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
