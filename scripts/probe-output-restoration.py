"""One full-frame Qwen generation, output-only SR, and exact synthetic GT metrics.

This is a development experiment on previously exposed training-data validation
crops. It does not access the sealed native acceptance set or grant any QA pass.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from watermark_studio.exact_metrics import detail_reconstruction_metrics, reconstruction_metrics
from watermark_studio.images import load_photo
from watermark_studio.training_features import digest, validate_pairs


def select_pairs(data, original_limit, variants_per_original, seed):
    groups = {}
    for row in data["pairs"]:
        if row["split"] == "validation" and row.get("training_crop_kind") == "lower" and row.get("model_input_resolution") == 1024:
            groups.setdefault(row["source_original_sha256"], []).append(row)
    keys = sorted(groups, key=lambda k: hashlib.sha256((str(seed) + k).encode()).hexdigest())
    if len(keys) < original_limit:
        raise ValueError("Insufficient held-out 1024 crop originals")
    rows = []
    for key in keys[:original_limit]:
        group = sorted(groups[key], key=lambda row: int(row["id"].rsplit("-", 1)[-1]))
        if len(group) < variants_per_original:
            raise ValueError("Insufficient pre-existing render variants")
        rows.extend(group[:variants_per_original])
    return rows


def known_regions(row, mask):
    """Frozen synthetic overlay geometry only; clean target pixels are never input."""
    regions = []
    w, h = mask.size
    remainder = mask.copy()
    metadata = row["render"].get("render", row["render"])
    panel = metadata.get("panel")
    if panel:
        l, t, r, b = map(int, panel)
        if not 0 <= l < r <= w or not 0 <= t < b <= h:
            raise ValueError("Synthetic source panel outside canvas")
        regions.append({"kind": "panel", "box": [l / w, t / h, r / w, b / h]})
        remainder.paste(0, (l, t, r, b))
    extra = remainder.getbbox()
    if extra:
        l, t, r, b = extra
        regions.append({"kind": "logo", "box": [l / w, t / h, r / w, b / h]})
    return regions


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--original-limit", type=int, default=4)
    parser.add_argument("--variants-per-original", type=int, choices=[1, 2, 3, 4], default=4)
    parser.add_argument("--seed", type=int, default=2026110213)
    parser.add_argument("--resolution", type=int, choices=[512, 768, 1024], default=512)
    parser.add_argument("--uniform-steps", type=int, choices=[12, 16, 20, 24, 40], default=12)
    parser.add_argument("--upscaler", type=Path, required=True)
    parser.add_argument("--upscaler-sha256", required=True)
    parser.add_argument("--sr-precision", choices=["float32", "float16"], default="float32")
    parser.add_argument("--sr-tile", type=int, choices=[0, 256, 512], default=256)
    parser.add_argument("--sr-overlap", type=int, choices=[0, 32, 64, 128], default=32)
    parser.add_argument("--generation-experiment", type=Path)
    args = parser.parse_args()
    if args.gpu < 0 or not 1 <= args.original_limit <= 24:
        parser.error("Explicit disjoint GPU and bounded original count")
    if not args.generation_experiment and not args.model:
        parser.error("Initial generation requires a local model")
    data, files = validate_pairs(args.manifest)
    rows = select_pairs(data, args.original_limit, args.variants_per_original, args.seed)
    if args.target.resolve().parent != args.manifest.resolve().parent:
        parser.error("Output must stay within the existing development dataset root")
    helper = runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    def isolation():
        with urllib.request.urlopen("http://127.0.0.1:8765/health", timeout=2) as response:
            return helper["require_disjoint_server_gpus"](json.load(response), [args.gpu], True)
    reservation = isolation()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest_sha = digest(args.manifest)
    args.target.mkdir(exist_ok=False)
    work = args.target.resolve()
    (work / "raw").mkdir()
    (work / "output").mkdir()
    (work / "baseline").mkdir()
    (work / "source-selection.json").write_text(json.dumps({
        "source_sha256": {r["source_original"]: r["source_original_sha256"] for r in rows},
        "purpose": "Previously exposed synthetic validation originals; not blind acceptance",
        "selected_ids": [r["id"] for r in rows]}, indent=2))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.pop("WATERMARK_REPAIR_CAPTURE", None)
    import torch

    from watermark_studio.batching import dimensions, independent_batch
    from watermark_studio.config import DeploymentConfig
    from watermark_studio.prompts import build_full_prompt
    from watermark_studio.swinir_probe import load_swinir, upscale_swinir
    from watermark_studio.worker import load_pipeline

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    began = time.monotonic()
    selected = [r["id"] for r in rows]
    generated = []
    additional_generations = 0
    generation_config = {"dataset_manifest_sha256": manifest_sha, "selected_ids": selected,
        "resolution": args.resolution, "uniform_steps": args.uniform_steps, "seed": args.seed,
        "model": str(args.model.resolve()) if args.model else None,
        "prompt": "neutral", "quantization": "float8_tensor", "context": "known_synthetic_source_regions"}
    if args.generation_experiment:
        previous_root = args.generation_experiment.resolve(strict=True)
        previous_config = json.loads((previous_root / "generation-config.json").read_text())
        if generation_config["model"] is None:
            generation_config["model"] = previous_config["model"]
        if previous_config != generation_config:
            raise ValueError("Frozen generation configuration cannot change in replay/resume")
        previous_selection = json.loads((previous_root / "source-selection.json").read_text())
        expected_sources = {r["source_original"]: r["source_original_sha256"] for r in rows}
        if previous_selection["selected_ids"] != selected or previous_selection["source_sha256"] != expected_sources:
            raise ValueError("Frozen source order cannot change")
        prior = list(map(json.loads, (previous_root / "generations.jsonl").read_text().splitlines()))
        if len(prior) > len(rows) or [r["id"] for r in prior] != selected[:len(prior)]:
            raise ValueError("Only the exact completed prefix can be resumed")
        for row, record in zip(rows, prior, strict=False):
            with Image.open(row["mask"]) as im:
                empty = im.convert("L").getbbox() is None
            if record["generation_count"] == 0:
                if not empty:
                    raise ValueError("An overlaid source cannot be replaced by a source copy")
            else:
                raw = Path(record["raw"]).resolve(strict=True)
                source = load_photo(row["input"])[0]
                if (record["generation_count"] != 1 or empty or
                    record["source_sha256"] != digest(row["input"]) or
                    record["seed"] != (row["seed"] + 1000003) % 2**32 or
                    record["actual_steps"] != args.uniform_steps or
                    record["raw_size"] != list(dimensions(source, args.resolution)) or
                    not raw.is_relative_to(previous_root / "raw") or raw.stat().st_size > 16 * 2**20 or
                    digest(raw) != record["raw_sha256"]):
                    raise ValueError("One immutable full-source generation is required for every resumed image")
                destination = work / "raw" / raw.name
                shutil.copyfile(raw, destination)
                if digest(destination) != record["raw_sha256"]:
                    raise ValueError("Raw replay copy changed")
                record = record | {"raw": str(destination), "replayed_from": str(raw)}
            generated.append(record)
    if len(generated) < len(rows) and args.model is None:
        parser.error("A partial generation prefix requires the same local model to finish")
    (work / "generation-config.json").write_text(json.dumps(generation_config, indent=2))
    (work / "generations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in generated))
    if len(generated) < len(rows):
        cfg = DeploymentConfig(model=str(args.model), gpus=[args.gpu], quantization="float8",
            float8_granularity="tensor", prompt_encoder="lean", compile=True,
            compile_scope="projections", compile_dynamic=True, separate_vae_decode=True,
            uniform_steps=args.uniform_steps)
        pipe = load_pipeline(cfg, str(args.model))
        for row in rows[len(generated):]:
            source = load_photo(row["input"])[0]
            with Image.open(row["mask"]) as im:
                mask = im.convert("L")
            if not 950000 <= source.width * source.height <= 1310720 or mask.size != source.size:
                raise ValueError("Exact existing one-megapixel crop canvas required")
            seed = (row["seed"] + 1000003) % 2**32
            if mask.getbbox() is None:
                record = {"id": row["id"], "generation_count": 0, "reason": "known_empty_synthetic_overlay"}
            else:
                item = {"image": source, "prompt": build_full_prompt("", "neutral"), "seed": seed,
                        "_denoising_context_regions": known_regions(row, mask)}
                started = time.monotonic()
                raw = independent_batch(pipe, [item], args.resolution, True)[0].convert("RGB")
                if len(pipe.scheduler.timesteps) != args.uniform_steps:
                    raise ValueError("Actual deployed step count mismatch")
                path = work / "raw" / (row["id"] + ".png")
                raw.save(path)
                record = {"id": row["id"], "generation_count": 1, "seed": seed,
                          "source_sha256": digest(row["input"]), "raw": str(path),
                          "raw_sha256": digest(path), "raw_size": list(raw.size),
                          "seconds": round(time.monotonic() - started, 4),
                          "actual_steps": len(pipe.scheduler.timesteps), "full_source_input": True}
                additional_generations += 1
            generated.append(record)
            with (work / "generations.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps({"stage": "generation", "completed": len(generated), "total": len(rows)}), flush=True)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    generation_phase_seconds = time.monotonic() - began
    (work / "generation-report.json").write_text(json.dumps({
        "state": "completed", "stage": "generation_only", "code_revision": revision,
        "dataset_manifest_sha256": manifest_sha, "selected_ids": selected,
        "resolution": args.resolution, "uniform_steps": args.uniform_steps,
        "generations_this_run": additional_generations, "total_original_generations": sum(r["generation_count"] for r in generated),
        "generation_records_sha256": digest(work / "generations.jsonl"), "restoration_completed": False}, indent=2))
    began = time.monotonic()
    net = load_swinir(args.upscaler, args.upscaler_sha256, "cuda:0", "lightweight", args.sr_precision)
    load_seconds = time.monotonic() - began
    began = time.monotonic()
    results = []
    for row, record in zip(rows, generated, strict=True):
        source = load_photo(row["input"])[0]
        clean = load_photo(row["target"])[0]
        with Image.open(row["mask"]) as im:
            support_image = im.convert("L")
        mask = np.asarray(support_image) > 0
        if source.size != clean.size or source.size != support_image.size:
            raise ValueError("Exact source/target/mask alignment required")
        if record["generation_count"] == 0:
            if mask.any() or source.tobytes() != clean.tobytes():
                raise ValueError("Only exact empty-overlay controls may skip generation")
            for directory in ("output", "baseline"):
                destination = work / directory / (row["id"] + ".png")
                shutil.copyfile(row["input"], destination)
                if digest(destination) != digest(row["input"]):
                    raise ValueError("Known empty-overlay source copy changed")
            results.append({"id": row["id"], "generation_count": 0, "unchanged_copy": True,
                            "native_output_size": list(source.size)})
            continue
        if record["generation_count"] != 1 or record["source_sha256"] != digest(row["input"]) or digest(record["raw"]) != record["raw_sha256"]:
            raise ValueError("Exactly one immutable original generation required")
        raw = load_photo(record["raw"])[0]
        if list(raw.size) != record["raw_size"]:
            raise ValueError("Captured generation geometry mismatch")
        started = time.monotonic()
        restored = upscale_swinir(raw, net, args.sr_tile, args.sr_overlap).resize(source.size, Image.Resampling.LANCZOS)
        sr_seconds = time.monotonic() - started
        baseline = raw.resize(source.size, Image.Resampling.LANCZOS)
        before = reconstruction_metrics(np.asarray(baseline), np.asarray(clean), mask)
        after = reconstruction_metrics(np.asarray(restored), np.asarray(clean), mask)
        composed = Image.composite(restored, source, support_image.point(lambda v: 255 if v else 0))
        if not np.array_equal(np.asarray(composed)[~mask], np.asarray(source)[~mask]):
            raise ValueError("Protected source pixels changed")
        composed.save(work / "output" / (row["id"] + ".png"))
        Image.composite(baseline, source, support_image.point(lambda v: 255 if v else 0)).save(work / "baseline" / (row["id"] + ".png"))
        result = {"id": row["id"], "source_original_sha256": row["source_original_sha256"],
                  "generation_count": 1, "baseline": before, "restored": after,
                  "baseline_detail": detail_reconstruction_metrics(np.asarray(baseline), np.asarray(clean), mask),
                  "restored_detail": detail_reconstruction_metrics(np.asarray(restored), np.asarray(clean), mask),
                  "sr_seconds": round(sr_seconds, 4), "protected_pixels_unchanged": True,
                  "native_output_size": list(composed.size), "quality_permission": False}
        results.append(result)
        print(json.dumps({"stage": "restoration", "completed": len(results), "total": len(rows)}), flush=True)
    if digest(args.manifest) != manifest_sha or any(digest(p) != sha for p, sha in files.items()) or isolation() != reservation:
        raise ValueError("Development inputs or GPU reservation changed")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != revision:
        raise ValueError("Pinned experiment code changed")
    positives = [r for r in results if r["generation_count"] == 1]
    comparable = [r for r in positives if r["restored_detail"]["highpass_correlation"] is not None
                  and r["baseline_detail"]["highpass_correlation"] is not None]
    report = {"state": "completed", "scope": "Exact synthetic 1024 development crops, not native final acceptance",
              "code_revision": revision, "dataset_manifest_sha256": manifest_sha, "selected_ids": selected,
              "source_originals": args.original_limit, "cases": len(results), "generations_this_run": additional_generations,
              "generation_replay": str(args.generation_experiment) if args.generation_experiment else None,
              "resolution": args.resolution, "uniform_steps": args.uniform_steps,
              "sr_sha256": args.upscaler_sha256, "sr_precision": args.sr_precision, "sr_tile": args.sr_tile,
              "sr_overlap": args.sr_overlap, "sr_input_source": "generated_RGB_only", "quality_permission": False,
              "generation_phase_seconds": round(generation_phase_seconds, 3),
              "all_generation_seconds_sum": sum(r.get("seconds",0) for r in generated),
              "sr_load_seconds": round(load_seconds, 3), "restoration_phase_seconds": round(time.monotonic()-began, 3),
              "psnr_improved": sum(r["restored"]["psnr"] > r["baseline"]["psnr"] for r in positives),
              "detail_correlation_improved": sum(r["restored_detail"]["highpass_correlation"] > r["baseline_detail"]["highpass_correlation"] for r in comparable),
              "detail_correlation_comparable": len(comparable),
              "mean_psnr_before": float(np.mean([r["baseline"]["psnr"] for r in positives])),
              "mean_psnr_after": float(np.mean([r["restored"]["psnr"] for r in positives])),
              "arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}}
    (work / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    (work / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
