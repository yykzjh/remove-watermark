"""Read-only output-presence ablation on frozen controls, without ground-truth crop hints."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import subprocess
import time
from pathlib import Path


def case_chat(case, resolution):
    from watermark_studio.config import Region
    from watermark_studio.images import load_photo
    from watermark_studio.quality_presence import output_chat
    # Only original source locations are supplied. Synthetic remaining-overlay
    # boxes, expected labels, filenames and QA explanations never enter the chat.
    return output_chat(load_photo(case["source"])[0], load_photo(case["output"])[0],
                       [Region.model_validate(r) for r in case["regions"]], resolution=resolution)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--gpu", type=int, choices=[0, 1, 2, 3], required=True)
    p.add_argument("--resolution", type=int, choices=[768, 1536, 2048, 3072, 4096], default=768)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--adapter-sha256", default="")
    p.add_argument("--contrast", type=float, default=0)
    a = p.parse_args()
    if bool(a.adapter) != bool(a.adapter_sha256) or not 0 <= a.contrast <= 8:
        p.error("Pin both adapter path/SHA and use bounded diagnostic contrast")
    if a.target.exists():
        p.error("Fresh result directory required")
    from watermark_studio.training_reservation import offline_reservation
    reserved = offline_reservation(a.gpu)
    helper = runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    cases = helper["manifest_cases"](a.manifest)
    if not 1 <= len(cases) <= 1000 or len({r["file"] for r in cases}) != len(cases):
        raise ValueError("Unique bounded frozen cases required")
    hashes = helper["file_hashes"](cases)
    manifest_sha = hashlib.sha256(a.manifest.read_bytes()).hexdigest()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    import torch
    from transformers.utils import logging

    from watermark_studio.config import DeploymentConfig
    from watermark_studio.model_identity import compare_model_weights
    from watermark_studio.quality import _structured_vqa
    from watermark_studio.quality_loading import load_quality_encoder, load_quality_processor
    from watermark_studio.quality_presence import (
        FIELDS,
        PROMPT,
        presence_chats,
        raw_presence_metrics,
    )
    from watermark_studio.training_features import digest

    logging.disable_progress_bar()
    torch.set_num_threads(1)
    cache = Path.home()/".cache/watermark-models/models"
    model_path = cache/"Qwen--Qwen3-VL-4B-Instruct/snapshots/master"
    identity = compare_model_weights(cache/"Qwen--Qwen-Image-2.1-Turbo/snapshots/master/text_encoder", model_path)
    if identity["relation"] != "different_weights":
        raise ValueError("Independent inspector required")
    started = time.monotonic()
    model = load_quality_encoder(model_path, DeploymentConfig(quality_verifier=True, quality_backend="independent"), "cuda:0")
    processor = load_quality_processor(model_path)
    if a.adapter:
        from peft import PeftModel
        metadata = json.loads((a.adapter/"presence-task.json").read_text())
        if (digest(a.adapter/"adapter_model.safetensors") != a.adapter_sha256
                or metadata.get("weights_sha256") != a.adapter_sha256
                or metadata.get("task") != "output_only_digital_overlay_presence"
                or metadata.get("prompt_sha256") != hashlib.sha256(PROMPT.encode()).hexdigest()
                or metadata.get("base_identity", {}).get("verifier_sha256") != identity["verifier_sha256"]):
            raise ValueError("Adapter base, prompt or weight mismatch")
        model = PeftModel.from_pretrained(model, a.adapter, is_trainable=False, local_files_only=True).eval()
    load_seconds = time.monotonic()-started
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    a.target.mkdir(parents=True)
    rows = []
    with (a.target/"results.jsonl").open("x") as output:
        for case in cases:
            began = time.monotonic()
            try:
                chats = presence_chats([case_chat(case, a.resolution)], contrast=a.contrast)
                values, _seconds = _structured_vqa(chats, FIELDS, model, processor, "cuda:0", 96)
                quality = {"output_checks": {"presence_predicate": values[0]}}
                error = None
            except Exception as e:  # noqa: BLE001 - errors stay in the denominator
                quality = {}
                error = f"{type(e).__name__}: {e}"[:400]
            row = case | {"quality": quality, "error": error, "seconds": time.monotonic()-began}
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False)+"\n")
            output.flush()
            print(json.dumps({"done": len(rows), "total": len(cases), "errors": sum(bool(r["error"]) for r in rows)}), flush=True)
    if (helper["file_hashes"](cases) != hashes
            or hashlib.sha256(a.manifest.read_bytes()).hexdigest() != manifest_sha
            or revision != subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            or reserved != offline_reservation(a.gpu)):
        raise ValueError("Frozen data, source code or reservation changed")
    errors = sum(bool(r["error"]) for r in rows)
    report = {"state": "completed" if not errors else "completed_with_errors",
              "images": len(rows), "errors": errors, "raw_presence": raw_presence_metrics(rows),
              "seconds": time.monotonic()-started, "load_seconds": load_seconds,
              "resolution": a.resolution, "contrast": a.contrast, "manifest_sha256": manifest_sha,
              "revision": revision, "adapter_sha256": a.adapter_sha256, "weight_identity": identity,
              "source_bytes_unchanged": True, "additional_generations": 0, "production_adopted": False,
              "scope": "Raw output-presence only; not full quality or thousand-image qualification. No ground-truth position hints."}
    (a.target/"report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
