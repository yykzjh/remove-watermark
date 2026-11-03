"""Read-only fixed semantic-inspector / residual-locator comparison."""
from __future__ import annotations

import argparse
import collections
import json
import os
import runpy
import subprocess
import time
from pathlib import Path


def detection_role(original, inspector, locator, locator_processor, counters, resolution=0):
    def routed(images, encoder, processor, device, *args, **kwargs):
        if encoder is inspector:
            counters["calls"] += 1
            counters["images"] += len(images)
            if resolution:
                if args:
                    args = (max(resolution, args[0]), *args[1:])
                else:
                    kwargs = dict(kwargs, resolution=max(resolution, kwargs.get("resolution", 0)))
            return original(images, locator, locator_processor, device, *args, **kwargs)
        return original(images, encoder, processor, device, *args, **kwargs)
    return routed


def source_aware_output_chat(chat, source, regions):
    """Extra original evidence is never a conclusion or an output image."""
    from watermark_studio.images import context_box, pixel_box
    from watermark_studio.quality import preview
    messages = [dict(message, content=[dict(item) for item in message["content"]]) for message in chat]
    content = messages[-1]["content"]
    tail = content.pop()
    content.append({"type": "text", "text":
        "SOURCE REFERENCE ONLY: The following images are the untouched ORIGINAL, not results. "
        "The ORIGINAL still has its overlays. Their presence in these references is not a result defect. "
        "Use the original only to establish real object boundaries, perspective and visible physical printing. "
        "A character being present in both pictures does NOT prove it is physical."})
    content.append({"type": "image", "image": preview(source, 768)})
    for region in sorted(regions, key=lambda r: (r.box[2]-r.box[0])*(r.box[3]-r.box[1]), reverse=True)[:2]:
        box = context_box(pixel_box(region.box, source.size), source.size, .45)
        content.extend([{"type": "text", "text": "ORIGINAL reference crop; this is NOT a result detail."},
                        {"type": "image", "image": preview(source.crop(box), 512)}])
    tail["text"] += (
        "\n最后标有SOURCE REFERENCE/ORIGINAL的图像仅为未经修改的原图参照，仍带输入水印，"
        "不能将它们当作结果照片。watermark_removed只按前面的完整结果和结果细节判断；"
        "原图只帮助辨认物体载体。文字两边都有并不证明实物：水印完全没动的结果必须判false。"
        "真实物体上的印字必须有材质、实体边缘或透视等可见证据；"
        "相机叠加卡片、日期、标识、细码和淡字即使与原图一致仍须判残留。"
        "检查整张结果，不限于原图候选范围，无法辨别则uncertain=true。")
    content.append(tail)
    return messages


def enforce_frozen_constraints(case, verdict):
    if case.get("frozen_vetoes"):
        if verdict["status"] != "error":
            verdict["status"] = "review"
        verdict["guard_codes"] = sorted(set(verdict.get("guard_codes", []) + case["frozen_vetoes"]))
        verdict["reason"] = (verdict.get("reason", "") + " 保留冻结候选的不可豁免约束。").strip()
    return verdict


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=[0, 1, 2, 3], required=True)
    parser.add_argument("--residual-resolution", type=int, choices=[0, 768, 1024, 1536, 2048], default=0)
    parser.add_argument("--output-source-reference", action="store_true")
    parser.add_argument("--inspector-profile", choices=["internvl8b", "qwen4b"], default="internvl8b")
    parser.add_argument("--inspection-profile", choices=["legacy", "defects"], default="legacy")
    args = parser.parse_args()
    from watermark_studio.training_reservation import offline_reservation
    reserved = offline_reservation(args.gpu)
    helper = runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    cases = [case for path in args.manifests for case in helper["manifest_cases"](path)]
    if len(cases) != len({case["file"] for case in cases}) or not 1 <= len(cases) <= 1000:
        raise ValueError("Unique bounded frozen cases required")
    args.target.mkdir(parents=True, exist_ok=False)
    original_hashes = helper["file_hashes"](cases)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    import torch
    from transformers.utils import logging

    from watermark_studio import detection, quality
    from watermark_studio.config import DeploymentConfig, Region
    from watermark_studio.images import load_photo
    from watermark_studio.model_identity import compare_model_weights
    from watermark_studio.quality import verify_batch
    from watermark_studio.quality_loading import load_quality_encoder, load_quality_processor

    logging.disable_progress_bar()
    torch.set_num_threads(1)
    models = Path.home()/".cache/watermark-models/models"
    locator_path = models/"Qwen--Qwen3-VL-4B-Instruct/snapshots/master"
    inspector_path = (models/"OpenGVLab--InternVL3_5-8B-HF/snapshots/master"
                      if args.inspector_profile == "internvl8b" else locator_path)
    editor_path = models/"Qwen--Qwen-Image-2.1-Turbo/snapshots/master/text_encoder"
    identities = {"inspector": compare_model_weights(editor_path, inspector_path),
                  "locator": compare_model_weights(editor_path, locator_path),
                  "inspector_locator": compare_model_weights(inspector_path, locator_path)}
    if any(identities[role]["relation"] != "different_weights" for role in ("inspector", "locator")):
        raise ValueError("Both QA roles must use weights independent of the image editor")
    if args.inspector_profile == "internvl8b" and identities["inspector_locator"]["relation"] != "different_weights":
        raise ValueError("Distinct specialist profiles require different weights")
    cfg = DeploymentConfig(quality_verifier=True, quality_backend="independent")
    began = time.monotonic()
    inspector = load_quality_encoder(inspector_path, cfg, "cuda:0")
    inspector_processor = load_quality_processor(inspector_path)
    locator = inspector if inspector_path == locator_path else load_quality_encoder(locator_path, cfg, "cuda:0")
    locator_processor = inspector_processor if locator is inspector else load_quality_processor(locator_path)
    load_seconds = time.monotonic()-began
    counters = collections.Counter()
    original_detect = detection.locate_watermarks_batch
    detection.locate_watermarks_batch = detection_role(
        original_detect, inspector, locator, locator_processor, counters, args.residual_resolution)
    original_inspection = quality._structured_vqa
    source_context = {}
    if args.output_source_reference:
        def source_inspection(chats, fields, *args, **kwargs):
            if fields[0] == "watermark_removed":
                if len(chats) != 1:
                    raise ValueError("Source-reference pilot requires fixed single-case correspondence")
                chats = [source_aware_output_chat(chats[0], source_context["image"], source_context["regions"])]
            elif fields[0] == "content_preserved":
                chats = [[dict(message, content=[dict(item) for item in message["content"]])
                          for message in chat] for chat in chats]
                for chat in chats:
                    chat[-1]["content"][-1]["text"] += (
                        "\n上游候选框不是已经证明的不透明水印。"
                        "先独立核实原图可见文字是否依附真实物体；"
                        "真实标牌、机械、衣物或纸张上原本可读的字符须逐项与结果比对，"
                        "变字、伪字、乱码或抹掉一律content_preserved=false，"
                        "即使该文字位于上游候选范围内也不豁免。"
                        "仅有后期数字叠加的文字可以删除，不能把真实印字当数字水印。"
                        "看不清是否相同则uncertain=true，不能因为两边都有字就称内容一致。")
            return original_inspection(chats, fields, *args, **kwargs)
        quality._structured_vqa = source_inspection
    rows = []
    began = time.monotonic()
    try:
        with (args.target/"results.jsonl").open("w") as output:
            for case in cases:
                started = time.monotonic()
                pair = (load_photo(case["source"])[0], load_photo(case["output"])[0])
                source_context.update(image=pair[0], regions=[Region.model_validate(r) for r in case["regions"]])
                try:
                    verdict = verify_batch(
                        [pair], inspector, inspector_processor, "cuda:0", resolution=768,
                        detail_patches=4, detection_batch_size=2,
                        source_regions=[[Region.model_validate(r) for r in case["regions"]]],
                        editor_context=[helper["case_context"](case)],
                        cross_review_allowed=[False], reviewer="fixed_role_specialists",
                        weights_relation=identities["inspector"]["relation"],
                        rectangle_guard=True, inspection_profile=args.inspection_profile)[0]
                    verdict = enforce_frozen_constraints(case, verdict)
                    torch.cuda.synchronize()
                except Exception as error:  # noqa: BLE001 - errors remain in the fixed denominator
                    verdict = {"status": "error", "reason": f"{type(error).__name__}: {error}"[:400]}
                rows.append(case | {"quality": verdict, "seconds": round(time.monotonic()-started, 4)})
                output.write(json.dumps(rows[-1], ensure_ascii=False)+"\n")
                output.flush()
                if len(rows)%4 == 0:
                    print(json.dumps({"done": len(rows), "total": len(cases),
                                      "counts": dict(collections.Counter(r["quality"]["status"] for r in rows))}), flush=True)
    finally:
        detection.locate_watermarks_batch = original_detect
        quality._structured_vqa = original_inspection
    if helper["file_hashes"](cases) != original_hashes:
        raise ValueError("Frozen source/candidate bytes changed")
    if revision != subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip():
        raise ValueError("Frozen source code changed")
    if reserved != offline_reservation(args.gpu):
        raise ValueError("Reserved GPU/API state changed")
    errors = sum(row["quality"]["status"] == "error" for row in rows)
    report = {"state": "completed" if not errors else "completed_with_errors", "revision": revision,
              "images": len(rows), "errors": errors, "seconds": round(time.monotonic()-began, 3),
              "load_seconds": round(load_seconds, 3), "inspection_profile": args.inspection_profile,
              "residual_resolution": args.residual_resolution, "inspector_profile": args.inspector_profile,
              "output_source_reference": args.output_source_reference,
              "identities": identities, "locator_calls": dict(counters),
              "counts": dict(collections.Counter(r["quality"]["status"] for r in rows)),
              "false_accept_files": [r["file"] for r in rows if r.get("expected") == "review" and r["quality"]["status"] == "passed"],
              "false_refuse_files": [r["file"] for r in rows if r.get("expected") == "passed" and r["quality"]["status"] != "passed"],
              "pixels_changed": False, "additional_generations": 0,
              "scope": "Offline QA-only fixed roles; every semantic, residual and numeric check remains mandatory"}
    (args.target/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
