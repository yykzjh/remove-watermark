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
    """Use an ordinary before/after layout; only results are inspected."""
    from watermark_studio.quality import preview
    messages = [dict(message, content=[dict(item) for item in message["content"]]) for message in chat]
    content = messages[-1]["content"]
    if content[0].get("type") != "text" or content[1].get("type") != "image":
        raise ValueError("Unknown result-view layout")
    content[0]["text"] = (
        "图1是未经修改的原照片，仅作实体载体参照，可能有水印也可能没有；"
        "图2是完整RESULT结果照片，其后局部裁切全部来自同一RESULT。只验收RESULT。")
    content.insert(1, {"type": "image", "image": preview(source, 768)})
    content.insert(2, {"type": "text", "text": "图2 RESULT，后续局部也全部为结果："})
    tail = content[-1]
    tail["text"] = tail["text"].replace(
        "图1是完整结果照片，后续图是标明原照片坐标的放大局部。结合整图透视和物体关系区分真实印刷文字。",
        "图1是原照片，图2是完整RESULT，后续图是RESULT局部。只验收RESULT。")
    tail["text"] += (
        "\n只按图2及RESULT局部判断残留，不把图1原照片中的数字覆盖误算为结果残留。"
        "原图仅帮助辨认真实物体的材质、边缘和透视；两边都有同样的文字并不能证明文字属于实物。"
        "只要RESULT仍有任何相机叠加信息卡、文字、细码、底板或残影，watermark_removed=false。"
        "没有水印的原照片可以原样保留，但带水印的原照片直接复制仍是不合格。"
        "检查整张RESULT，不限于上游候选范围，看不清则uncertain=true。")
    return messages


def carrier_context_views(photo, boxes, limit=2):
    from watermark_studio.evidence import valid_box
    from watermark_studio.images import pixel_box
    from watermark_studio.quality import preview
    scopes=[]
    views=[]
    for value in sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True):
        box=valid_box(value)
        if box is None:
            raise ValueError("Invalid OCR carrier-view coordinates")
        l,t,r,b=pixel_box(box,photo.size)
        side=max(288,round(max(r-l,b-t)*2.6))
        cw,ch=min(photo.width,side),min(photo.height,side)
        x=max(0,min(photo.width-cw,round((l+r-cw)/2)))
        y=max(0,min(photo.height-ch,round((t+b-ch)/2)))
        scope=(x,y,x+cw,y+ch)
        if any(max(0,min(scope[2],p[2])-max(scope[0],p[0]))*
               max(0,min(scope[3],p[3])-max(scope[1],p[1]))>.7*cw*ch for p in scopes):
            continue
        scopes.append(scope)
        views.append(preview(photo.crop(scope),512))
        if len(views)>=limit:
            break
    return views,scopes


def carrier_context_chat(chat, before, after, boxes, reference=False):
    from watermark_studio.quality import preview
    messages=[dict(message,content=[dict(item) for item in message["content"]]) for message in chat]
    content=messages[-1]["content"]
    tail=content.pop()
    views,scopes=carrier_context_views(after,boxes)
    for view,scope in zip(views,scopes):
        content.append({"type":"text","text":
            "以下是结果中OCR文字候选周围的完整载体上下文；OCR只定位字形，不证明它是实物或水印。"
            "请看真实材质、实体边缘和透视，数字叠加仍须拒绝，不得因文字被OCR找到而豁免。"})
        if reference:
            content.extend([{"type":"text","text":"同位置ORIGINAL原图局部，仅作原可见内容对照："},
                            {"type":"image","image":preview(before.crop(scope),512)},
                            {"type":"text","text":"对应RESULT结果局部："}])
        content.append({"type":"image","image":view})
    if reference and views:
        tail["text"]+=(
            "\n对上述同位置原图与结果，逐项检查真实标牌、器具、衣物或纸张上原本可见的字符。"
            "若可见实物英文、汉字、数字变成不同字词、乱码或被抹掉，content_preserved=false。"
            "候选框不授权重写真实印字；只允许删除确认为后期数字叠加的字。看不清则uncertain=true。")
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
    parser.add_argument("--presence-predicate", action="store_true")
    parser.add_argument("--presence-contrast",type=float,default=0)
    parser.add_argument("--presence-adapter", type=Path)
    parser.add_argument("--presence-adapter-sha256", default="")
    parser.add_argument("--output-source-reference", action="store_true")
    parser.add_argument("--output-carrier-context", action="store_true")
    parser.add_argument("--inspector-profile", choices=["internvl8b", "qwen4b"], default="internvl8b")
    parser.add_argument("--inspection-profile", choices=["legacy", "defects"], default="legacy")
    args = parser.parse_args()
    if args.presence_contrast and not args.presence_predicate:
        parser.error("Diagnostic contrast applies only to the dedicated predicate")
    if args.presence_predicate and (args.inspector_profile!="qwen4b" or args.output_source_reference or args.output_carrier_context):
        parser.error("Dedicated-predicate pilot requires isolated Qwen4B base roles")
    if bool(args.presence_adapter)!=bool(args.presence_adapter_sha256) or (args.presence_adapter and not args.presence_predicate):
        parser.error("Pin adapter SHA256 and explicitly enable the dedicated predicate")
    if args.output_source_reference and args.output_carrier_context:
        parser.error("Independent evidence-layout ablations cannot be combined")
    if (args.output_source_reference or args.output_carrier_context or args.presence_predicate) and args.inspection_profile != "legacy":
        parser.error("Evidence-layout probes require the unchanged legacy predicates")
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
    text_presence=None
    if args.output_carrier_context:
        from watermark_studio.text_presence import TargetTextPresence
        text_presence=TargetTextPresence()
        text_presence._engine()
    cfg = DeploymentConfig(quality_verifier=True, quality_backend="independent")
    began = time.monotonic()
    inspector = load_quality_encoder(inspector_path, cfg, "cuda:0")
    inspector_processor = load_quality_processor(inspector_path)
    if args.presence_adapter:
        import hashlib

        from peft import PeftModel

        from watermark_studio.quality_presence import PROMPT
        metadata=json.loads((args.presence_adapter/"presence-task.json").read_text())
        weights=args.presence_adapter/"adapter_model.safetensors"
        expected=args.presence_adapter_sha256
        if (hashlib.sha256(weights.read_bytes()).hexdigest()!=expected or
                metadata.get("weights_sha256")!=expected or
                metadata.get("task")!="output_only_digital_overlay_presence" or
                metadata.get("prompt_sha256")!=hashlib.sha256(PROMPT.encode()).hexdigest() or
                metadata.get("base_identity",{}).get("verifier_sha256")!=identities["inspector"]["verifier_sha256"]):
            raise ValueError("Presence adapter/base/protocol identity mismatch")
        inspector=PeftModel.from_pretrained(inspector,args.presence_adapter,is_trainable=False,local_files_only=True)
        inspector.base_model.disable_adapter_layers()
        inspector.eval()
    locator = inspector if inspector_path == locator_path else load_quality_encoder(locator_path, cfg, "cuda:0")
    locator_processor = inspector_processor if locator is inspector else load_quality_processor(locator_path)
    load_seconds = time.monotonic()-began
    counters = collections.Counter()
    original_detect = detection.locate_watermarks_batch
    detection.locate_watermarks_batch = detection_role(
        original_detect, inspector, locator, locator_processor, counters, args.residual_resolution)
    original_inspection = quality._structured_vqa
    source_context = {}
    if args.output_source_reference or args.output_carrier_context or args.presence_predicate:
        def source_inspection(chats, fields, encoder, processor, device, max_tokens):
            if args.output_source_reference and fields[0] == "watermark_removed":
                if len(chats) != 1:
                    raise ValueError("Source-reference pilot requires fixed single-case correspondence")
                chats = [source_aware_output_chat(chats[0], source_context["image"], source_context["regions"])]
            if args.output_carrier_context and fields[0] in {"watermark_removed","content_preserved"}:
                if len(chats)!=1:
                    raise ValueError("Carrier pilot requires exact source/output correspondence")
                chats=[carrier_context_chat(chats[0],source_context["image"],source_context["output"],
                    source_context["carrier_boxes"],reference=fields[0]=="content_preserved")]
            values,seconds=original_inspection(chats,fields,encoder,processor,device,max_tokens)
            if args.presence_predicate and fields[0]=="watermark_removed":
                from watermark_studio.quality_presence import (
                    FIELDS,
                    merge_presence_checks,
                    presence_chats,
                )
                if args.presence_adapter:
                    encoder.base_model.enable_adapter_layers()
                try:
                    predicted,extra=original_inspection(presence_chats(chats,contrast=args.presence_contrast),FIELDS,encoder,processor,device,min(96,max_tokens))
                finally:
                    if args.presence_adapter:
                        encoder.base_model.disable_adapter_layers()
                values=[merge_presence_checks(base,prediction) for base,prediction in zip(values,predicted,strict=True)]
                seconds+=extra
            return values,seconds
        quality._structured_vqa = source_inspection
    rows = []
    began = time.monotonic()
    try:
        with (args.target/"results.jsonl").open("w") as output:
            for case in cases:
                started = time.monotonic()
                pair = (load_photo(case["source"])[0], load_photo(case["output"])[0])
                source_context.update(image=pair[0], regions=[Region.model_validate(r) for r in case["regions"]])
                carrier_evidence={}
                try:
                    if text_presence is not None:
                        carrier_evidence=text_presence.inspect(pair[1],[[0.,0.,1.,1.]])
                        if carrier_evidence["status"]=="error":
                            raise ValueError("Carrier OCR evidence unavailable: "+str(carrier_evidence))
                        boxes=[b for r in carrier_evidence["regions"] for b in r["textlike_boxes"]]
                        source_context.update(output=pair[1],carrier_boxes=boxes)
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
                rows.append(case | {"carrier_evidence":carrier_evidence,"quality": verdict, "seconds": round(time.monotonic()-started, 4)})
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
    from watermark_studio.quality_presence import raw_presence_metrics
    errors = sum(row["quality"]["status"] == "error" for row in rows)
    report = {"state": "completed" if not errors else "completed_with_errors", "revision": revision,
              "images": len(rows), "errors": errors, "seconds": round(time.monotonic()-began, 3),
              "load_seconds": round(load_seconds, 3), "inspection_profile": args.inspection_profile,
              "residual_resolution": args.residual_resolution, "inspector_profile": args.inspector_profile,
              "output_source_reference": args.output_source_reference,
              "presence_predicate":args.presence_predicate,"presence_contrast":args.presence_contrast,
              "presence_adapter":str(args.presence_adapter) if args.presence_adapter else None,
              "presence_adapter_sha256":args.presence_adapter_sha256,
              "raw_presence":raw_presence_metrics(rows),
              "output_carrier_context": args.output_carrier_context,
              "identities": identities, "locator_calls": dict(counters),
              "counts": dict(collections.Counter(r["quality"]["status"] for r in rows)),
              "false_accept_files": [r["file"] for r in rows if r.get("expected") == "review" and r["quality"]["status"] == "passed"],
              "false_refuse_files": [r["file"] for r in rows if r.get("expected") == "passed" and r["quality"]["status"] != "passed"],
              "pixels_changed": False, "additional_generations": 0,
              "scope": ("Offline dedicated-presence role; base naturalness/uncertainty, reference, numeric/source/residual checks unchanged; no image generation"
                        if args.presence_predicate else
                        "Offline QA-only fixed roles; every semantic, residual and numeric check remains mandatory")}
    (args.target/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
