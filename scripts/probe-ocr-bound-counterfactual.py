"""Offline physical-letter counterfactual inquiry; all actual-output vetoes remain mandatory."""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import os
import runpy
import subprocess
import time
from pathlib import Path

import numpy as np


def inside_fraction(word,box):
    a=word["box"]
    overlap=max(0,min(a[2],box[2])-max(a[0],box[0]))*max(0,min(a[3],box[3])-max(a[1],box[1]))
    return overlap/max((a[2]-a[0])*(a[3]-a[1]),1e-12)


def global_word(word,crop,size):
    left,top,right,bottom=crop
    width,height=size
    if not 0<=left<right<=width or not 0<=top<bottom<=height:
        raise ValueError("OCR crop escaped its original photograph")
    quad=np.asarray(word["quad"],dtype=float);box=np.asarray(word["box"],dtype=float)
    if (quad.shape!=(4,2) or box.shape!=(4,) or not np.isfinite(quad).all() or
        not np.isfinite(box).all() or np.any(quad<0) or np.any(quad>1) or
        not np.allclose(np.r_[quad.min(axis=0),quad.max(axis=0)],box,atol=1e-6)):
        raise ValueError("OCR word and quadrilateral disagree")
    mapped=copy.deepcopy(word)
    mapped["box"]=[float((left+box[0]*(right-left))/width),float((top+box[1]*(bottom-top))/height),
                   float((left+box[2]*(right-left))/width),float((top+box[3]*(bottom-top))/height)]
    mapped["quad"]=[[float((left+x*(right-left))/width),float((top+y*(bottom-top))/height)] for x,y in quad]
    return mapped


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--target",type=Path,required=True)
    p.add_argument("--gpu",type=int,choices=[0,1,2,3],required=True)
    p.add_argument("--presence-adapter",type=Path,required=True)
    p.add_argument("--presence-adapter-sha256",required=True)
    a=p.parse_args()
    assert not a.target.exists()
    from watermark_studio.training_reservation import offline_reservation
    reserved=offline_reservation(a.gpu)
    helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    constraint=runpy.run_path(str(Path(__file__).with_name("probe-quality-specialists.py")))["enforce_frozen_constraints"]
    ocr_helper=runpy.run_path(str(Path(__file__).with_name("prepare-native-ocr-witnesses.py")))
    cases=helper["manifest_cases"](a.manifest);hashes=helper["file_hashes"](cases)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
    import torch
    from peft import PeftModel
    from PIL import Image, ImageDraw, ImageFilter
    from rapidocr_onnxruntime import RapidOCR
    from transformers.utils import logging

    from watermark_studio import cross_review, detection
    from watermark_studio.config import DeploymentConfig, Region
    from watermark_studio.images import load_photo, pixel_box
    from watermark_studio.model_identity import compare_model_weights
    from watermark_studio.quality import _structured_vqa, preview, verify_batch
    from watermark_studio.quality_loading import load_quality_encoder, load_quality_processor
    from watermark_studio.quality_presence import FIELDS, PROMPT, output_chat
    from watermark_studio.text_preservation import box_iou
    from watermark_studio.training_features import digest

    torch.set_num_threads(1);logging.disable_progress_bar()
    cache=Path.home()/".cache/watermark-models/models"
    path=cache/"Qwen--Qwen3-VL-4B-Instruct/snapshots/master"
    editor_path=cache/"Qwen--Qwen3-VL-8B-Instruct/snapshots/master"
    identity=compare_model_weights(editor_path,path)
    assert identity["relation"]=="different_weights"
    cfg=DeploymentConfig(quality_verifier=True,quality_backend="independent")
    model=load_quality_encoder(path,cfg,"cuda:0");processor=load_quality_processor(path)
    editor=load_quality_encoder(editor_path,cfg,"cuda:0");editor_processor=load_quality_processor(editor_path)
    metadata=json.loads((a.presence_adapter/"presence-task.json").read_text())
    if (digest(a.presence_adapter/"adapter_model.safetensors")!=a.presence_adapter_sha256 or
        metadata["weights_sha256"]!=a.presence_adapter_sha256 or
        metadata.get("task")!="output_only_digital_overlay_presence" or
        metadata["prompt_sha256"]!=hashlib.sha256(PROMPT.encode()).hexdigest() or
        metadata["base_identity"]["verifier_sha256"]!=identity["verifier_sha256"]):
        raise ValueError("Dedicated predicate base/protocol/artifact mismatch")
    adapter_controller=PeftModel.from_pretrained(model,a.presence_adapter,is_trainable=False,local_files_only=True).eval()
    adapter_controller.base_model.disable_adapter_layers()
    # Keep the original generation API for every base inspection; only linear
    # adapter activation changes during the dedicated actual-output predicate.
    model=adapter_controller.get_base_model()
    engine=RapidOCR(det_use_cuda=False,rec_use_cuda=False,cls_use_cuda=False,
        intra_op_num_threads=1,inter_op_num_threads=1,det_limit_type="max",
        det_limit_side_len=4096,max_side_len=4096,det_box_thresh=.5,use_cls=False,use_rec=True)
    current={};original_reinspect=cross_review._reinspect;original_detect=detection.locate_watermarks_batch
    def native_words(before,after,native,box):
        local=[ocr_helper["scan"](engine,photo.crop(native)) for photo in (before,after)]
        words=[]
        for word in local[0]:
            if word["confidence"]<.9 or word["characters"]<2 or word["app_marker"]:continue
            matches=[w for w in local[1] if w["confidence"]>=.9 and
                w["transcript_sha256"]==word["transcript_sha256"] and box_iou(w["box"],word["box"])>=.7]
            if not matches:continue
            mapped=global_word(word,native,after.size)
            pixel=cross_review.preserved_patch(before,after,mapped["box"])
            if pixel["unchanged"] and inside_fraction(mapped,box)>=.9:
                words.append(mapped|{"pixels":pixel})
        return words

    def reinspect(before,after,boxes,encoder,proc,device,resolution,max_tokens,**_options):
        selected=[]
        audit=current["audit"]
        for box in boxes:
            from watermark_studio.images import context_box
            native=context_box(pixel_box(box,after.size),after.size,.15)
            words=native_words(before,after,native,box)
            if not words:
                audit["gate"]="missing_readable_unchanged_native_word"
                return {"watermark_removed":False,"natural":True,"uncertain":True,"area":"none"}
            selected.extend(words)
        if not 1<=len(selected)<=12:
            audit["gate"]="unbounded_native_word_count"
            return {"watermark_removed":False,"natural":True,"uncertain":True,"area":"none"}
        mask=Image.new("L",after.size);draw=ImageDraw.Draw(mask)
        for word in selected:
            draw.polygon([(round(x*after.width),round(y*after.height)) for x,y in word["quad"]],fill=255)
        mask=mask.filter(ImageFilter.MaxFilter(5))
        fraction=float((np.asarray(mask)>0).mean())
        if fraction>.02:
            audit["gate"]="counterfactual_word_area_exceeded"
            return {"watermark_removed":False,"natural":True,"uncertain":True,"area":"none"}
        diagnostic=after.copy()
        for word in selected:
            native=pixel_box(word["box"],after.size)
            radius=max(8.,(native[3]-native[1])*.75)
            localmask=Image.new("L",after.size)
            ImageDraw.Draw(localmask).polygon([(round(x*after.width),round(y*after.height)) for x,y in word["quad"]],fill=255)
            localmask=localmask.filter(ImageFilter.MaxFilter(5))
            diagnostic=Image.composite(after.filter(ImageFilter.GaussianBlur(radius)),diagnostic,localmask)
        audit.update(gate="native_word_and_dual_carrier_proofs",words=selected,
            mask_fraction=fraction,actual_output_pixels_changed=False,
            naturalness_source="Frozen actual-output base verdict, never the diagnostic blur")
        # Inspect the ACTUAL output with a separate learned predicate before any
        # counterfactual decision. It can veto; it cannot approve by itself.
        adapter_controller.base_model.enable_adapter_layers()
        try:
            actual_chat=output_chat(before,after,current["regions"])
            from watermark_studio.images import context_box
            for box in boxes:
                native=context_box(pixel_box(box,after.size),after.size,1.)
                actual_chat[0]["content"][-1:-1]=[
                    {"type":"text","text":"同一真实输出的原生文字局部；检查实物文字上是否另叠加淡字、代码或底纹。裁切边缘不是水印。"},
                    {"type":"image","image":preview(after.crop(native),768)}]
            pred,_=_structured_vqa([actual_chat],FIELDS,model,processor,"cuda:0",96)
        finally:adapter_controller.base_model.disable_adapter_layers()
        audit["actual_presence"]=pred[0]
        if pred[0]["overlay_present"] or pred[0]["uncertain"]:
            return {"watermark_removed":False,"natural":True,"uncertain":True,"area":pred[0]["area"]}
        content=[{"type":"text","text":"这是一张检验专用辅助图，不是输出照片。"},
                 {"type":"image","image":preview(diagnostic,resolution)}]
        for box in boxes:
            content.extend([{"type":"text","text":"同一辅助图的原生局部，范围"+str(box)},
                {"type":"image","image":preview(diagnostic.crop(pixel_box(box,after.size)),512)}])
        content.append({"type":"text","text":
            "工具仅模糊了经两个不同权重模型确认有实物载体、且原图和输出OCR读字与像素对应一致的实物印字。"
            "真实输出未模糊；这些字的模糊仅用于消除实物文字干扰，不能豁免任何其他区域。"
            "遍查整图其余像素，数字信息卡、相机标识、时间地点、字母代码、透明底板、淡字和碎片任一存在，"
            "watermark_removed=false。工具模糊不是数字水印；其余位置看不清或类别不明确时uncertain=true。"
            "不推测被模糊文字，不评价工具模糊的自然度，实际输出自然度已经在独立原图检验中保留。"
            "只返回JSON：watermark_removed、uncertain为布尔值；area为"
            "none/top_left/top/top_right/left/center/right/bottom_left/bottom/bottom_right。"})
        checked,_=_structured_vqa([[{"role":"user","content":content}]],
            ("watermark_removed","uncertain"),model,processor,"cuda:0",128)
        audit["counterfactual_presence"]=checked[0]
        if checked[0]["watermark_removed"] and not checked[0]["uncertain"]:
            current["diagnostic"]=diagnostic
        return checked[0]|{"natural":current["base"]["natural"]}
    def seed_words(before,after,value):
        area=(value.get("output_checks") or {}).get("area","none")
        if (value.get("output_checks") or {}).get("watermark_removed") is not False:
            area=value.get("area","none")
        positions={"top_left":(0,0),"top":(1,0),"top_right":(2,0),
            "left":(0,1),"center":(1,1),"right":(2,1),
            "bottom_left":(0,2),"bottom":(1,2),"bottom_right":(2,2)}
        if area not in positions:return []
        x,y=positions[area]
        box=[max(0,x/3-.03),max(0,y/3-.03),min(1,(x+1)/3+.03),min(1,(y+1)/3+.03)]
        words=native_words(before,after,pixel_box(box,after.size),box)
        eligible=[]
        for word in words:
            proposal={"kind":"panel","box":word["box"]}
            found,reason=cross_review.candidate_boxes(value|{"residual_regions":[proposal]},after.size,12)
            if found and not reason:eligible.append(word)
        eligible.sort(key=lambda w:(w["box"][2]-w["box"][0])*(w["box"][3]-w["box"][1]),reverse=True)
        current["audit"]["native_ocr_seed_words"]=eligible[:2]
        current["audit"]["seed_semantics"]="Localization hypotheses only, never a digital/physical decision"
        return [{"kind":"panel","box":w["box"]} for w in eligible[:2]]

    def locate(photos,encoder,proc,device,*args,**kwargs):
        if current.get("diagnostic") is not None and len(photos)==1 and photos[0] is current.get("after"):
            current["audit"]["residual_scan_view"]="Only proven physical OCR glyphs blurred; all other pixels retained"
            return original_detect([current["diagnostic"]],encoder,proc,device,*args,**kwargs)
        return original_detect(photos,encoder,proc,device,*args,**kwargs)
    cross_review._reinspect=reinspect;detection.locate_watermarks_batch=locate
    a.target.mkdir(parents=True);rows=[];began=time.monotonic()
    revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    try:
        with (a.target/"results.jsonl").open("x") as log:
            for case in cases:
                before,after=[load_photo(case[k])[0] for k in ("source","output")]
                if before.size!=after.size:raise ValueError("Native counterfactual requires unchanged output size")
                regions=[Region.model_validate(r) for r in case["regions"]]
                current.clear();current.update(after=after,regions=regions,audit={})
                value=verify_batch([(before,after)],model,processor,"cuda:0",resolution=768,
                    detail_patches=4,detection_batch_size=2,source_regions=[regions],
                    editor_context=[helper["case_context"](case)],cross_review_allowed=[False],
                    reviewer="ocr_bound_counterfactual",weights_relation=identity["relation"],rectangle_guard=True)[0]
                value=constraint(case,value);baseline=copy.deepcopy(value)
                current["base"]=baseline
                if (value.get("status")=="review" and value.get("watermark_removed") is False and
                    value.get("natural") is True and value.get("content_preserved") is True and
                    value.get("uncertain") is False and not value.get("guard_codes") and
                    not case.get("frozen_vetoes") and regions):
                    inquiry=copy.deepcopy(value);started_inquiry=time.monotonic()
                    if not inquiry.get("residual_regions"):
                        seeded=seed_words(before,after,inquiry)
                        if seeded:inquiry["residual_regions"]=seeded
                    seeding_seconds=time.monotonic()-started_inquiry
                    inquiry["seconds"]=round(inquiry.get("seconds",0)+seeding_seconds,3)
                    cross_review.reconsider(before,after,inquiry,editor,editor_processor,model,processor,
                        "cuda:0",weights_relation=identity["relation"],full_context=True)
                    if inquiry.get("status")=="passed":
                        value=inquiry
                    else:
                        # Hypothesized real lettering must never become a digital
                        # residual location supplied to destructive rework.
                        value["cross_review"]=inquiry.get("cross_review",{})
                        value["seconds"]=inquiry.get("seconds",value.get("seconds",0))
                        if inquiry.get("retry_recheck_only"):value["retry_recheck_only"]=True
                value=constraint(case,value);value["ocr_bound_counterfactual"]=current["audit"]
                row=case|{"baseline_quality":baseline,"quality":value}
                rows.append(row);log.write(json.dumps(row,ensure_ascii=False)+chr(10));log.flush()
                if len(rows)%4==0:print(json.dumps({"done":len(rows),"counts":dict(collections.Counter(r["quality"]["status"] for r in rows))}),flush=True)
    finally:
        cross_review._reinspect=original_reinspect;detection.locate_watermarks_batch=original_detect
    assert helper["file_hashes"](cases)==hashes
    assert revision==subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    assert reserved==offline_reservation(a.gpu)
    oldbad={r["file"] for r in rows if r.get("expected")=="review" and r["baseline_quality"]["status"]=="passed"}
    newbad={r["file"] for r in rows if r.get("expected")=="review" and r["quality"]["status"]=="passed"}
    report={"state":"completed_offline_experiment","images":len(rows),"seconds":time.monotonic()-began,
        "baseline_counts":dict(collections.Counter(r["baseline_quality"]["status"] for r in rows)),
        "counts":dict(collections.Counter(r["quality"]["status"] for r in rows)),
        "added_false_accept_files":sorted(newbad-oldbad),"false_accept_files":sorted(newbad),
        "false_refuse_files":[r["file"] for r in rows if r.get("expected")=="passed" and r["quality"]["status"]!="passed"],
        "revision":revision,"source_bytes_unchanged":True,"additional_generations":0,"production_adopted":False,
        "scope":"Experimental physically-proven OCR-glyph counterfactual inspection; original natural/content/numeric/uncertainty vetoes retained"}
    (a.target/"report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
