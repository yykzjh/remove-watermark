"""Exact synthetic test of transmitted detail, without any new image generation."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from watermark_studio.exact_metrics import reconstruction_metrics
from watermark_studio.images import load_photo, pixel_box
from watermark_studio.text_presence import TargetTextPresence
from watermark_studio.training_features import digest, validate_pairs
from watermark_studio.transmitted_detail import (
    estimate_patch_transmission,
    neutral_detail_candidates,
    recover_patch_detail,
)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--experiment",type=Path)
    p.add_argument("--mask-only",action="store_true")
    p.add_argument("--strict-neutral-support",action="store_true")
    p.add_argument("--pixel-veto-only",action="store_true")
    p.add_argument("--estimate-transmission",action="store_true")
    p.add_argument("--reference-blur-clean",action="store_true")
    p.add_argument("--target",type=Path,required=True)
    p.add_argument("--transmission",type=float,default=.5)
    p.add_argument("--radius",type=int,default=16)
    p.add_argument("--dark-channel",action="store_true")
    a=p.parse_args()
    if a.pixel_veto_only and not a.strict_neutral_support:
        p.error("Pixel-only support requires the stronger neutral veto")
    if a.reference_blur_clean and (not a.mask_only or not a.estimate_transmission):
        p.error("A synthetic blurred reference is only for mask-only opacity diagnostics")
    if a.estimate_transmission and a.dark_channel:
        p.error("Select one transmission diagnostic")
    if bool(a.experiment)==a.mask_only:
        p.error("Choose completed generation outputs or an explicit mask-only safety diagnostic")
    data,_=validate_pairs(a.manifest);pairs={r["id"]:r for r in data["pairs"]}
    if a.experiment:
        frozen=json.loads((a.experiment/"report.json").read_text())
        if frozen.get("state")!="completed" or frozen["dataset_manifest_sha256"]!=digest(a.manifest):
            p.error("Exact completed whole-frame synthetic validation outputs required")
        outputs=[json.loads(x) for x in (a.experiment/"results.jsonl").read_text().splitlines()]
    else:
        outputs=[{"id":r["id"]} for r in data["pairs"] if r["split"]=="validation"]
    a.target.mkdir(exist_ok=False);work=a.target.resolve();(work/"output").mkdir()
    detector=TargetTextPresence(upsample_small=True);records=[];began=time.monotonic()
    for output in outputs:
        row=pairs[output["id"]]
        if row["split"]!="validation" or (not a.mask_only and
                (digest(output["output"])!=output["output_sha256"] or output["generation_count"]!=1)):
            raise ValueError("Immutable one-generation validation outputs required")
        metadata=row["render"].get("render",row["render"])
        if not metadata.get("panel"):continue
        source=load_photo(row["input"])[0];clean=load_photo(row["target"])[0]
        generated=(clean.filter(ImageFilter.GaussianBlur(2)) if a.reference_blur_clean else clean) if a.mask_only else load_photo(output["output"])[0]
        w,h=source.size
        l,t,r,b=metadata["panel"]
        variant=int(row["id"].rsplit("-",1)[-1])
        if variant==0:t+=max(16,round((b-t)*.25))
        box=(l+4,t+4,r-4,b-4)
        if box[2]-box[0]<24 or box[3]-box[1]<24:continue
        target=[box[0]/w,box[1]/h,box[2]/w,box[3]/h]
        detected=detector.inspect(source,[target])
        trace={"id":row["id"],"source_original_sha256":row["source_original_sha256"],
            "body_box":box,"text_veto":detected,"quality_permission":False,"extra_generations":0}
        records.append(trace)
        boxes=[v for group in detected.get("regions",[]) for v in group.get("textlike_boxes",[]) if v]
        if detected.get("status")!="text_present" or not boxes:
            trace["skip_reason"]="source_body_text_not_safely_localized";continue
        support=Image.new("L",source.size,0);draw=ImageDraw.Draw(support);draw.rectangle((box[0],box[1],box[2]-1,box[3]-1),fill=255)
        if not a.pixel_veto_only:
            for region in boxes:
                x0,y0,x1,y1=pixel_box(region,source.size)
                draw.rectangle((max(0,x0-2),max(0,y0-2),min(w-1,x1+1),min(h-1,y1+1)),fill=0)
        luminance=np.asarray(source.convert("L"));src=np.asarray(source)
        dark=(luminance<(140 if a.strict_neutral_support else 100)).astype(np.uint8)*255
        ink=np.asarray(Image.fromarray(dark).filter(ImageFilter.MaxFilter(9 if a.strict_neutral_support else 5)))>0
        valid=(np.asarray(support)>0)&~ink
        if a.strict_neutral_support:
            valid&=neutral_detail_candidates(source)
            trace["strict_neutral_support"]={"dark_below":140,"bright_above":225,"chroma_above":24,
                "dark_dilation":9,"bright_dilation":5,"chroma_dilation":3,"pixel_veto_only":a.pixel_veto_only}
        alpha=round(metadata["opacity"]*255)
        plate=Image.new("RGBA",source.size,(*metadata["foreground"],alpha))
        expected=Image.alpha_composite(clean.convert("RGBA"),plate).convert("RGB")
        tolerance=1 if a.strict_neutral_support else 3
        glyph=np.any(np.abs(src.astype(float)-np.asarray(expected,dtype=float))>tolerance,axis=2)
        trace["known_foreground_difference_tolerance"]=tolerance
        trace["known_visible_glyph_leak_pixels"]=int((glyph&valid).sum())
        # Ground truth only diagnoses leakage; it never corrects the proposed mask.
        trace["safe_on_known_glyphs"]=trace["known_visible_glyph_leak_pixels"]==0
        cropped=valid[box[1]:box[3],box[0]:box[2]]
        trace["support_fraction"]=float(cropped.mean())
        if cropped.mean()<.1:trace["skip_reason"]="too_little_text_excluded_body";continue
        tvalue=a.transmission
        if a.estimate_transmission:
            inferred=estimate_patch_transmission(source.crop(box),generated.crop(box),cropped.astype(float))
            trace["transmission_estimate"]=inferred
            trace["diagnostic_true_transmission"]=1-alpha/255
            if not inferred["eligible"]:
                trace["skip_reason"]=inferred["reason"];continue
            tvalue=inferred["transmission"]
        if a.mask_only:
            trace["mask_only"]=True;continue
        if a.dark_channel:
            minimum=np.min(src[valid],axis=1)
            tvalue=float(np.clip(1-np.quantile(minimum,.01)/235.,.25,.85))
        patched,audit=recover_patch_detail(source.crop(box),generated.crop(box),cropped.astype(float),
            transmission=tvalue,radius=a.radius,max_delta=24)
        after=generated.copy();after.paste(patched,box)
        trace.update(audit=audit,assumed_transmission=tvalue,true_synthetic_transmission=1-alpha/255,
            before=reconstruction_metrics(np.asarray(generated),np.asarray(clean),valid),
            after=reconstruction_metrics(np.asarray(after),np.asarray(clean),valid))
        dest=work/"output"/(row["id"]+".png");after.save(dest);trace["output"]=str(dest);trace["output_sha256"]=digest(dest)
    measured=[r for r in records if "after" in r]
    report={"state":"completed","version":1,"cases":len(records),"applied":len(measured),
        "glyph_leak_cases":sum(r.get("known_visible_glyph_leak_pixels",0)>0 for r in records),
        "nonempty_safe_support_cases":sum(r.get("support_fraction",0)>=.1 and
            r.get("safe_on_known_glyphs") is True for r in records),
        "mask_only":a.mask_only,
        "before_psnr":statistics.mean(r["before"]["psnr"] for r in measured) if measured else None,
        "after_psnr":statistics.mean(r["after"]["psnr"] for r in measured) if measured else None,
        "seconds":round(time.monotonic()-began,3),"extra_generations":0,"quality_permission":False,
        "scope":"Synthetic known layout; native source support is NOT certified by this experiment",
        "arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}}
    (work/"results.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    (work/"report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
