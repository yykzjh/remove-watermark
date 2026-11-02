"""Augment a screened pilot with freshly proved lower-frame crops; preserve parent source splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
import urllib.request
from pathlib import Path

from PIL import Image

from watermark_studio.training_features import clone_pair_assets, digest, validate_pairs


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--parent-manifest",type=Path,required=True)
    p.add_argument("--proposals",type=Path,required=True)
    p.add_argument("--target",type=Path,required=True)
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--train-originals",type=int,default=32)
    p.add_argument("--validation-originals",type=int,default=4)
    p.add_argument("--resolution",type=int,choices=[768,1024],default=1024)
    p.add_argument("--font",type=Path,default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    p.add_argument("--seed",type=int,default=2026110207)
    a=p.parse_args();parent,parent_files=validate_pairs(a.parent_manifest)
    if (a.gpu<0 or not 4<=a.train_originals<=48 or not 4<=a.validation_originals<=12 or
            len(parent["pairs"])+4*(a.train_originals+a.validation_originals)>640):
        p.error("Bounded disjoint-GPU augmentation within 640 pairs required")
    if digest(a.font)!=parent["font_sha256"]:p.error("Exact parent font required")
    parent_sha=digest(a.parent_manifest);proposal_sha=digest(a.proposals)
    proposals=json.loads(a.proposals.read_text())
    if proposals.get("version")!=1:p.error("Versioned geometry proposals required")
    parents={}
    for row in parent["pairs"]:
        previous=parents.setdefault(row["source_original_sha256"],row)
        if previous["split"]!=row["split"]:p.error("Parent source splits conflict")
    candidates=proposals["cases"]
    if len({r["source_original_sha256"] for r in candidates})!=len(candidates):
        p.error("One candidate per known original required")
    for c in candidates:
        known=parents.get(c["source_original_sha256"])
        if known is None or c["split"]!=known["split"] or Path(c["source"]).resolve()!=Path(known["source_original"]).resolve():
            p.error("Geometry proposal must preserve its exact parent source and split")
    candidates.sort(key=lambda c:hashlib.sha256((str(a.seed)+c["source_original_sha256"]).encode()).hexdigest())
    a.target.mkdir(exist_ok=False);work=a.target.resolve();(work/"lower-targets").mkdir()
    (work/"lower-pairs").mkdir();(work/"lower-masks").mkdir()
    forbidden=(a.parent_manifest.resolve().parent/"forbidden-sha256.json").read_bytes()
    (work/"forbidden-sha256.json").write_bytes(forbidden)
    pairs=clone_pair_assets(parent["pairs"],work)
    selection={r["source_original"]:r["source_original_sha256"] for r in parent["pairs"]}
    (work/"source-selection.json").write_text(json.dumps({"source_sha256":selection,
        "purpose":"Derived training data only; parent source roles retained, not independent evaluation"}))
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
    import runpy

    import torch
    from transformers import Qwen3VLProcessor

    from watermark_studio.batching import dimensions
    from watermark_studio.config import DeploymentConfig
    from watermark_studio.detection import locate_watermarks_batch
    from watermark_studio.images import load_photo
    from watermark_studio.model_identity import compare_model_weights
    from watermark_studio.quality_loading import load_quality_encoder, load_quality_processor
    from watermark_studio.text_presence import TargetTextPresence
    from watermark_studio.training_pairs import render_profile_pair
    helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as response:
        reserved=helper["require_disjoint_server_gpus"](json.load(response),[a.gpu],True)
    model_root=Path.home()/".cache/watermark-models/models"
    eroot=model_root/"Qwen--Qwen-Image-2.1-Turbo/snapshots/master"
    vroot=model_root/"Qwen--Qwen3-VL-4B-Instruct/snapshots/master"
    identity=compare_model_weights(eroot/"text_encoder",vroot)
    if identity["relation"]!="different_weights":raise ValueError("Independent crop screening weights required")
    torch.set_num_threads(1);began=time.monotonic()
    cfg=DeploymentConfig(quality_verifier=True,quality_backend="independent",quality_quantization="none")
    editor=load_quality_encoder(eroot/"text_encoder",cfg,"cuda:0")
    ep=Qwen3VLProcessor.from_pretrained(eroot/"processor",local_files_only=True)
    verifier=load_quality_encoder(vroot,cfg,"cuda:0");vp=load_quality_processor(vroot);ocr=TargetTextPresence()
    load_seconds=time.monotonic()-began;began=time.monotonic()
    need={"train":a.train_originals,"validation":a.validation_originals};accepted={"train":0,"validation":0};scanned=0
    revision=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    with (work/"screening.jsonl").open("x") as output:
        for case in candidates:
            split=case["split"]
            if accepted[split]>=need[split]:continue
            record={**case,"accepted":False,"role":"additional_training_crop_not_independent_source"}
            try:
                if digest(case["source"])!=case["source_original_sha256"]:raise ValueError("Parent original bytes changed")
                original=load_photo(case["source"])[0];box=tuple(case["box"]);l,t,r,b=box
                if (any(type(v) is not int for v in box) or not 0<=l<r<=original.width or not 0<=t<b<=original.height
                        or (r-l)*(b-t)<.12*original.width*original.height or (t+b)/(2*original.height)<.55):
                    raise ValueError("Lower native proposal geometry invalid")
                clean=original.crop(box);width,height=dimensions(clean,a.resolution)
                if width>clean.width or height>clean.height:raise ValueError("Upsampled target refused")
                crop_checks=[]
                for model,processor in ((editor,ep),(verifier,vp)):
                    result=locate_watermarks_batch([clean],model,processor,"cuda:0",1280,1)[0]
                    if isinstance(result,Exception) or result[0]:raise ValueError("Fresh crop localization not clear")
                    crop_checks.append(result[1])
                text=ocr.inspect(clean,[[0.,0.,1.,1.]])
                if text.get("status")!="clear":raise ValueError("Fresh crop OCR not clear")
                target=clean.resize((width,height),Image.Resampling.LANCZOS).convert("RGB")
                target_path=work/"lower-targets"/(case["source_original_sha256"][:24]+".png");target.save(target_path)
                added=[]
                for variant in range(4):
                    seed=(a.seed+int(case["source_original_sha256"][:16],16)+variant)%(2**32)
                    image,mask,render=render_profile_pair(target,variant,random.Random(seed),a.font,profile="camera-v2")
                    name=case["source_original_sha256"][:24]+"-lower"+str(a.resolution)+"-"+str(variant)
                    ip=work/"lower-pairs"/(name+".png");mp=work/"lower-masks"/(name+".png");image.save(ip);mask.save(mp)
                    added.append({"id":name,"split":split,"input":str(ip),"input_sha256":digest(ip),
                        "target":str(target_path),"target_sha256":digest(target_path),"mask":str(mp),"mask_sha256":digest(mp),
                        "source_original":case["source"],"source_original_sha256":case["source_original_sha256"],
                        "crop":list(box),"render":render,"seed":seed,"training_crop_kind":"lower",
                        "model_input_resolution":a.resolution})
                pairs.extend(added)
                accepted[split]+=1;record.update(accepted=True,target=str(target_path),target_size=[width,height],
                    native_size=list(original.size),fresh_crop_checks=crop_checks,ocr=text)
            except Exception as error:  # noqa: BLE001 - uncertainty cannot become clean supervision
                record["rejection"]=str(error)[:220]
            scanned+=1;output.write(json.dumps(record,ensure_ascii=False)+"\n");output.flush()
            print(json.dumps({"screened":scanned,"accepted":accepted,"needed":need}),flush=True)
            if accepted==need:break
    if accepted!=need:
        (work/"report.json").write_text(json.dumps({"state":"incomplete","accepted":accepted,"needed":need}))
        raise ValueError("Insufficient freshly proved lower crops; no completed dataset or training")
    if (digest(a.parent_manifest)!=parent_sha or digest(a.proposals)!=proposal_sha
            or any(digest(path)!=sha for path,sha in parent_files.items())):
        raise ValueError("Parent/proposal assets changed")
    manifest={**parent,"revision":revision,"pairs":pairs,"resolution":None,
        "model_input_resolutions":[parent.get("resolution"),a.resolution],
        "parent_manifest":str(a.parent_manifest.resolve()),"parent_manifest_sha256":parent_sha,
        "geometry_proposal_sha256":proposal_sha,"crop_screening_identity":identity,
        "lower_augmentation":accepted,"previous_exposure_overlap":len(parents),
        "parent_exposure_policy":"Reuses parent source splits; no new independent source claims",
        "new_original_sources":0,"source_overlap":0}
    (work/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    validate_pairs(work/"manifest.json")
    with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as response:
        if reserved!=helper["require_disjoint_server_gpus"](json.load(response),[a.gpu],True):
            raise ValueError("Server reservation changed")
    if revision!=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip():
        raise ValueError("Pinned source changed")
    report={"state":"completed","pairs":len(pairs),"new_lower_originals":accepted,"new_original_sources":0,
        "parent_source_roles_unchanged":True,"seconds":round(time.monotonic()-began,3),
        "load_seconds":round(load_seconds,3),"image_generations":0,"quality_permission":False}
    (work/"report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=="__main__":main()
