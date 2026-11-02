"""Small source-isolated exact-pixel LoRA pilot dataset; all image work stays remote."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import re
import subprocess
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--target",type=Path,required=True)
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--train-originals",type=int,default=48)
    p.add_argument("--validation-originals",type=int,default=12)
    p.add_argument("--candidate-limit",type=int,default=320)
    p.add_argument("--candidate-files-from",type=Path)
    p.add_argument("--require-native-pixels",action="store_true")
    p.add_argument("--render-profile",choices=["legacy","camera-v2"],default="legacy")
    p.add_argument("--seed",type=int,default=2026110204)
    p.add_argument("--resolution",type=int,choices=[512,768],default=512)
    p.add_argument("--font",type=Path,default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    a=p.parse_args()
    if (a.gpu<0 or not 8<=a.train_originals<=128 or not 4<=a.validation_originals<=32
            or not a.train_originals+a.validation_originals<=a.candidate_limit<=1024 or not a.font.is_file()):
        p.error("Small explicit source-isolated pilot and installed Linux CJK font required")
    root=a.root.resolve();data=root/"八标照片集"
    work=a.target.resolve()
    if work.parent!=root or not re.fullmatch(r"[A-Za-z0-9_-]+",work.name) or work==data:
        p.error("New top-level experiment folder required so future evaluation discovers exposure")
    work.mkdir(exist_ok=False);(work/"clean").mkdir();(work/"pairs").mkdir();(work/"masks").mkdir()
    from watermark_studio.evaluation import exclusion_sources
    exposed,folders=exclusion_sources(root,root/"quality-benchmark-cross-review-v7-64",[])
    exposed={str(Path(v).resolve()) for v in exposed}
    forbidden=set()
    for f in root.glob("*/source-selection.json"):
        metadata=json.loads(f.read_text())
        for value in metadata.get("source_sha256",{}).values():
            if isinstance(value,str) and len(value)==64:forbidden.add(value)
    def prior_hash(path):
        path=Path(path)
        return digest(path) if path.is_file() else None
    with ThreadPoolExecutor(max_workers=4) as pool:
        forbidden.update(value for value in pool.map(prior_hash,sorted(exposed)) if value)
    (work/"forbidden-sha256.json").write_text(json.dumps(sorted(forbidden)))
    cloud=json.loads((data/".download-state/manifest.json").read_text())
    prefix="/codex-download-1mQf6F6g-20261031/八标照片集/"
    candidates=[]
    for row in cloud:
        name=row["path"]
        if not name.startswith(prefix):continue
        path=(data/name.removeprefix(prefix)).resolve()
        if (path.is_relative_to(data.resolve()) and path.is_file() and str(path) not in exposed
                and path.suffix.lower() in {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"}):
            candidates.append(path)
    selector_sha=None;expected_hashes={}
    if a.candidate_files_from:
        raw=a.candidate_files_from.read_bytes();selector=json.loads(raw)
        entries=selector.get("sources",[])
        if selector.get("version")!=1 or not 1<=len(entries)<=2048:
            raise ValueError("Bounded versioned candidate source reservation required")
        allowed={str(path):path for path in candidates};chosen=[]
        for entry in entries:
            path=Path(entry["path"]).resolve(strict=True);sha=entry["sha256"]
            if (not path.is_relative_to(data.resolve()) or not re.fullmatch(r"[a-f0-9]{64}",sha)
                    or str(path) in expected_hashes):
                raise ValueError("Invalid or repeated candidate source identity")
            expected_hashes[str(path)]=sha
            if str(path) in allowed:chosen.append(allowed[str(path)])
        candidates=chosen;selector_sha=hashlib.sha256(raw).hexdigest()
    rng=random.Random(a.seed);groups=defaultdict(list)
    for path in candidates:groups[path.relative_to(data).parts[0]].append(path)
    for group in groups.values():rng.shuffle(group)
    ordered=[p for group in itertools.zip_longest(*[groups[k] for k in sorted(groups)]) for p in group if p is not None]
    selected=[];scanned={};rejected=Counter();accepted=[];parent_hashes=set()
    for path in ordered:
        if len(selected)>=a.candidate_limit:break
        sha=digest(path)
        if expected_hashes and expected_hashes[str(path)]!=sha:
            raise ValueError("Reserved original source bytes changed")
        if sha in forbidden or sha in parent_hashes:
            rejected["previously_exposed_or_duplicate"]+=1;continue
        selected.append((path,sha));parent_hashes.add(sha)
    if len(selected)<a.train_originals+a.validation_originals:
        raise ValueError("Insufficient independent original sources for the pilot")
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
    import torch
    from PIL import Image
    from transformers import Qwen3VLProcessor

    from watermark_studio.batching import dimensions
    from watermark_studio.config import DeploymentConfig
    from watermark_studio.detection import locate_watermarks_batch
    from watermark_studio.images import load_photo
    from watermark_studio.model_identity import compare_model_weights
    from watermark_studio.quality_loading import load_quality_encoder, load_quality_processor
    from watermark_studio.text_presence import TargetTextPresence
    from watermark_studio.training_pairs import clean_crop, render_profile_pair
    with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as response:
        health=json.load(response)
    import runpy
    helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    isolation=helper["require_disjoint_server_gpus"](health,[a.gpu],True)
    model_root=Path.home()/".cache/watermark-models/models"
    eroot=model_root/"Qwen--Qwen-Image-2.1-Turbo/snapshots/master"
    vroot=model_root/"Qwen--Qwen3-VL-4B-Instruct/snapshots/master"
    identity=compare_model_weights(eroot/"text_encoder",vroot)
    if identity["relation"]!="different_weights":raise ValueError("Independent screening checkpoints required")
    torch.set_num_threads(1)
    cfg=DeploymentConfig(quality_verifier=True,quality_backend="independent",quality_quantization="none")
    editor=load_quality_encoder(eroot/"text_encoder",cfg,"cuda:0")
    ep=Qwen3VLProcessor.from_pretrained(eroot/"processor",local_files_only=True)
    verifier=load_quality_encoder(vroot,cfg,"cuda:0");vp=load_quality_processor(vroot);ocr=TargetTextPresence()
    revision=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    began=time.monotonic();need=a.train_originals+a.validation_originals
    def persist_exposure():
        for name,value in (("sources.json",list(scanned)),("source-selection.json",
                {"source_sha256":scanned,"sources":list(scanned),"sampling_seed":a.seed,
                 "purpose":"LoRA pilot screening; excluded from future evaluation"})):
            temp=work/(name+".tmp");temp.write_text(json.dumps(value,ensure_ascii=False))
            os.replace(temp,work/name)
    with (work/"screening.jsonl").open("x") as stream:
        for path,sha in selected:
            scanned[str(path)]=sha;persist_exposure();record={"source":str(path),"source_sha256":sha,"accepted":False}
            try:
                if digest(path)!=sha:raise ValueError("Original changed before screening")
                original=load_photo(path)[0]
                first=locate_watermarks_batch([original],editor,ep,"cuda:0",1280,1)[0]
                second=locate_watermarks_batch([original],verifier,vp,"cuda:0",1280,1)[0]
                if isinstance(first,Exception) or isinstance(second,Exception):
                    raise RuntimeError("Original localization uncertain")  # noqa: TRY004 - model failure, not caller type validation
                regions=first[0]+second[0]
                if len(regions)>12:raise ValueError("Too many independent source hypotheses")
                aspect=[4/3,3/4,1.,16/9][len(accepted)%4]
                box=clean_crop(original.size,regions,aspect=aspect)
                if box is None:raise ValueError("No sufficiently large clear source rectangle")
                clean=original.crop(box)
                size=dimensions(clean,a.resolution)
                if a.require_native_pixels and (size[0]>clean.width or size[1]>clean.height):
                    raise ValueError("Native clean crop has insufficient pixels for requested training resolution")
                crop_checks=[]
                for model,processor in [(editor,ep),(verifier,vp)]:
                    result=locate_watermarks_batch([clean],model,processor,"cuda:0",1280,1)[0]
                    if isinstance(result,Exception) or result[0]:
                        raise ValueError("Clean crop has a digital hypothesis or uncertainty")
                    crop_checks.append(result[1])
                text=ocr.inspect(clean,[[0.,0.,1.,1.]])
                if text.get("status")!="clear":raise ValueError("Clean crop OCR not clear")
                size=dimensions(clean,a.resolution)
                target=clean.resize(size,Image.Resampling.LANCZOS).convert("RGB")
                target_path=work/"clean"/(sha[:24]+".png");target.save(target_path)
                record.update(accepted=True,crop=list(box),native_size=list(original.size),
                    target=str(target_path),target_sha256=digest(target_path),target_size=list(size),
                    original_hypotheses=[r.model_dump() for r in regions],
                    source_checks=[first[1],second[1]],crop_checks=crop_checks,ocr=text)
                accepted.append(record)
            except Exception as error:  # noqa: BLE001 - screening uncertainty never becomes clean supervision
                record["rejection"]=str(error)[:200];rejected[record["rejection"]]+=1
            stream.write(json.dumps(record,ensure_ascii=False)+"\n");stream.flush()
            print(json.dumps({"screened":len(scanned),"accepted":len(accepted),"needed":need}),flush=True)
            if len(accepted)==need:break
    (work/"sources.json").write_text(json.dumps(list(scanned)))
    (work/"source-selection.json").write_text(json.dumps({"source_sha256":scanned,"sources":list(scanned),
        "sampling_seed":a.seed,"purpose":"LoRA pilot screening; excluded from future evaluation"}))
    if len(accepted)!=need:
        (work/"report.json").write_text(json.dumps({"state":"incomplete","accepted":len(accepted),"needed":need,"rejected":dict(rejected)}))
        raise ValueError("Clean-screening coverage insufficient; do not train partial unapproved data")
    order=list(range(need));rng.shuffle(order);validation=set(order[:a.validation_originals]);pairs=[]
    for index,row in enumerate(accepted):
        split="validation" if index in validation else "train"
        target=load_photo(row["target"])[0]
        for variant in range(4):
            seed=(a.seed+int(row["source_sha256"][:16],16)+variant)%(2**32)
            edited,mask,render=render_profile_pair(target,variant,random.Random(seed),a.font,profile=a.render_profile)
            name=row["source_sha256"][:24]+"-"+str(variant)+".png"
            input_path=work/"pairs"/name;mask_path=work/"masks"/name
            edited.save(input_path);mask.save(mask_path)
            pairs.append({"id":name.removesuffix(".png"),"split":split,"input":str(input_path),
                "input_sha256":digest(input_path),"target":row["target"],"target_sha256":row["target_sha256"],
                "mask":str(mask_path),"mask_sha256":digest(mask_path),"source_original":row["source"],
                "source_original_sha256":row["source_sha256"],"crop":row["crop"],"render":render,"seed":seed})
    train_sha={r["source_original_sha256"] for r in pairs if r["split"]=="train"}
    val_sha={r["source_original_sha256"] for r in pairs if r["split"]=="validation"}
    if train_sha&val_sha or (train_sha|val_sha)&forbidden:raise ValueError("Source-level training/evaluation leakage")
    if any(digest(path)!=sha for path,sha in scanned.items()):raise ValueError("Screened original changed")
    if revision!=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip():raise ValueError("Code changed during preparation")
    with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as response:
        if isolation!=helper["require_disjoint_server_gpus"](json.load(response),[a.gpu],True):
            raise ValueError("Server GPU reservation changed")
    manifest={"version":1,"state":"completed","revision":revision,"resolution":a.resolution,
        "source_identity":identity,"train_originals":len(train_sha),"validation_originals":len(val_sha),
        "source_overlap":0,"previous_exposure_overlap":0,"candidate_selector_sha256":selector_sha,
        "require_native_pixels":a.require_native_pixels,"render_profile":a.render_profile,"font_sha256":digest(a.font),"pairs":pairs,
        "source_screening":"union original hypotheses + .04 frame padding; fresh dual crop localization empty + OCR clear"}
    path=work/"manifest.json";path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    report={"state":"completed","train_pairs":sum(r["split"]=="train" for r in pairs),
        "validation_pairs":sum(r["split"]=="validation" for r in pairs),"originals":need,
        "screened":len(scanned),"seconds":round(time.monotonic()-began,3),
        "manifest_sha256":digest(path),"exclusion_folders":folders,"forbidden_sha256_count":len(forbidden),
        "rejected":dict(rejected),"source_overlap":0,"previous_exposure_overlap":0}
    (work/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!="exclusion_folders"}),flush=True)


if __name__=="__main__":
    main()
