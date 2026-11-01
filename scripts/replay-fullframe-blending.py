"""Development-only same-GEN compositing comparison; strict fresh QA remains separate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import runpy
import subprocess
import time
import uuid
from pathlib import Path

from PIL import Image

from watermark_studio.config import ProcessingConfig, Region
from watermark_studio.images import (
    composite_protected,
    load_photo,
    protect_edge_completion_context,
    save_output,
)
from watermark_studio.source_panel_rims import effective_rim_scope, restore_proven_rims


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--job",required=True)
    p.add_argument("--capture",type=Path,required=True)
    p.add_argument("--tag",required=True)
    p.add_argument("--blending",choices=["feather","harmonic","harmonic-fine"],required=True)
    p.add_argument("--sr-model",type=Path,help="Optional pinned offline learned full-frame upscaler")
    p.add_argument("--sr-sha256")
    p.add_argument("--gpu",type=int)
    a=p.parse_args()
    if any(v is not None for v in (a.sr_model,a.sr_sha256,a.gpu)) and not all(v is not None for v in (a.sr_model,a.sr_sha256,a.gpu)):
        p.error("SR requires explicit model, SHA and GPU together")
    if not re.fullmatch(r"[a-f0-9]{32}",a.job) or not re.fullmatch(r"[a-zA-Z0-9_-]+",a.tag):
        p.error("Explicit completed development job and new simple tag required")
    os.environ.pop("WATERMARK_REPAIR_CAPTURE",None)
    state=Path.home()/".local/state/watermark-studio"
    summary=json.loads((state/a.job/"summary.json").read_text())
    rows=list(map(json.loads,(state/a.job/"results.jsonl").read_text().splitlines()))
    if summary["state"]!="completed" or not 1<=len(rows)<=64 or len(rows)!=summary["total"]:
        raise ValueError("Complete bounded development source required")
    captures={}
    for meta in a.capture.glob("*/capture.json"):
        info=json.loads(meta.read_text());key=info["original_rgb_sha256"]
        if key in captures:
            raise ValueError("Repeated raw generation for one source")
        captures[key]=(meta.parent,info)
    options=ProcessingConfig.model_validate(summary["options"])
    input_root=Path(summary["input_dir"]).resolve()
    base_root=Path(summary["output_dir"]).resolve()
    work=Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031")/a.tag
    work.mkdir(exist_ok=False);out=work/"output";out.mkdir()
    (work/"sources.json").write_text(json.dumps([str(input_root/r["file"]) for r in rows],ensure_ascii=False))
    job=uuid.uuid4().hex;folder=state/job;folder.mkdir(exist_ok=False)
    revision=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    options.repair_blending=a.blending
    net=None
    if a.sr_model is not None:
        import urllib.request
        helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
        with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as response:
            helper["require_disjoint_server_gpus"](json.load(response),[a.gpu],True)
        os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
        import torch

        from watermark_studio.super_resolution_probe import load_upscaler
        torch.set_num_threads(1)
        net=load_upscaler(a.sr_model,a.sr_sha256,"cuda:0")
    began=time.monotonic();hashes={}
    for row in rows:
        name=Path(row["file"])
        source=(input_root/name).resolve();base=(base_root/name).resolve()
        if name.is_absolute() or not source.is_relative_to(input_root) or not base.is_relative_to(base_root):
            raise ValueError("Path escapes immutable original roots")
        if not row.get("ok") or not row.get("background_preserved") or not row.get("repair_regions"):
            raise ValueError("Every selected source needs an actual protected full-frame generation")
        hashes[str(source)]=digest(source);hashes[str(base)]=digest(base)
        before,exif,icc,alpha=load_photo(source)
        raw_dir,info=captures[hashlib.sha256(before.convert("RGB").tobytes()).hexdigest()]
        raw=Image.open(raw_dir/"generated.png").convert("RGB")
        if list(raw.size)!=info["native_size"]:
            raise ValueError("Raw native dimensions differ")
        if hashlib.sha256(raw.tobytes()).hexdigest()!=info["generated_rgb_sha256"]:
            raise ValueError("Raw full-frame RGB digest mismatch")
        regions=[Region.model_validate(r) for r in row["repair_regions"]]
        cores=row.get("protected_core_boxes",[]);rims=row.get("source_rim_protection",[])
        scoped,mapped,invalid=effective_rim_scope(before,regions,cores,rims,header_core=True)
        if invalid or [r.model_dump(mode="json") for r in scoped]!=row["source_rim_composition"]["regions"]:
            raise ValueError("Frozen composition source proof mismatch")
        def compose(style,frame=raw,before=before,scoped=scoped,mapped=mapped,row=row,rims=rims,alpha=alpha):
            image=composite_protected(before,frame,scoped,options.feather,style,
                core_boxes={tuple(c["mask"]):tuple(c["core"]) for c in mapped})
            image,_=protect_edge_completion_context(before,image,row.get("source_edge_completion",[]))
            image,count=restore_proven_rims(before,image,rims)
            if count!=sum(r.get("approved") is True for r in rims):
                raise ValueError("Cannot restore all frozen source rim proofs")
            if alpha is not None:
                image.putalpha(alpha)
            return image
        old_options=ProcessingConfig.model_validate(summary["options"])
        reproduction=work/("reproduction"+base.suffix)
        save_output(compose(old_options.repair_blending),source,reproduction,exif,icc,old_options)
        if digest(reproduction)!=hashes[str(base)]:
            raise ValueError("Current baseline cannot be reproduced byte-for-byte")
        reproduction.unlink()
        frame=raw
        sr_seconds=0.
        if net is not None:
            from watermark_studio.super_resolution_probe import upscale_tiled
            model_size=row["controlnet_probe"]["model_input_size"]
            if not isinstance(model_size,list) or len(model_size)!=2:
                raise ValueError("Exact frozen whole-frame inference size required for SR probe")
            started=time.monotonic()
            frame=upscale_tiled(raw.resize(tuple(model_size),Image.Resampling.LANCZOS),net).resize(before.size,Image.Resampling.LANCZOS)
            sr_seconds=round(time.monotonic()-started,3)
        image=compose(a.blending,frame=frame);destination=out/name;destination.parent.mkdir(parents=True,exist_ok=True)
        size=save_output(image,source,destination,exif,icc,options)
        if size!=row["output_size"]:
            raise ValueError("Source output resolution changed")
        result=row|{"quality":{},"output_size":size,"harmonization":image.info.get("harmonization"),
            "composition_replay":{"source_job":a.job,"raw_rgb_sha256":info["generated_rgb_sha256"],
                "baseline_output_sha256":hashes[str(base)],"baseline_bytes_reproduced":True,
                "blending":a.blending,"learned_sr":net is not None,"sr_model_sha256":a.sr_sha256,
                "sr_seconds":sr_seconds,"sr_input_resampled_from_native_capture":net is not None,
                "additional_generations":0,"quality_permission":False}}
        with (folder/"results.jsonl").open("a") as f:
            f.write(json.dumps(result,ensure_ascii=False)+"\n")
    if any(digest(path)!=value for path,value in hashes.items()):
        raise ValueError("Read-only original or baseline changed")
    report=summary|{"job":job,"source_job":a.job,"output_dir":str(out),"options":options.model_dump(),
        "code_revision":revision,"scope":"Same frozen raw GEN1 pixels, different CPU blending; fresh QA required",
        "seconds":round(time.monotonic()-began,3),"quality_evaluated":False,"additional_generations":0,
        "originals_and_baselines_unchanged":True,"all_baseline_bytes_reproduced":True}
    (folder/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (work/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:report[k] for k in ("job","source_job","seconds","total","additional_generations","all_baseline_bytes_reproduced")}))


if __name__=="__main__":
    main()
