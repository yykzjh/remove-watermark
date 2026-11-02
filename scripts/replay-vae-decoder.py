"""Same-GEN latent decoder A/B; baseline RGB and final bytes must reproduce first."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import runpy
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--job",required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--weights",type=Path)
    p.add_argument("--sha256",default="")
    p.add_argument("--original-decoder",action="store_true",help="Output-only SR comparison using the same original BF16 VAE")
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--tag",required=True)
    p.add_argument("--super-resolution-weights",type=Path,
        help="Optional pinned RGB x4 operator applied directly to model-scale decoder pixels")
    p.add_argument("--super-resolution-sha256",default="")
    p.add_argument("--super-resolution-profile",choices=["rrdb","swinir-classical","swinir-lightweight"],default="rrdb")
    p.add_argument("--super-resolution-precision",choices=["float32","float16"])
    p.add_argument("--super-resolution-tile",type=int,choices=[0,256,512])
    p.add_argument("--super-resolution-overlap",type=int,choices=[0,32,64,128])
    p.add_argument("--precision",choices=["bfloat16","float16"],default="bfloat16")
    a=p.parse_args()
    if a.gpu<0 or not re.fullmatch(r"[a-f0-9]{32}",a.job) or not re.fullmatch(r"[a-zA-Z0-9_-]+",a.tag):
        p.error("Exact source job, GPU and new simple tag required")
    if bool(a.super_resolution_weights)!=bool(a.super_resolution_sha256):
        p.error("Pinned SR weights and SHA must be specified together")
    if a.original_decoder:
        if a.weights or a.sha256 or not a.super_resolution_weights or a.precision != "bfloat16":
            p.error("Original decoder requires SR only and original BF16 precision")
    elif not a.weights or not a.sha256:
        p.error("Alternative decoder requires weights and SHA")
    if not a.super_resolution_weights and a.super_resolution_profile != "rrdb":
        p.error("An SR profile requires its pinned weights")
    sr_options=(a.super_resolution_precision,a.super_resolution_tile,a.super_resolution_overlap)
    if any(v is not None for v in sr_options) and (not a.super_resolution_weights or a.super_resolution_profile=="rrdb"):
        p.error("Explicit SR precision/tile/overlap are for SwinIR comparisons only")
    helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as response:
        server=helper["require_disjoint_server_gpus"](json.load(response),[a.gpu],True)
    state=Path.home()/".local/state/watermark-studio";root=state/a.job
    summary=json.loads((root/"summary.json").read_text())
    rows=list(map(json.loads,(root/"results.jsonl").read_text().splitlines()))
    if summary["state"]!="completed" or not 1<=len(rows)<=64 or len(rows)!=summary["total"]:
        p.error("Completed bounded development job required")
    if any(not r.get("latent_capture") or r.get("model_inferences")!=1 or not r.get("background_preserved") for r in rows):
        p.error("Every selected image needs one valid protected generation and its original latent capture")
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu);os.environ.pop("WATERMARK_REPAIR_CAPTURE",None)
    import torch
    from safetensors.torch import load_file

    from watermark_studio.batching import decode_latents
    from watermark_studio.config import ProcessingConfig, Region
    from watermark_studio.decoder_probe import decoder_pair
    from watermark_studio.full_frame_sampling import (
        full_frame_mapping,
        restore_full_frame,
        sample_full_frame,
    )
    from watermark_studio.images import (
        composite_protected,
        load_photo,
        protect_edge_completion_context,
        save_output,
    )
    from watermark_studio.source_panel_rims import effective_rim_scope, restore_proven_rims
    torch.set_num_threads(1)
    revision=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    began=time.monotonic()
    if a.original_decoder:
        from watermark_studio.decoder_probe import original_decoder
        old=new=original_decoder(a.model,"cuda:0")
    else:
        old,new=decoder_pair(a.model,a.weights,a.sha256,"cuda:0",a.precision)
    sr=None
    if a.super_resolution_weights:
        if a.super_resolution_profile == "rrdb":
            from watermark_studio.super_resolution_probe import load_upscaler, upscale_tiled
            sr=load_upscaler(a.super_resolution_weights,a.super_resolution_sha256,"cuda:0")
            upscale=upscale_tiled
        else:
            from watermark_studio.swinir_probe import load_swinir, upscale_swinir
            sr=load_swinir(a.super_resolution_weights,a.super_resolution_sha256,"cuda:0",
                a.super_resolution_profile.removeprefix("swinir-"),a.super_resolution_precision or "float32")
            from functools import partial
            upscale=partial(upscale_swinir,
                tile=a.super_resolution_tile if a.super_resolution_tile is not None else 256,
                overlap=a.super_resolution_overlap if a.super_resolution_overlap is not None else 32)
    load_seconds=time.monotonic()-began
    work=Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031")/a.tag
    work.mkdir(exist_ok=False);out=work/"output";out.mkdir()
    source_root=Path(summary["input_dir"]).resolve();base_root=Path(summary["output_dir"]).resolve()
    (work/"sources.json").write_text(json.dumps([str(source_root/r["file"]) for r in rows],ensure_ascii=False))
    job=uuid.uuid4().hex;folder=state/job;folder.mkdir(exist_ok=False)
    options=ProcessingConfig.model_validate(summary["options"]);hashes={};began=time.monotonic();sr_seconds_sum=0.
    for row in rows:
        name=Path(row["file"]);source=(source_root/name).resolve();base=(base_root/name).resolve()
        if name.is_absolute() or not source.is_relative_to(source_root) or not base.is_relative_to(base_root):
            raise ValueError("Source path escapes immutable roots")
        capture=row["latent_capture"];path=Path(capture["path"]).resolve(strict=True)
        if not path.is_relative_to((root/"latents").resolve()) or path.stat().st_size>16*2**20:
            raise ValueError("Bounded source-job latent path required")
        if digest(path)!=capture["file_sha256"] or digest(source)!=capture["source_file_sha256"]:
            raise ValueError("Latent or original digest mismatch")
        hashes[str(source)]=digest(source);hashes[str(base)]=digest(base);hashes[str(path)]=digest(path)
        latent=load_file(path,device="cpu")
        key="vae_input" if capture.get("layout")=="actual_vae_input_BCFHW" else "latents"
        if set(latent)!={key} or list(latent[key].shape)!=capture["shape"]:
            raise ValueError("Exact original target latent layout required")
        latent=latent[key].to("cuda:0")
        if not torch.isfinite(latent).all().item():
            raise ValueError("Nonfinite source latent")
        width,height=capture["model_input_size"]
        probe=row.get("controlnet_probe",row.get("native_generation_probe",{}))
        if [width,height]!=probe.get("model_input_size") or width*height>3*2**20:
            raise ValueError("Frozen bounded full-frame inference dimensions required")
        before,exif,icc,alpha=load_photo(source)
        regions=[Region.model_validate(r) for r in row["repair_regions"]]
        rims=row.get("source_rim_protection",[])
        scoped,cores,invalid=effective_rim_scope(before,regions,row.get("protected_core_boxes",[]),rims,header_core=True)
        if invalid or [r.model_dump(mode="json") for r in scoped]!=row["source_rim_composition"]["regions"]:
            raise ValueError("Frozen source composition proof mismatch")
        hint=capture.get("translucent_conditioning")
        if hint is not None:
            from watermark_studio.translucent_hint import translucent_body_hint
            if capture.get("full_frame_sampling") or row.get("source_scene")=="document":
                raise ValueError("Dense conditioning requires native photo coordinates")
            reference,audits=translucent_body_hint(before,rims,hint["transmission"],
                foreground=hint["foreground"],_dense_transmission_experiment=hint["transmission"]<.35)
            if audits!=hint["audits"] or hashlib.sha256(reference.tobytes()).hexdigest()!=hint["source_rgb_sha256"]:
                raise ValueError("Original VAE conditioning does not reproduce")
        sampling=capture.get("full_frame_sampling")
        if sampling is not None:
            expected=full_frame_mapping(scoped,before.size,sampling["mass"],sampling["padding"])
            if row.get("source_scene")=="document" or sampling!=expected:
                raise ValueError("Sampling does not match the frozen source inspection scope")
            forward=sample_full_frame(before,sampling)
            if hashlib.sha256(forward.tobytes()).hexdigest()!=capture.get("forward_source_rgb_sha256"):
                raise ValueError("Original full-frame sampling input does not reproduce")
        for label,pipe in (("baseline",old),("modified",new)):
            started=time.monotonic()
            if key=="vae_input":
                with torch.inference_mode():
                    decoded=pipe.vae.decode(latent.to(pipe.vae.dtype),return_dict=False)[0][:,:,0]
                    frame=pipe.image_processor.postprocess(decoded,output_type="pil")[0]
            else:
                frame=decode_latents(pipe,latent,height,width)[0]
            frame=frame.convert("RGB")
            sr_input_size=list(frame.size)
            sr_seconds=0.
            if label=="modified" and sr is not None:
                torch.cuda.synchronize()
                sr_started=time.monotonic()
                frame=upscale(frame,sr)
                torch.cuda.synchronize()
                sr_seconds=round(time.monotonic()-sr_started,4)
                sr_seconds_sum+=sr_seconds
            frame=restore_full_frame(frame,before.size,sampling)
            if label=="baseline" and hashlib.sha256(frame.tobytes()).hexdigest()!=capture["native_generated_rgb_sha256"]:
                raise ValueError("Original decoder does not reproduce captured GEN RGB exactly")
            image=composite_protected(before,frame,scoped,options.feather,options.repair_blending,
                core_boxes={tuple(c["mask"]):tuple(c["core"]) for c in cores})
            image,_=protect_edge_completion_context(before,image,row.get("source_edge_completion",[]))
            image,count=restore_proven_rims(before,image,rims)
            if count!=sum(r.get("approved") is True for r in rims):
                raise ValueError("Incomplete frozen source rim proof")
            if alpha is not None:
                image.putalpha(alpha)
            destination=work/("baseline"+base.suffix) if label=="baseline" else out/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            size=save_output(image,source,destination,exif,icc,options)
            if size!=row["output_size"]:
                raise ValueError("Native output size changed")
            if label=="baseline":
                if digest(destination)!=hashes[str(base)]:
                    raise ValueError("Baseline final bytes do not reproduce")
                destination.unlink()
            else:
                result=row|{"quality":{},"harmonization":image.info.get("harmonization"),
                    "decoder_replay":{"source_job":a.job,"artifact_sha256":a.sha256,"precision":a.precision,
                        "same_latent_sha256":capture["file_sha256"],"baseline_rgb_and_bytes_reproduced":True,
                        "seconds":round(time.monotonic()-started,4),"additional_generations":0,"quality_permission":False,
                        "model_scale_super_resolution":({
                            "artifact_sha256":a.super_resolution_sha256,
                            "input_size":sr_input_size,"decoded_native_pixels":True,"factor":4,
                            "profile":a.super_resolution_profile,"original_decoder":a.original_decoder,
                            "precision":a.super_resolution_precision or ("float16" if a.super_resolution_profile=="rrdb" else "float32"),
                            "tile":a.super_resolution_tile if a.super_resolution_tile is not None else 256,
                            "overlap":(a.super_resolution_overlap if a.super_resolution_overlap is not None else 32) if a.super_resolution_profile!="rrdb" else None,
                            "output_only":True,"sr_seconds":sr_seconds
                        } if sr is not None else None)}}
                with (folder/"results.jsonl").open("a") as f:
                    f.write(json.dumps(result,ensure_ascii=False)+"\n")
    if any(digest(path)!=value for path,value in hashes.items()):
        raise ValueError("Original, latent or baseline candidate changed")
    if revision!=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip():
        raise ValueError("Code changed during same-latent experiment")
    with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as response:
        assert server==helper["require_disjoint_server_gpus"](json.load(response),[a.gpu],True)
    report=summary|{"job":job,"source_job":a.job,"output_dir":str(out),"code_revision":revision,
        "scope":"Same original target latent, decoder/SR A/B, fresh strict QA required",
        "super_resolution_sha256":a.super_resolution_sha256 or None,
        "super_resolution_profile":a.super_resolution_profile if sr is not None else None,
        "original_decoder":a.original_decoder,"sr_seconds_sum":round(sr_seconds_sum,4),
        "seconds":round(time.monotonic()-began,3),"decoder_load_seconds":round(load_seconds,3),
        "decoder_sha256":a.sha256,"decoder_precision":a.precision,"additional_generations":0,
        "quality_evaluated":False,"all_baseline_rgb_and_bytes_reproduced":True}
    (folder/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (work/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:report[k] for k in ("job","total","seconds","decoder_load_seconds","additional_generations","all_baseline_rgb_and_bytes_reproduced")}))


if __name__=="__main__":
    main()
