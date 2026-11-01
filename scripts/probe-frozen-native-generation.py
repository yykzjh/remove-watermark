"""Frozen-source whole-photo native edit, one GEN per case, exact VAE-input capture."""
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
from functools import wraps
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--source-job",required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--tag",required=True)
    p.add_argument("--files",nargs="+")
    p.add_argument("--lora-path",type=Path)
    p.add_argument("--lora-sha256",default="")
    p.add_argument("--lora-alpha",type=float,default=0.)
    p.add_argument("--lora-scale",type=float,default=1.)
    p.add_argument("--resolution",type=int,choices=[0,512,768,1024,1280,1536,2048],default=0,
        help="Explicit development override; 0 reuses frozen/default budget")
    p.add_argument("--generation-encoder-offload",action="store_true",
        help="Park only the already-encoded BF16 prompt encoder during DiT/VAE")
    p.add_argument("--vae-tiling",action="store_true")
    p.add_argument("--caption-job",help="Optional SHA-bound existing source observations; no additional caption/GEN calls")
    p.add_argument("--frozen-input-budget",action="store_true",help="Reuse each source record original single inference budget, including document overrides")
    p.add_argument("--sigma-profile",choices=["native","tail9","tail10","tail12","mid10","mid12","turturbo3"],default="native",help="Offline tail refinement; adds measured forward cost but no second image generation")
    p.add_argument("--quantization",choices=["float8","none"],default="float8")
    p.add_argument("--granularity",choices=["tensor","row"],default="tensor")
    p.add_argument("--transformer-phase-offload",action="store_true")
    p.add_argument("--kv-cache",action="store_true")
    p.add_argument("--source-flow-context",action="store_true")
    p.add_argument("--compile-projections",action="store_true")
    p.add_argument("--separate-vae-decode",action="store_true")
    a=p.parse_args()
    if a.gpu<0 or not re.fullmatch(r"[a-f0-9]{32}",a.source_job) or not re.fullmatch(r"[a-zA-Z0-9_-]+",a.tag):
        p.error("Explicit completed source job, GPU and simple new tag required")
    if a.lora_path and (not re.fullmatch(r"[a-f0-9]{64}",a.lora_sha256) or a.lora_alpha<=0):
        p.error("Explicit artifact SHA and alpha required for offline adapter fusion")
    if a.sigma_profile=="turturbo3" and (not a.lora_path or
            a.lora_sha256!="fa465274a7bb69687bfd6fd36d3e6cda967efba7a649aab25a37d62adf6286be" or
            a.lora_alpha!=128):
        p.error("Published three-step schedule requires exact pinned Turbo-T2I-3steps weights/alpha128")
    if a.resolution and a.frozen_input_budget:
        p.error("Explicit resolution override cannot be combined with frozen budgets")
    if a.generation_encoder_offload and a.transformer_phase_offload:
        p.error("Select one encoder residency mechanism")
    h=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as response:
        isolation=h["require_disjoint_server_gpus"](json.load(response),[a.gpu],True)
    state=Path.home()/".local/state/watermark-studio"
    summary=json.loads((state/a.source_job/"summary.json").read_text())
    rows=list(map(json.loads,(state/a.source_job/"results.jsonl").read_text().splitlines()))
    if a.files:
        if len(set(a.files))!=len(a.files) or not set(a.files).issubset({r["file"] for r in rows}):
            p.error("Existing unique source subset required")
        rows=[r for r in rows if r["file"] in set(a.files)]
    if summary["state"]!="completed" or not 1<=len(rows)<=64 or any(not r.get("ok") or not r.get("background_preserved") for r in rows):
        p.error("1..64 valid protected frozen sources required")
    if a.frozen_input_budget and any(not isinstance(r.get("model_input_budgets"),list) or len(r["model_input_budgets"])!=1 or r["model_input_budgets"][0] not in (512,768,1024,1280,1536,2048) for r in rows):
        p.error("Every frozen source needs one explicit supported model input budget")
    source_root=Path(summary["input_dir"]).resolve();hashes={}
    for row in rows:
        source=(source_root/row["file"]).resolve()
        if Path(row["file"]).is_absolute() or not source.is_relative_to(source_root):
            p.error("Source must remain under immutable input root")
        hashes[row["file"]]=digest(source)
    captions={}
    if a.caption_job:
        from watermark_studio.probe_conditioning import frozen_source_captions
        captions=frozen_source_captions(state,a.caption_job,hashes)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu);os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
    import torch
    from PIL import Image
    from safetensors.torch import save_file

    from watermark_studio.batching import dimensions, independent_batch, override_methods
    from watermark_studio.config import DeploymentConfig, ProcessingConfig, Region
    from watermark_studio.images import (
        composite_protected,
        load_photo,
        protect_edge_completion_context,
        save_output,
    )
    from watermark_studio.prompts import build_full_prompt
    from watermark_studio.source_panel_rims import effective_rim_scope, restore_proven_rims
    from watermark_studio.worker import load_pipeline
    torch.set_num_threads(1)
    cfg=DeploymentConfig(model=str(a.model),gpus=[a.gpu],quantization=a.quantization,float8_granularity=a.granularity,
        transformer_phase_offload=a.transformer_phase_offload,prompt_encoder="lean",
        compile=a.compile_projections,compile_scope="projections",compile_dynamic=a.compile_projections,
        separate_vae_decode=a.separate_vae_decode,vae_tiling=a.vae_tiling,
        vae_tile_size=1024,vae_tile_stride=768,lora_path=str(a.lora_path) if a.lora_path else "",
        lora_sha256=a.lora_sha256,lora_alpha=a.lora_alpha,lora_scale=a.lora_scale)
    began=time.monotonic();pipe=load_pipeline(cfg,str(a.model))
    if a.generation_encoder_offload:
        from watermark_studio.source_residency import prepare_transformer_host_mirror
        pipe.text_encoder.to("cpu")
        prepare_transformer_host_mirror(pipe.text_encoder)
        pipe.text_encoder.to("cuda:0",non_blocking=True)
        torch.cuda.synchronize()
        pipe._watermark_phase_offload=True
    load_seconds=time.monotonic()-began
    from watermark_studio.probe_schedules import tail_sigma_grid
    original_sigmas=list(pipe.config.sample_sigmas)
    experimental_sigmas=tail_sigma_grid(original_sigmas,a.sigma_profile)
    if a.sigma_profile!="native":
        pipe.register_to_config(sample_sigmas=experimental_sigmas)
    options=ProcessingConfig.model_validate(summary["options"])
    options.quality_auto_retry=False;options.quality_max_attempts=0;options.overwrite=False
    work=Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031")/a.tag
    work.mkdir(exist_ok=False);outputs=work/"output";outputs.mkdir()
    (work/"sources.json").write_text(json.dumps([str(source_root/r["file"]) for r in rows],ensure_ascii=False))
    job=uuid.uuid4().hex;folder=state/job;folder.mkdir(exist_ok=False);(folder/"latents").mkdir()
    revision=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    print(json.dumps({"stage":"ready","job":job,"total":len(rows),"load_seconds":round(load_seconds,3)}),flush=True)
    began=time.monotonic()
    for index,row in enumerate(rows):
        name=row["file"];source,exif,icc,alpha=load_photo(source_root/name)
        regions=[Region.model_validate(r) for r in row["repair_regions"]]
        if not regions:
            raise ValueError("Select generated development sources, not source-copy cases")
        rims=row.get("source_rim_protection",[])
        scoped,cores,invalid=effective_rim_scope(source,regions,row.get("protected_core_boxes",[]),rims,header_core=True)
        if invalid:raise ValueError("Frozen source rim proof invalid")
        budget=a.resolution or (row["model_input_budgets"][0] if a.frozen_input_budget else options.resolution)
        width,height=dimensions(source,budget)
        seed=(options.seed+int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],"big"))%(2**32)
        prompt=build_full_prompt(options.extra_prompt,options.whole_prompt_profile)
        if captions:
            from watermark_studio.probe_conditioning import observed_scene_prompt
            prompt=observed_scene_prompt(prompt,captions[name])
        item={"image":source,"prompt":prompt,"seed":seed}
        if a.source_flow_context:
            item["_denoising_context_regions"]=[r.model_dump() for r in scoped]
        capture={};original_decode=pipe.vae.decode
        @wraps(original_decode)
        def decode(z,*args,capture=capture,name=name,width=width,height=height,seed=seed,original_decode=original_decode,**kwargs):
            if capture or z.shape[0]!=1 or not torch.isfinite(z).all().item():
                raise ValueError("Exactly one finite whole-photo VAE input required")
            path=folder/"latents"/(hashlib.sha256(name.encode()).hexdigest()[:24]+".safetensors")
            save_file({"vae_input":z.detach().cpu().contiguous()},path)
            capture.update(path=str(path),file_sha256=digest(path),shape=list(z.shape),
                model_input_size=[width,height],source_file_sha256=hashes[name],seed=seed,
                layout="actual_vae_input_BCFHW",additional_generations=0)
            return original_decode(z,*args,**kwargs)
        started=time.monotonic()
        with override_methods(pipe.vae,decode=decode):
            generated=independent_batch(pipe,[item],budget,a.kv_cache)[0].convert("RGB").resize(source.size,Image.Resampling.LANCZOS)
        capture["native_generated_rgb_sha256"]=hashlib.sha256(generated.tobytes()).hexdigest()
        generated=composite_protected(source,generated,scoped,options.feather,options.repair_blending,
            core_boxes={tuple(c["mask"]):tuple(c["core"]) for c in cores})
        generated,_=protect_edge_completion_context(source,generated,row.get("source_edge_completion",[]))
        generated,count=restore_proven_rims(source,generated,rims)
        if count!=sum(r.get("approved") is True for r in rims):raise ValueError("Incomplete frozen rim proof")
        if alpha is not None:generated.putalpha(alpha)
        destination=outputs/name;destination.parent.mkdir(parents=True,exist_ok=True)
        size=save_output(generated,source_root/name,destination,exif,icc,options)
        source_fields=("file","background_preserved","repair_regions","protected_core_boxes","source_scene",
            "source_rim_protection","source_edge_completion","source_geometry_requery","source_physical_vetting",
            "source_detail_views","source_detail_context_requeries")
        result={k:row[k] for k in source_fields if k in row}
        if captions:
            result["observed_scene_conditioning"]=captions[name]
        if a.lora_path:
            result["image_adapter"]=pipe._image_adapter
        result.update(ok=True,quality={},quality_attempts=1,model_inferences=1,processing_path="whole_composite",
            action="restored",output_size=size,model_input_budgets=[budget],latent_capture=capture,harmonization=generated.info.get("harmonization"),
            denoising_context=[item["_denoising_context_log"]] if "_denoising_context_log" in item else [],
            source_rim_composition={"regions":[r.model_dump() for r in scoped],"cores":cores},
            native_generation_probe={"model_input_size":[width,height],"seed":seed,"whole_frame":True,
                "seconds":round(time.monotonic()-started,4),"kv_cache":a.kv_cache,
                "encoder_residency":item.get("_encoder_generation_log"),"sigma_profile":a.sigma_profile,"steps":len(pipe.scheduler.timesteps)})
        with (folder/"results.jsonl").open("a") as f:f.write(json.dumps(result,ensure_ascii=False)+"\n")
        print(json.dumps({"completed":index+1,"total":len(rows),"job":job}),flush=True)
    if any(digest(source_root/name)!=value for name,value in hashes.items()):
        raise ValueError("Original source changed")
    if revision!=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip():
        raise ValueError("Code changed during probe")
    report={"state":"completed","job":job,"source_job":a.source_job,"input_dir":str(source_root),"output_dir":str(outputs),
        "total":len(rows),"options":options.model_dump(),"code_revision":revision,"sources_sha256":hashes,
        "seconds":round(time.monotonic()-began,3),"load_seconds":round(load_seconds,3),"quality_evaluated":False,
        "native_settings":{"quantization":a.quantization,"granularity":a.granularity,"kv_cache":a.kv_cache,
            "compile":a.compile_projections,"transformer_phase_offload":a.transformer_phase_offload,
            "sigma_profile":a.sigma_profile,"frozen_input_budget":a.frozen_input_budget,"resolution_override":a.resolution,
            "generation_encoder_offload":a.generation_encoder_offload,"vae_tiling":a.vae_tiling,
            "image_adapter":getattr(pipe,"_image_adapter",None),"caption_job":a.caption_job,"original_sigmas":original_sigmas,"actual_sigmas":experimental_sigmas,
            "source_flow_context":a.source_flow_context,"separate_vae_decode":a.separate_vae_decode},
        "server_isolation":isolation,"scope":"Frozen-source native whole-photo GEN1 with exact VAE-input capture"}
    (folder/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (work/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:report[k] for k in ("job","total","seconds","load_seconds")}),flush=True)


if __name__=="__main__":
    main()
