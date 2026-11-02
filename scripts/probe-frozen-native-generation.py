"""Frozen-source whole-photo native edit, one GEN per case, exact VAE-input capture."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    p.add_argument("--files-from",type=Path,help="Bounded JSON filename list, avoids large command lines")
    p.add_argument("--base-steps",type=int,choices=[0,12,16,20,24,40],default=0,
        help="Offline original Qwen2.1 family checkpoint only; never overrides a Turbo grid")
    p.add_argument("--uniform-steps",type=int,choices=[0,12,16,20,24,40],default=0,
        help="Explicit offline Turbo uniform schedule comparison; leaves checkpoint/default deployment unchanged")
    p.add_argument("--deployment-uniform-steps",type=int,choices=[0,12,16,20,24,40],default=0,
        help="Verify the actual deployment sampler without per-item offline overrides")
    p.add_argument("--foveation-mass",type=float,default=0.,
        help="Offline reversible complete-frame sampling mass in0..0.35; 0 keeps native input")
    p.add_argument("--hint-transmission",type=float,default=0.,
        help="Offline conditioning-only transmission assumption; 0 disables")
    p.add_argument("--hint-foreground",type=float,default=255.)
    p.add_argument("--fresh-rim-reports",nargs="+",type=Path)
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
    p.add_argument("--context-margin-cells",type=int,choices=[0,1,2,3,4,6,8,12,16],default=1,
        help="Offline latent denoising buffer only; does not enlarge final output edit scope")
    p.add_argument("--compile-projections",action="store_true")
    p.add_argument("--exclude-reference-keys",action="store_true",
        help="Offline target-only attention probe; complete original source pixels still enter both encoders")
    p.add_argument("--exclude-prefix-reference-keys",action="store_true",
        help="Offline all-query reference ablation, including constant source/text prefix; implies neither an extra generation nor changed source pixels")
    p.add_argument("--reference-exclusion-margin-cells",type=int,choices=[0,1,2,3,4,6,8],default=0,
        help="Offline VAE reference-token exclusion buffer only; never expands output pixel edits")
    p.add_argument("--compact-reference-cache",action="store_true",
        help="Offline lossless compaction of already-excluded prefix keys; requires KV cache and reference exclusion")
    p.add_argument("--reference-exclusion-max-sigma",type=float,default=1.,
        help="Offline target-only late-step exclusion: keep full constant prefix, suppress selected reference keys only at sigma<=cutoff;1 preserves full-time exclusion")
    p.add_argument("--separate-vae-decode",action="store_true")
    a=p.parse_args()
    selector_sha=None
    if a.files_from:
        if a.files:p.error("Select one filename-list mechanism")
        raw=a.files_from.read_bytes();a.files=json.loads(raw)
        if not isinstance(a.files,list) or not 1<=len(a.files)<=64 or any(
                not isinstance(value,str) or not value for value in a.files):
            p.error("JSON must contain 1..64 nonempty source filenames")
        selector_sha=hashlib.sha256(raw).hexdigest()
    if a.gpu<0 or not re.fullmatch(r"[a-f0-9]{32}",a.source_job) or not re.fullmatch(r"[a-zA-Z0-9_-]+",a.tag):
        p.error("Explicit completed source job, GPU and simple new tag required")
    if not 0<=a.foveation_mass<=.35:
        p.error("Full-frame sampling mass must be in0..0.35")
    if a.hint_transmission and (not .10<=a.hint_transmission<=.85 or
            not 190<=a.hint_foreground<=255 or not a.fresh_rim_reports or a.foveation_mass):
        p.error("Hint needs bounded assumptions, fresh whole-source proofs and native coordinates")
    if a.lora_path and (not re.fullmatch(r"[a-f0-9]{64}",a.lora_sha256) or a.lora_alpha<=0):
        p.error("Explicit artifact SHA and alpha required for offline adapter fusion")
    if a.sigma_profile=="turturbo3" and (not a.lora_path or
            a.lora_sha256!="fa465274a7bb69687bfd6fd36d3e6cda967efba7a649aab25a37d62adf6286be" or
            a.lora_alpha!=128):
        p.error("Published three-step schedule requires exact pinned Turbo-T2I-3steps weights/alpha128")
    if a.resolution and a.frozen_input_budget:
        p.error("Explicit resolution override cannot be combined with frozen budgets")
    if a.context_margin_cells!=1 and not a.source_flow_context:
        p.error("Explicit latent margin requires source flow context")
    if a.generation_encoder_offload and a.transformer_phase_offload:
        p.error("Select one encoder residency mechanism")
    if a.uniform_steps and (a.base_steps or a.sigma_profile!="native" or a.hint_transmission or a.foveation_mass):
        p.error("Uniform Turbo comparison cannot mix base mode, custom grids or conditioning")
    if not math.isfinite(a.reference_exclusion_max_sigma) or not 0<a.reference_exclusion_max_sigma<=1:
        p.error("Reference exclusion cutoff must be finite in0..1")
    if a.reference_exclusion_max_sigma<1 and (not a.exclude_reference_keys or
            a.exclude_prefix_reference_keys or a.compact_reference_cache):
        p.error("Scheduled exclusion requires an unchanged complete prefix and cache")
    if a.compact_reference_cache and (not a.exclude_reference_keys or not a.kv_cache):
        p.error("Compact reference cache requires reference exclusion and KV cache")
    if (a.exclude_prefix_reference_keys or a.reference_exclusion_margin_cells) and not a.exclude_reference_keys:
        p.error("Prefix exclusion and margins require --exclude-reference-keys")
    if a.exclude_reference_keys and (not a.source_flow_context or not a.fresh_rim_reports):
        p.error("Offline reference exclusion requires source flow context and fresh original rim proofs")
    if a.deployment_uniform_steps and (a.base_steps or a.uniform_steps or a.sigma_profile!="native"
            or a.hint_transmission or a.foveation_mass):
        p.error("Deployment sampler verification requires a standalone whole-frame schedule")
    base_probe=None
    if a.base_steps:
        if a.sigma_profile!="native" or a.lora_path or a.hint_transmission or a.foveation_mass:
            p.error("Base comparison cannot mix adapters, Turbo grids or experimental conditioning")
        from watermark_studio.probe_schedules import validate_base_probe
        base_probe=validate_base_probe(a.model,a.base_steps)
        base_probe["transformer_files_sha256"]={}
        for weight in sorted((a.model/"transformer").glob("*.safetensors")):
            with weight.open("rb") as stream:
                base_probe["transformer_files_sha256"][weight.name]=hashlib.file_digest(stream,"sha256").hexdigest()
        if not base_probe["transformer_files_sha256"]:
            p.error("Exact downloaded base transformer artifacts required")
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
    refreshed={}
    if a.fresh_rim_reports:
        from watermark_studio.probe_conditioning import fresh_rim_reports
        refreshed=fresh_rim_reports(a.fresh_rim_reports,source_root,hashes)
        if a.hint_transmission and set(refreshed)!=set(hashes):
            p.error("Every hinted source needs fresh exact-file source proof")
    captions={}
    if a.caption_job:
        from watermark_studio.probe_conditioning import frozen_source_captions
        captions=frozen_source_captions(state,a.caption_job,hashes)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu);os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
    import torch
    from safetensors.torch import save_file

    from watermark_studio.batching import dimensions, independent_batch, override_methods
    from watermark_studio.config import DeploymentConfig, ProcessingConfig, Region
    from watermark_studio.full_frame_sampling import (
        full_frame_mapping,
        map_region,
        restore_full_frame,
        sample_full_frame,
    )
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
        uniform_steps=a.deployment_uniform_steps,
        compile=a.compile_projections,compile_scope="projections",compile_dynamic=a.compile_projections,
        separate_vae_decode=a.separate_vae_decode,vae_tiling=a.vae_tiling,
        vae_tile_size=1024,vae_tile_stride=768,lora_path=str(a.lora_path) if a.lora_path else "",
        lora_sha256=a.lora_sha256,lora_alpha=a.lora_alpha,lora_scale=a.lora_scale)
    began=time.monotonic();pipe=load_pipeline(cfg,str(a.model),_allow_unconfigured_sampling=bool(a.base_steps))
    if a.generation_encoder_offload:
        from watermark_studio.source_residency import prepare_transformer_host_mirror
        pipe.text_encoder.to("cpu")
        prepare_transformer_host_mirror(pipe.text_encoder)
        pipe.text_encoder.to("cuda:0",non_blocking=True)
        torch.cuda.synchronize()
        pipe._watermark_phase_offload=True
    load_seconds=time.monotonic()-began
    from watermark_studio.probe_schedules import tail_sigma_grid
    original_sigmas=list(getattr(pipe.config,"sample_sigmas",None) or [])
    if a.base_steps and original_sigmas:
        raise ValueError("Base probe cannot discard a stored Turbo schedule")
    uniform_probe=None
    if a.deployment_uniform_steps:
        uniform_probe=pipe._watermark_sampling_info
        original_sigmas=list(uniform_probe["original_sigmas"])
    if a.uniform_steps:
        from watermark_studio.probe_schedules import configure_turbo_uniform_probe
        uniform_probe=configure_turbo_uniform_probe(pipe,a.uniform_steps)
    experimental_sigmas=(None if a.base_steps or a.uniform_steps or a.deployment_uniform_steps else tail_sigma_grid(original_sigmas,a.sigma_profile))
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
        refresh_trace=[]
        if name in refreshed:
            from watermark_studio.probe_conditioning import refresh_verified_rims
            rims,refresh_trace=refresh_verified_rims(source,rims,refreshed[name])
        scoped,cores,invalid=effective_rim_scope(source,regions,row.get("protected_core_boxes",[]),rims,header_core=True)
        if invalid:raise ValueError("Frozen source rim proof invalid")
        budget=a.resolution or (row["model_input_budgets"][0] if a.frozen_input_budget else options.resolution)
        width,height=dimensions(source,budget)
        seed=(options.seed+int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],"big"))%(2**32)
        prompt=build_full_prompt(options.extra_prompt,options.whole_prompt_profile)
        if captions:
            from watermark_studio.probe_conditioning import observed_scene_prompt
            prompt=observed_scene_prompt(prompt,captions[name])
        sampling=(full_frame_mapping(scoped,source.size,a.foveation_mass)
                  if row.get("source_scene")!="document" else None)
        model_source=sample_full_frame(source,sampling)
        model_regions=[map_region(r,sampling) for r in scoped]
        hint_audits=[]
        if a.hint_transmission and row.get("source_scene")!="document":
            from watermark_studio.translucent_hint import translucent_body_hint
            model_source,hint_audits=translucent_body_hint(source,rims,a.hint_transmission,
                foreground=a.hint_foreground,_dense_transmission_experiment=a.hint_transmission<.35)
        item={"image":model_source,"prompt":prompt,"seed":seed}
        if any(r["applied"] for r in hint_audits):
            item["_semantic_reference"]=source
        if a.base_steps or a.uniform_steps:
            item["_probe_num_inference_steps"]=a.base_steps or a.uniform_steps
        if a.source_flow_context:
            item["_denoising_context_regions"]=[r.model_dump() for r in model_regions]
            item["_denoising_context_margin"]=a.context_margin_cells
        if a.exclude_reference_keys:
            item["_reference_exclusion_regions"]=[r.model_dump() for r in model_regions]
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
        from watermark_studio.reference_exclusion import reference_exclusion
        with override_methods(pipe.vae,decode=decode), reference_exclusion(
                pipe,[item],enabled=a.exclude_reference_keys,
                exclude_prefix=a.exclude_prefix_reference_keys,
                margin_cells=a.reference_exclusion_margin_cells,
                compact_cache=a.compact_reference_cache,
                max_sigma=a.reference_exclusion_max_sigma) as reference_audit:
            generated=restore_full_frame(independent_batch(pipe,[item],budget,a.kv_cache)[0],source.size,sampling)
        if (a.base_steps or a.uniform_steps or a.deployment_uniform_steps) and len(pipe.scheduler.timesteps)!=(a.base_steps or a.uniform_steps or a.deployment_uniform_steps):
            raise ValueError("Actual diffusion forward count differs from explicit requested steps")
        if a.exclude_reference_keys:
            if reference_audit["forward_calls"]!=len(pipe.scheduler.timesteps):
                raise ValueError("Reference exclusion must cover every and only the requested denoising forwards")
            capture["reference_key_exclusion"]=reference_audit
        if sampling is not None:
            capture["full_frame_sampling"]=sampling
            capture["forward_source_rgb_sha256"]=hashlib.sha256(model_source.tobytes()).hexdigest()
        if hint_audits:
            capture["translucent_conditioning"]={"transmission":a.hint_transmission,
                "foreground":a.hint_foreground,"audits":hint_audits,
                "source_rgb_sha256":hashlib.sha256(model_source.tobytes()).hexdigest(),
                "quality_permission":False}
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
        result["source_rim_protection"]=rims
        if refresh_trace:
            result["source_rim_refresh"]=refresh_trace
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
                "base_model":base_probe,"uniform_turbo":uniform_probe,"encoder_residency":item.get("_encoder_generation_log"),"sigma_profile":a.sigma_profile,"steps":len(pipe.scheduler.timesteps),
                "full_frame_sampling":sampling})
        with (folder/"results.jsonl").open("a") as f:f.write(json.dumps(result,ensure_ascii=False)+"\n")
        print(json.dumps({"completed":index+1,"total":len(rows),"job":job}),flush=True)
    if any(digest(source_root/name)!=value for name,value in hashes.items()):
        raise ValueError("Original source changed")
    if revision!=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip():
        raise ValueError("Code changed during probe")
    report={"state":"completed","job":job,"source_job":a.source_job,"input_dir":str(source_root),"output_dir":str(outputs),
        "total":len(rows),"file_selector_sha256":selector_sha,"options":options.model_dump(),"code_revision":revision,"sources_sha256":hashes,
        "seconds":round(time.monotonic()-began,3),"load_seconds":round(load_seconds,3),"quality_evaluated":False,
        "native_settings":{"base_model":base_probe,"uniform_turbo":uniform_probe,"quantization":a.quantization,"granularity":a.granularity,"kv_cache":a.kv_cache,
            "compile":a.compile_projections,"transformer_phase_offload":a.transformer_phase_offload,
            "sigma_profile":a.sigma_profile,"frozen_input_budget":a.frozen_input_budget,"foveation_mass":a.foveation_mass,"resolution_override":a.resolution,
            "generation_encoder_offload":a.generation_encoder_offload,"vae_tiling":a.vae_tiling,
            "hint_transmission":a.hint_transmission,"hint_foreground":a.hint_foreground,
            "fresh_rim_reports":[str(v) for v in a.fresh_rim_reports or []],
            "image_adapter":getattr(pipe,"_image_adapter",None),"caption_job":a.caption_job,"original_sigmas":original_sigmas,"actual_sigmas":experimental_sigmas,
            "source_flow_context":a.source_flow_context,"context_margin_cells":a.context_margin_cells,
            "exclude_reference_keys":a.exclude_reference_keys,
            "exclude_prefix_reference_keys":a.exclude_prefix_reference_keys,
            "reference_exclusion_margin_cells":a.reference_exclusion_margin_cells,
            "compact_reference_cache":a.compact_reference_cache,
            "reference_exclusion_max_sigma":a.reference_exclusion_max_sigma,"deployment_uniform_steps":a.deployment_uniform_steps,"separate_vae_decode":a.separate_vae_decode},
        "server_isolation":isolation,"scope":"Frozen-source native whole-photo GEN1 with exact VAE-input capture"}
    (folder/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (work/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:report[k] for k in ("job","total","seconds","load_seconds")}),flush=True)


if __name__=="__main__":
    main()
