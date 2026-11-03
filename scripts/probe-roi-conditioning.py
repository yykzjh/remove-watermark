"""First-generation paired ROI conditioning probe; never retries or promotes QA results."""
import argparse
import collections
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

p=argparse.ArgumentParser(__doc__)
p.add_argument("--gpu",type=int,required=True)
p.add_argument("--variant",choices=["raw","raw-opaque","prefill","split"],required=True)
p.add_argument("--baseline-job",required=True)
p.add_argument("--selection",type=Path,required=True)
p.add_argument("--work",type=Path,required=True)
p.add_argument("--prompt-profile",choices=["detailed","compact-zh","detailed-zh","precise-en"],default="detailed")
p.add_argument("--source-material-guide",action="store_true")
p.add_argument("--source-edge-small-panels",action="store_true")
p.add_argument("--tone-step-repair",action="store_true")
p.add_argument("--qwen-glyph-stencil",action="store_true")
p.add_argument("--source-transmission-hint",action="store_true")
p.add_argument("--context",type=float,default=.45)
p.add_argument("--repair-blending",choices=["feather","harmonic","harmonic-fine"])
p.add_argument("--model-resolution",type=int,choices=[512,768,1024,1536],default=1024)
p.add_argument("--sigma-profile",choices=["native","tail9","tail10","tail12","mid10","mid12"],default="native")
p.add_argument("--uniform-steps",type=int,choices=[0,12,16,20,24,40],default=0)
p.add_argument("--roi-resolution",type=int,choices=[0,1536],default=0)
p.add_argument("--roi-steps",type=int,choices=[0,12,16,20,24,40],default=0)
p.add_argument("--roi-step-min-area",type=float,default=.14)
p.add_argument("--float8-granularity",choices=["tensor","row"],default="tensor")
p.add_argument("--quantization",choices=["none","float8","int8"],default="float8")
p.add_argument("--batch-size",type=int,choices=[1,2],default=2)
p.add_argument("--quality-phase-offload",action="store_true")
p.add_argument("--roi-context-margin",type=int,choices=range(-1,17),default=-1,
               help="Offline original-ROI source-flow context; -1 disables, otherwise latent-cell buffer")
p.add_argument("--float8-accurate-accum",action="store_true")
p.add_argument("--float8-amax-floor",type=float,default=0)
p.add_argument("--tiny-inpaint-path",type=Path)
p.add_argument("--tiny-inpaint-sha256",default="")
p.add_argument("--tiny-inpaint-scope",choices=["tiny","all"],default="tiny")
p.add_argument("--tiny-inpaint-mask",choices=["box","white-glyph"],default="box")
p.add_argument("--lora-path",type=Path)
p.add_argument("--lora-sha256",default="")
p.add_argument("--lora-alpha",type=float,default=0)
p.add_argument("--lora-scale",type=float,default=1)
a=p.parse_args()
assert .1<=a.context<=2
assert a.sigma_profile=="native" or not (a.uniform_steps or a.roi_steps), "Explicit sigma and uniform/area schedules are mutually exclusive"
assert bool(a.lora_path)==bool(a.lora_sha256), "Pin an explicit adapter SHA"
assert bool(a.tiny_inpaint_path)==bool(a.tiny_inpaint_sha256), "Pin auxiliary weights"
loaded_revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
assert a.gpu in (0,1,2,3)
assert not a.work.exists(), "Use a fresh output directory"
names=json.loads(a.selection.read_text())
assert 1<=len(names)<=128 and len(set(names))==len(names)
os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch

from watermark_studio.config import DeploymentConfig, ProcessingConfig
from watermark_studio.images import load_photo
from watermark_studio.quality_loading import load_quality_encoder, load_quality_processor
from watermark_studio.worker import load_pipeline, process_batch

if a.prompt_profile=="compact-zh":
    from watermark_studio import worker
    from watermark_studio.prompts import build_compact_focused_prompt
    worker.build_focused_prompt=build_compact_focused_prompt
    worker.build_small_overlay_prompt=lambda box,crop,extra="":build_compact_focused_prompt("panel",box,crop,extra)

if a.prompt_profile=="detailed-zh":
    from watermark_studio import worker
    from watermark_studio.prompts import (
        build_detailed_chinese_focused_prompt,
        build_detailed_chinese_small_prompt,
    )
    worker.build_focused_prompt=build_detailed_chinese_focused_prompt
    worker.build_small_overlay_prompt=build_detailed_chinese_small_prompt

if a.prompt_profile=="precise-en":
    from watermark_studio import worker
    from watermark_studio.prompts import (
        build_precise_english_focused_prompt,
        build_precise_english_small_prompt,
    )
    worker.build_focused_prompt=build_precise_english_focused_prompt
    worker.build_small_overlay_prompt=build_precise_english_small_prompt

guidance_audits=[]
context_mappings=[]
transmission_audits=[]
stencil_audits=[]
if a.source_transmission_hint:
    from watermark_studio import roi_grouping
    from watermark_studio.translucent_recovery import recover_translucent_body
    source_coalesce=roi_grouping.coalesce_edits
    def hinted_coalesce(edits,mode):
        for edit in edits:
            state=edit["state"]
            if state["whole_generation"] or state["source_scene"]!="outdoor":
                continue
            source=state["source"];l,t,r,b=edit["box"]
            if (r-l)*(b-t)<.035*source.width*source.height:
                continue
            hint,audit=recover_translucent_body(source,source,
                (l/source.width,t/source.height,r/source.width,b/source.height),
                _native_padding_search=True)
            transmission_audits.append({"file":state["task"]["relative"],**audit,
                                         "quality_permission":False,"final_source_pixels":False})
            if audit["applied"]:
                edit["_semantic_reference"]=source.crop(edit["crop"])
                edit["image"]=hint.crop(edit["crop"])
        return source_coalesce(edits,mode)
    roi_grouping.coalesce_edits=hinted_coalesce

if a.qwen_glyph_stencil:
    from PIL import Image

    from watermark_studio import roi_grouping, worker
    from watermark_studio.tiny_inpaint import tiny_eligible, white_glyph_support
    stencil_scopes=set()
    stencil_coalesce=roi_grouping.coalesce_edits
    def register_stencils(edits,mode):
        grouped=stencil_coalesce(edits,mode)
        for edit in grouped:
            if tiny_eligible(edit):
                stencil_scopes.add((id(edit["state"]["source"]),tuple(edit["box"]),tuple(edit["crop"])))
        return grouped
    roi_grouping.coalesce_edits=register_stencils
    original_blend=worker.blend_patch
    def stencil_blend(original,patch,box,crop,*args,**kwargs):
        result=original_blend(original,patch,box,crop,*args,**kwargs)
        reference=kwargs.get("reference_original",original)
        if (id(reference),tuple(box),tuple(crop)) not in stencil_scopes:
            return result
        # Select the repair mask from the untouched original, before reading
        # the candidate; no image-level candidate choice or quality waiver.
        mask,kind=white_glyph_support(reference,box)
        stencil_audits.append({"box":list(box),"method":kind,"quality_permission":False})
        if kind!="white_glyph_source_stencil":
            return result
        final=Image.composite(result,original,mask)
        final.info.update(result.info)
        return final
    worker.blend_patch=stencil_blend

if a.source_material_guide or a.roi_context_margin>=0 or a.roi_resolution:
    from watermark_studio import roi_grouping
    from watermark_studio.roi_guidance import describe_visible_context
    original_coalesce=roi_grouping.coalesce_edits
    def observed_coalesce(edits,mode):
        grouped=original_coalesce(edits,mode)
        if a.roi_resolution:
            from watermark_studio.batching import dimensions
            for edit in grouped:
                state=edit["state"];box=edit["box"]
                area=(box[2]-box[0])*(box[3]-box[1])
                if not state["whole_generation"] and area>=a.roi_step_min_area*state["source"].width*state["source"].height:
                    edit["resolution"]=a.roi_resolution
                    edit["size"]=dimensions(edit["image"],a.roi_resolution)
        if a.source_material_guide:
            guidance_audits.extend(describe_visible_context(grouped,pipe.text_encoder,pipe.processor,pipe._execution_device))
        if a.roi_context_margin>=0:
            from roi_context_probe import prepare_roi_context
            context_mappings.extend(prepare_roi_context(grouped,a.roi_context_margin))
        return grouped
    roi_grouping.coalesce_edits=observed_coalesce

assert torch.cuda.mem_get_info()[0]>40*2**30, "GPU must be free before probe"
state=Path.home()/".local/state/watermark-studio"
b=state/a.baseline_job
summary=json.loads((b/"summary.json").read_text())
original={r["file"]:r for r in map(json.loads,(b/"results.jsonl").read_text().splitlines())}
assert set(names)<=set(original)
cfg_values=json.loads(Path(".run/deployment.json").read_text())
cfg=DeploymentConfig.model_validate({**cfg_values,
            "gpus":[a.gpu],"uniform_steps":a.uniform_steps,"quantization":a.quantization,
            "batch_size":a.batch_size,"small_batch_size":a.batch_size,
            **({"tone_step_repair":True} if a.tone_step_repair else {}),
            **({"source_edge_small_panels":True} if a.source_edge_small_panels else {}),
            **({"lora_path":str(a.lora_path),"lora_sha256":a.lora_sha256,
                "lora_alpha":a.lora_alpha,"lora_scale":a.lora_scale} if a.lora_path else {}),
            **({"quality_phase_offload":True} if a.quality_phase_offload else {}),"float8_granularity":a.float8_granularity,"float8_accurate_accum":a.float8_accurate_accum,"float8_amax_floor":a.float8_amax_floor,
            "roi_steps":a.roi_steps,"roi_step_min_area":a.roi_step_min_area,
            "prefill_overlays":a.variant not in {"raw","raw-opaque"},
            "protection_core_feather":False if a.variant=="raw-opaque" else cfg_values["protection_core_feather"],
            "prefill_original_semantics":a.variant=="split"})
opts=ProcessingConfig.model_validate(summary["options"]).model_copy(
    update={"quality_auto_retry":False,"quality_cross_review":False,"overwrite":False,"context":a.context,
        "resolution":a.model_resolution,**({"repair_blending":a.repair_blending} if a.repair_blending else {})})
assert opts.first_pass_mode=="complete" and opts.context==a.context and opts.resolution==a.model_resolution
a.work.mkdir(parents=True)
out=a.work/"output"
hashes={}
tasks=[]
root=Path(summary["input_dir"]).resolve()
for name in names:
    src=(root/name).resolve()
    assert src.is_relative_to(root) and src.is_file()
    with src.open("rb") as f:hashes[name]=hashlib.file_digest(f,"sha256").hexdigest()
    dst=out/name;dst.parent.mkdir(parents=True,exist_ok=True)
    tasks.append({"source":str(src),"destination":str(dst),"relative":name})
cache=Path(cfg.cache_dir)/"models"
model=cache/"Qwen--Qwen-Image-2.1-Turbo/snapshots/master"
load_start=time.monotonic()
pipe=load_pipeline(cfg,str(model))
if a.sigma_profile!="native":
    from watermark_studio.probe_schedules import tail_sigma_grid
    original_sigmas=list(pipe.config.sample_sigmas)
    sigmas=tail_sigma_grid(original_sigmas,a.sigma_profile)
    pipe.register_to_config(sample_sigmas=sigmas)
    pipe._watermark_sampling_info={"sampling":"offline_native_nodes_plus_refinement",
        "profile":a.sigma_profile,"original_sigmas":original_sigmas,"sigmas":sigmas,
        "checkpoint_modified":False,"quality_permission":False}

pipe._quality_encoder=load_quality_encoder(cfg.quality_model,cfg,pipe._execution_device)
pipe._quality_processor=load_quality_processor(cfg.quality_model)
pipe._quality_weights_relation="different_weights"
if a.tiny_inpaint_path:
    from watermark_studio import batching
    from watermark_studio.tiny_inpaint import TinyInpainter, tiny_eligible
    auxiliary = TinyInpainter(a.tiny_inpaint_path, a.tiny_inpaint_sha256, pipe._execution_device, a.tiny_inpaint_mask)
    original_batch = batching.independent_batch
    def routed_batch(pipeline, items, resolution, kv_cache):
        # No candidate contest or fallback: each ROI runs exactly one chosen backend.
        chosen = [tiny_eligible(item, 1 if a.tiny_inpaint_scope=="all" else .035,
                                1536 if a.tiny_inpaint_scope=="all" else 512) for item in items]
        patches = [None]*len(items)
        ordinary = [item for item, route in zip(items, chosen) if not route]
        if ordinary:
            generated = iter(original_batch(pipeline, ordinary, resolution, kv_cache))
        for index, (item, route) in enumerate(zip(items, chosen)):
            patches[index] = auxiliary(item) if route else next(generated)
        return patches
    batching.independent_batch = routed_batch

print(json.dumps({"phase":"loaded","variant":a.variant,"seconds":time.monotonic()-load_start}),flush=True)
started=time.monotonic();rows=[];limits={}
for start in range(0,len(tasks),cfg.batch_window):
    group=tasks[start:start+cfg.batch_window]
    results=process_batch(group,opts,cfg,pipe,limits)
    assert len(results)==len(group)
    with (a.work/"results.jsonl").open("a") as f:
        for row in results:f.write(json.dumps(row,ensure_ascii=False)+"\n")
    rows.extend(results)
    print(json.dumps({"phase":"progress","variant":a.variant,"done":len(rows),
        "counts":dict(collections.Counter(r.get("quality",{}).get("status","error") for r in rows))}),flush=True)
bad_sizes=0
for task in tasks:
    with open(task["source"],"rb") as f:assert hashlib.file_digest(f,"sha256").hexdigest()==hashes[task["relative"]]
    if Path(task["destination"]).exists():
        bad_sizes+=load_photo(task["source"])[0].size!=load_photo(task["destination"])[0].size
counts=dict(collections.Counter(r.get("quality",{}).get("status","error") for r in rows))
report={"sigma_profile":a.sigma_profile,"qwen_glyph_stencil":a.qwen_glyph_stencil,"source_transmission_hint":a.source_transmission_hint,"stencil_audits":stencil_audits,"transmission_audits":transmission_audits,"variant":a.variant,"roi_resolution":a.roi_resolution,"roi_context_margin":a.roi_context_margin,"context_mappings":context_mappings,"source_material_guide":a.source_material_guide,"guidance_audits":guidance_audits,"prompt_profile":a.prompt_profile,"images":len(rows),"seconds":time.monotonic()-started,
    "counts":counts,"errors":sum(not r.get("ok") for r in rows),"bad_sizes":bad_sizes,
    "source_sha256":hashes,"selection_sha256":hashlib.sha256(a.selection.read_bytes()).hexdigest(),
    "config":cfg.model_dump(mode="json"),"options":opts.model_dump(mode="json"),
    "revision":loaded_revision,
    "baseline_review_to_pass":sum(original[r["file"]]["quality"]["status"]!="passed" and r.get("quality",{}).get("status")=="passed" for r in rows),
    "baseline_pass_to_review":sum(original[r["file"]]["quality"]["status"]=="passed" and r.get("quality",{}).get("status")!="passed" for r in rows),
    "actual_steps":dict(collections.Counter(log.get("actual_steps") for row in rows for log in row.get("model_sampling",[]))),
    "sampling":pipe._watermark_sampling_info,"single_gpu":True,"first_generation_only":True,"development_reuse":True}
(a.work/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in {"config","options","source_sha256","guidance_audits","context_mappings"}}),flush=True)
