from __future__ import annotations

import hashlib
import os
import time
import traceback

from .config import DeploymentConfig, ProcessingConfig, Region
from .detection import confirm_source_hints, locate_watermarks_batch
from .images import (
    atomic_copy,
    blend_patch,
    composite_protected,
    context_box,
    load_photo,
    pixel_box,
    prefill_overlay_reference,
    regions_for,
    save_output,
)
from .prompts import (
    build_focused_prompt,
    build_full_prompt,
    build_gray_outpaint_prompt,
    build_prefilled_full_prompt,
    build_prefilled_prompt,
    build_prompt,
    build_small_overlay_prompt,
    scene_instruction,
)


def load_pipeline(config, model_path, *, _allow_unconfigured_sampling=False):
    import torch
    from diffusers import QwenImage21Pipeline

    torch.set_num_threads(config.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = config.tf32
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查 NVIDIA 驱动及 PyTorch CUDA 环境")
    rim_detector = None
    if config.protect_source_rims:
        from .text_presence import TargetTextPresence
        rim_detector = TargetTextPresence()
        rim_detector._engine()  # Fail before loading GPU weights if the extra is missing.
    dtype = getattr(torch, config.dtype)
    pipe = QwenImage21Pipeline.from_pretrained(model_path, dtype=dtype, local_files_only=True)
    if _allow_unconfigured_sampling and getattr(pipe.config,"sample_sigmas",None):
        raise ValueError("A base-only probe cannot replace the stored Turbo sampling grid")
    pipe._image_adapter = {}
    if config.lora_path:
        from .adapters import fuse_cpu_adapter
        pipe._image_adapter=fuse_cpu_adapter(pipe.transformer,config.lora_path,
            scale=config.lora_scale,alpha=config.lora_alpha,expected_sha256=config.lora_sha256)
    pipe.set_progress_bar_config(disable=True)
    if config.vae_tiling:
        pipe.vae.enable_tiling(
            tile_sample_min_height=config.vae_tile_size,
            tile_sample_min_width=config.vae_tile_size,
            tile_sample_stride_height=config.vae_tile_stride,
            tile_sample_stride_width=config.vae_tile_stride,
        )
    if config.vae_slicing:
        pipe.vae.enable_slicing()
    if config.transformer_phase_offload:
        from .source_residency import prepare_transformer_host_mirror
        prepare_transformer_host_mirror(pipe.transformer)
        prepare_transformer_host_mirror(pipe.text_encoder)
    if config.offload == "model":
        pipe.enable_model_cpu_offload(gpu_id=0)
    elif config.offload == "sequential":
        pipe.enable_sequential_cpu_offload(gpu_id=0)
    else:
        pipe.to("cuda:0")
    if config.vae_channels_last:
        # Convert convolution weights only, not unrelated temporal/linear tensors.
        torch.nn.utils.convert_conv2d_weight_memory_format(pipe.vae, torch.channels_last)
        torch.nn.utils.convert_conv3d_weight_memory_format(pipe.vae, torch.channels_last_3d)
    if config.attention == "sdpa":
        from .attention import IndependentSDPAProcessor
        backend = config.attention_backend
        if backend.startswith("native_"):
            backend = "_" + backend
        # Selecting a dispatch name alone does not initialize Hub kernels.
        # The official context manager validates dependencies and downloads once.
        from diffusers.models.attention_dispatch import attention_backend
        with attention_backend(backend):
            pass
        pipe.transformer.set_attn_processor(IndependentSDPAProcessor(backend))
    if config.attention == "flex":
        from diffusers.models.transformers.transformer_qwenimage21 import (
            QwenImage21FlexAttnProcessor,
        )

        pipe.transformer.set_attn_processor(QwenImage21FlexAttnProcessor())
    if config.fuse_qkv:
        from .optimizations import fuse_projections
        fuse_projections(pipe.transformer)
    if config.quantization != "none":
        from .optimizations import quantize_transformer
        quantize_transformer(pipe.transformer, config.quantization, config.float8_granularity)
    pipe._watermark_phase_offload=config.transformer_phase_offload
    pipe._source_rim_text_detector=rim_detector
    pipe._watermark_separate_vae_decode=config.separate_vae_decode
    pipe._watermark_prompt_encoder = config.prompt_encoder
    pipe._watermark_encoder_batch_size = config.encoder_batch_size
    if config.compile:
        if config.compile_scope == "projections":
            # Keep KV mutation, masks and Python segment branches outside compilation.
            for block in pipe.transformer.transformer_blocks:
                attn = block.attn
                linears = [attn.to_qkv] if hasattr(attn, "to_qkv") else [attn.to_q, attn.to_k, attn.to_v]
                for module in [*linears, attn.to_out[0], block.img_mlp]:
                    module.compile(mode=config.compile_mode, dynamic=config.compile_dynamic)
        else:
            compile_fn = (pipe.transformer.compile_repeated_blocks if config.compile_scope == "blocks"
                          else pipe.transformer.compile)
            compile_fn(mode=config.compile_mode, dynamic=config.compile_dynamic)
    if not getattr(pipe.config, "sample_sigmas", None) and not _allow_unconfigured_sampling:
        raise RuntimeError("模型缺少 Turbo sample_sigmas；请检查模型及 Diffusers 版本")
    if config.uniform_steps and _allow_unconfigured_sampling:
        raise ValueError("Deployment uniform sampling cannot be mixed with a base-only model probe")
    pipe._watermark_uniform_steps=config.uniform_steps
    if config.uniform_steps:
        from .sampling import configure_uniform_sampling
        pipe._watermark_sampling_info=configure_uniform_sampling(pipe,config.uniform_steps)
    else:
        pipe._watermark_sampling_info={"sampling":"checkpoint_sample_sigmas",
            "requested_steps":len(getattr(pipe.config,"sample_sigmas",None) or []),
            "checkpoint_modified":False}
    if config.roi_steps:
        if _allow_unconfigured_sampling:
            raise ValueError("ROI sampling routing requires the stored Turbo checkpoint")
        from .roi_sampling import configure_roi_sampling
        configure_roi_sampling(pipe,config.roi_steps,config.roi_step_min_area)
    return pipe


def needs_source_detection(options):
    """Share original-image detection and carrier protection across auto routes."""
    return ((options.scope=="full_image" and options.preserve_background)
            or (options.scope=="regions" and options.preset=="auto"))


def process_batch(tasks, options, config, pipe, batch_limits=None):
    """Bounded lookahead; independently detect files, then batch compatible crop edits."""
    from collections import defaultdict

    from .batching import dimensions, independent_batch

    start = time.monotonic()
    batch_limits = {} if batch_limits is None else batch_limits
    states, results, edits = [], {}, []
    source_images, existing_images = {}, {}
    cuda = config.backend != "mock"
    execution_device = pipe._execution_device if cuda else "cpu"
    focused_auto = options.first_pass_mode != "whole" and (
        options.scope == "full_image" or options.preset == "auto"
    )
    if cuda:
        import torch
        torch.cuda.reset_peak_memory_stats()
    quality_transfer_seconds=0.
    phase_manage=(cuda and (config.quality_phase_offload or config.quality_phase_offload_above or config.transformer_phase_offload)
                  and config.quality_backend=="independent" and config.quality_quantization=="none"
                  and hasattr(pipe,"_quality_encoder"))
    phase_offload=(cuda and (config.quality_phase_offload or config.transformer_phase_offload or
                   (config.quality_phase_offload_above and options.resolution>config.quality_phase_offload_above))
                   and config.quality_backend=="independent"
                   and config.quality_quantization=="none" and hasattr(pipe,"_quality_encoder"))
    detections, decoded = {}, {}
    detail_counts = {}
    edge_evidence = {}
    context_evidence = {}
    source_geometry_requeries = {}
    physical_audits = {}
    source_rim_audits = {}
    gray_certificates = {}
    needs_detection = cuda and needs_source_detection(options)
    from .source_residency import transformer_source_phase
    source_transformer_transfer={}
    def before_transformer_restore():
        nonlocal quality_transfer_seconds
        if phase_offload:
            from .quality_loading import park_quality_encoder
            began=time.monotonic()
            pipe._quality_encoder=park_quality_encoder(pipe._quality_encoder)
            torch.cuda.empty_cache()
            quality_transfer_seconds+=time.monotonic()-began
    with transformer_source_phase(pipe, cuda and config.transformer_phase_offload
                                  and (needs_detection or options.whole_prompt_profile=="scene"),
                                  execution_device, source_transformer_transfer,
                                  before_restore=before_transformer_restore):
        if needs_detection:
            detect_tasks = []
            for task in tasks:
                if task.get("reuse_existing"):
                    continue
                try:
                    decoded[task["relative"]] = load_photo(task["source"])
                    detect_tasks.append(task)
                except (OSError, ValueError):
                    pass  # Normal per-file handling below records the decoding failure.
            # The independent verifier is unused by this editor-only scan.
            # Park it before the scan to preserve source-stage memory headroom.
            if phase_offload and detect_tasks:
                from .quality_loading import park_quality_encoder
                transfer_started=time.monotonic()
                pipe._quality_encoder=park_quality_encoder(pipe._quality_encoder)
                torch.cuda.empty_cache()
                quality_transfer_seconds+=time.monotonic()-transfer_started
            started_detection = time.monotonic()
            found = locate_watermarks_batch(
                [decoded[t["relative"]][0] for t in detect_tasks],
                pipe.text_encoder, pipe.processor, execution_device,
                options.detection_resolution,config.detection_batch_size,True,
            )
            protected_details=(config.protection_detail_scan and options.scope=="full_image"
                               and options.preserve_background and options.first_pass_mode=="whole")
            if protected_details:
                from .detection import DETECTION_PROMPT, DetectionAreaError
                for i,result in enumerate(found):
                    if isinstance(result,DetectionAreaError):
                        retried=locate_watermarks_batch([decoded[detect_tasks[i]["relative"]][0]],
                            pipe.text_encoder,pipe.processor,execution_device,options.detection_resolution,1,True,
                            DETECTION_PROMPT+"\nInspect this ORIGINAL photograph afresh. Discard the previous oversized box. "
                            "Physical documents/inspection boards are not digital overlays. Locate only the real "
                            "digitally superimposed groups; keep disconnected groups separate. Never clamp boxes, "
                            "invent empty results or weaken area limits to avoid an error.")[0]
                        found[i]=retried
                        source_geometry_requeries[detect_tasks[i]["relative"]]={
                            "attempts":1,"initial_error":str(result)[:160],
                            "status":"error" if isinstance(retried,Exception) else "valid","same_area_limit":True}
            if options.first_pass_mode in {"thorough","complete"} or protected_details:
                from .detection import refine_source_details
                edge_audit={}
                context_audit={}
                found, counts = refine_source_details([decoded[t["relative"]][0] for t in detect_tasks],found,
                    pipe.text_encoder,pipe.processor,execution_device,options.detection_resolution,config.detection_batch_size,
                    complete=options.first_pass_mode=="complete" or protected_details,edge_audit=edge_audit,context_audit=context_audit)
                detail_counts={t["relative"]:counts[i] for i,t in enumerate(detect_tasks)}
                edge_evidence={t["relative"]:edge_audit.get(i,[]) for i,t in enumerate(detect_tasks)}
                context_evidence={t["relative"]:context_audit.get(i,[]) for i,t in enumerate(detect_tasks)}
            for index, task in enumerate(detect_tasks):
                if task.get("_extra_regions") and not isinstance(found[index], Exception):
                    try:
                        regions, text = found[index]
                        found[index] = (confirm_source_hints(
                            decoded[task["relative"]][0], regions, task["_extra_regions"],
                            pipe.text_encoder, pipe.processor, execution_device,
                            max(1024, options.quality_resolution), config.detection_batch_size,
                        ), text)
                    except (ValueError, TypeError, RuntimeError) as error:
                        found[index] = error
            if (config.protection_physical_vet and needs_detection
                    and config.quality_backend=="independent" and hasattr(pipe,"_quality_encoder")):
                from .quality_loading import restore_quality_encoder
                from .source_vetting import vet_sources
                if str(getattr(pipe._quality_encoder,"device","")).startswith("cpu"):
                    pipe._quality_encoder=restore_quality_encoder(pipe._quality_encoder,execution_device)
                found,vet_audits=vet_sources(
                    [decoded[t["relative"]][0] for t in detect_tasks],found,pipe.text_encoder,pipe.processor,
                    pipe._quality_encoder,pipe._quality_processor,execution_device,
                    getattr(pipe,"_quality_weights_relation","unknown"),config.source_vet_batch_size,
                    recheck=config.source_carrier_recheck)
                physical_audits={t["relative"]:vet_audits.get(index,[]) for index,t in enumerate(detect_tasks)}
            if config.protect_source_rims and options.scope=="full_image" and options.preserve_background and options.first_pass_mode=="whole":
                from .detection import source_core_boxes
                from .quality_loading import restore_quality_encoder
                from .source_panel_rims import inspect_photo_rims, propose_blue_card_rims
                if str(getattr(pipe._quality_encoder,"device","")).startswith("cpu"):
                    pipe._quality_encoder=restore_quality_encoder(pipe._quality_encoder,execution_device)
                for index,task in enumerate(detect_tasks):
                    detection=found[index]
                    if isinstance(detection,Exception):
                        continue
                    photo=decoded[task["relative"]][0]
                    cores=source_core_boxes(detection[1])
                    core_rows=[{"mask":list(mask),"core":list(core)} for mask,core in cores.items()]
                    proposals=propose_blue_card_rims(photo,detection[0],core_rows,options.feather)
                    audit=inspect_photo_rims(photo,proposals,pipe.text_encoder,pipe.processor,
                        pipe._quality_encoder,pipe._quality_processor,execution_device,
                        pipe._quality_weights_relation,pipe._source_rim_text_detector)
                    source_rim_audits[task["relative"]]=audit
                    blocked=[]
                    for row in audit:
                        if any(report.get("digital_card") is False and report.get("uncertain") is False
                               for report in [row.get("editor",{}),row.get("independent",{})]):
                            blocked.append(tuple(row["mask"]))
                            physical_audits.setdefault(task["relative"],[]).append({
                                "box":row["mask"],"status":"uncertain_physical_carrier",
                                "version":"source-rim-identity-v1","requires_review":True,
                                "reason":"Source digital identity was not established; preserve original pixels"})
                    if blocked:
                        found[index]=([r for r in detection[0] if tuple(r.box) not in blocked],detection[1])
            if (config.prefill_overlays and config.prefill_style=="gray"
                    and options.scope=="full_image" and options.preserve_background and options.first_pass_mode=="whole"):
                from .source_digital_proof import certify_gray_regions, gray_source_cores
                for index,task in enumerate(detect_tasks):
                    detection=found[index]
                    if isinstance(detection,Exception):
                        continue
                    audit=certify_gray_regions(decoded[task["relative"]][0],detection[0],
                        gray_source_cores(detection[1]),source_rim_audits.get(task["relative"],[]),
                        pipe.text_encoder,pipe.processor,pipe._quality_encoder,pipe._quality_processor,
                        execution_device,pipe._quality_weights_relation)
                    gray_certificates[task["relative"]]=audit
                    blocked={tuple(item["mask"]) for item in audit if item.get("approved") is not True}
                    if blocked:
                        found[index]=([r for r in detection[0] if tuple(r.box) not in blocked],detection[1])
                        physical_audits.setdefault(task["relative"],[]).extend(
                            item|{"box":item["mask"],"status":"uncertain_gray_source_overlay"}
                            for item in audit if item.get("requires_review"))
            elapsed = (time.monotonic()-started_detection)/max(1, len(detect_tasks))
            detections = {t["relative"]: (r, elapsed) for t, r in zip(detect_tasks, found)}
        source_briefs={}
        if cuda and options.scope=="full_image" and options.first_pass_mode=="whole" and options.whole_prompt_profile=="scene":
            from .source_brief import inspect_sources
            brief_tasks=[]
            for task in tasks:
                if task.get("reuse_existing"):
                    continue
                try:
                    if task["relative"] not in decoded:
                        decoded[task["relative"]]=load_photo(task["source"])
                    brief_tasks.append(task)
                except (OSError,ValueError):
                    pass
            briefs=inspect_sources([decoded[t["relative"]][0] for t in brief_tasks],
                pipe.text_encoder,pipe.processor,execution_device,config.detection_batch_size)
            source_briefs={t["relative"]:brief for t,brief in zip(brief_tasks,briefs)}
    for task in tasks:
        state = None
        try:
            source, exif, icc, alpha = (decoded.pop(task["relative"]) if task["relative"] in decoded
                                       else load_photo(task["source"]))
            if options.quality_check:
                source_images[task["relative"]] = source
            if task.get("reuse_existing"):
                existing = load_photo(task["destination"])[0]
                existing_images[task["relative"]] = existing
                from PIL import Image
                with Image.open(task["destination"]) as stored:
                    stored_size = list(stored.size)
                results[task["relative"]] = {
                    "file": task["relative"], "ok": True, "action": "existing_output_verified",
                    "regions": [], "detection_seconds": 0, "output_size": stored_size,
                    **({"background_preserved":True,
                        "repair_regions":task["_repair_provenance"]["regions"],
                        "source_scene":task["_repair_provenance"].get("scene","unknown"),
                        "harmonization":task["_repair_provenance"].get("harmonization"),
                        "protected_core_boxes":task["_repair_provenance"].get("protected_core_boxes",[]),
                        "source_rim_protection":task["_repair_provenance"].get("source_rim_protection",[]),
                        "source_rim_composition":task["_repair_provenance"].get("source_rim_composition",{}),
                        "denoising_context":task["_repair_provenance"].get("denoising_context",[])}
                       if task.get("_repair_provenance") else {}),
                }
                continue
            detection_start = time.monotonic()
            protected_regions = None
            if not cuda:
                regions = []
            elif options.scope == "full_image":
                if options.preserve_background:
                    found, _ = detections[task["relative"]]
                    if isinstance(found, Exception):
                        raise found
                    protected_regions, _ = found
                    regions = ([Region(box=(0, 0, 1, 1))] if options.first_pass_mode == "whole"
                               else protected_regions) if protected_regions else []
                else:
                    regions = [Region(box=(0, 0, 1, 1))]
            elif options.preset == "auto":
                found, _ = detections[task["relative"]]
                if isinstance(found, Exception):
                    raise found
                regions, _ = found
                regions = [r for r in regions if options.remove_logo or r.kind != "logo"]
            else:
                regions = regions_for(source, options)
            detected = round(detections[task["relative"]][1] if task["relative"] in detections
                             else time.monotonic() - detection_start, 3)
            if not regions:
                if options.output_width:
                    if alpha is not None:
                        source.putalpha(alpha)
                    output_size = save_output(source, task["source"], task["destination"], exif, icc, options)
                else:
                    atomic_copy(task["source"], task["destination"], options.overwrite)
                    from PIL import Image
                    with Image.open(task["source"]) as stored:
                        output_size = list(stored.size)
                results[task["relative"]] = {
                    "file": task["relative"], "ok": True, "regions": [],
                    "action": "no_watermark_detected" if cuda else "mock_copy",
                    "first_pass_strategy_version":"2026-11-01.source-details-core-v2",
                    "first_pass_mode":options.first_pass_mode,
                    "processing_path":"source_copy",
                    "source_detail_views":detail_counts.get(task["relative"],0),
            "source_edge_completion":edge_evidence.get(task["relative"],[]),
            "source_detail_context_requeries":context_evidence.get(task["relative"],[]),
                    "model_inferences":0,"model_input_budgets":[],
                    "detection_seconds": detected, "output_size": output_size,
                    "background_preserved": bool(protected_regions is not None),
                }
                continue
            state = {"task": task, "source": source, "result": source.copy(), "exif": exif, "icc": icc,
                         "alpha": alpha, "regions": regions, "protected_regions": protected_regions,
                         "whole_generation":options.scope=="full_image" and options.first_pass_mode=="whole",
                         "source_scene":"unknown",
                         "detection_seconds": detected,
                         "batches": [], "failed": None}
            states.append(state)
            core_regions=[]
            scene="unknown"
            if task["relative"] in detections and not isinstance(detections[task["relative"]][0],Exception):
                from .detection import source_scene
                scene=source_scene(detections[task["relative"]][0][1])
            if focused_auto:
                from .detection import parse_detection
                try:
                    core_regions=parse_detection(detections[task["relative"]][0][1],relative_padding=0)
                except (ValueError,TypeError,KeyError):
                    pass  # A missing raw core keeps the fully opaque padded mask.
            state["source_scene"]=scene
            state["protection_cores"]={}
            if state["whole_generation"] and state["protected_regions"] is not None and config.protection_core_feather:
                from .detection import source_core_boxes
                try:
                    state["protection_cores"]=source_core_boxes(detections[task["relative"]][0][1])
                except (ValueError,TypeError,KeyError):
                    pass  # Invalid/missing source-core provenance keeps the whole padded mask opaque.
            state["denoising_regions"]=state["protected_regions"]
            state["composition_regions"]=state["protected_regions"]
            state["composition_cores"]=state["protection_cores"]
            if state["protected_regions"] is not None and source_rim_audits.get(task["relative"]):
                from .source_panel_rims import effective_rim_scope
                scoped,scoped_cores,invalid=effective_rim_scope(source,state["protected_regions"],
                    [{"mask":list(mask),"core":list(core)} for mask,core in state["protection_cores"].items()],
                    source_rim_audits[task["relative"]],header_core=True)
                if invalid:
                    raise RuntimeError("Source-rim evidence does not match the original composition reference")
                state["composition_regions"]=scoped
                state["composition_cores"]={tuple(item["mask"]):tuple(item["core"]) for item in scoped_cores}
                if config.protect_denoising_context:
                    state["denoising_regions"]=scoped
            stable = int.from_bytes(hashlib.sha256(task["relative"].encode()).digest()[:4], "big")
            for index, region in enumerate(regions):
                box = pixel_box(region.box, source.size)
                crop = (0,0,*source.size) if state["whole_generation"] else context_box(box, source.size, options.context)
                reference = source if state["whole_generation"] else source.crop(crop)
                prefilled=False
                hint_audits=[]
                if config.translucent_body_hint and state["whole_generation"] and state["protected_regions"] is not None:
                    from .translucent_hint import translucent_body_hint
                    reference,hint_audits=translucent_body_hint(source,source_rim_audits.get(task["relative"],[]),
                        config.translucent_transmission)
                hint_applied=any(r.get("applied") for r in hint_audits)
                if config.prefill_overlays and state["whole_generation"] and state["protected_regions"] is not None:
                    from .images import prefill_whole_reference
                    if config.prefill_style=="gray":
                        from .gray_conditioning import gray_whole_reference
                        from .source_digital_proof import gray_source_cores
                        reference,prefilled=gray_whole_reference(source,state["protected_regions"],
                            gray_source_cores(detections[task["relative"]][0][1]),source_rim_audits.get(task["relative"],[]),
                            gray_certificates.get(task["relative"],[]))
                    else:
                        reference,prefilled=prefill_whole_reference(
                            source,state["protected_regions"],state["protection_cores"])
                elif config.prefill_overlays and focused_auto:
                    reference,prefilled=prefill_overlay_reference(source,box,crop,regions)
                extra_prompt = options.extra_prompt
                if state["whole_generation"] and options.whole_prompt_profile=="located":
                    from .prompts import whole_location_guidance
                    extra_prompt+=whole_location_guidance(state["protected_regions"] or [])
                if state["whole_generation"] and options.whole_prompt_profile=="scene":
                    from .source_brief import guidance
                    extra_prompt+=guidance(source_briefs.get(task["relative"]))
                if task.get("_retry_feedback"):
                    from .evidence import local_retry_guidance
                    local_guidance = local_retry_guidance(task["_retry_feedback"], crop, source.size)
                    if local_guidance:
                        extra_prompt += "\n本次实际模型输入中的返工位置：\n" + local_guidance
                focused = focused_auto
                small = focused and (region.box[2]-region.box[0])*(region.box[3]-region.box[1])<.035
                budget = min(options.resolution,512) if small else options.resolution
                if (state["whole_generation"] and options.preserve_background and scene=="document"
                        and config.protected_document_resolution and options.resolution>=1024
                        and source.width*source.height>options.resolution**2):
                    budget=max(budget,config.protected_document_resolution)
                cores=[pixel_box(r.box,source.size) for r in core_regions
                       if config.protection_core_feather and not small and r.box[0]>=region.box[0] and r.box[1]>=region.box[1]
                       and r.box[2]<=region.box[2] and r.box[3]<=region.box[3]]
                core_box=(min(b[0] for b in cores),min(b[1] for b in cores),
                          max(b[2] for b in cores),max(b[3] for b in cores)) if cores else None
                if prefilled:
                    # The entire declared mask was initialized for inpainting:
                    # do not blend original text/backing into its inner margin.
                    core_box=None
                edits.append({
                    "state": state, "box": box, "crop": crop, "index": index, "image": reference,
                    "_denoising_context_regions": ([r.model_dump() for r in state["denoising_regions"]]
                        if config.protect_denoising_context and state["whole_generation"]
                        and state["protected_regions"] is not None else None),
                    # Original visual semantics use the same ROI geometry as the VAE.
                    "_semantic_reference": (source if hint_applied else
                        (source if state["whole_generation"] else source.crop(crop))
                        if prefilled and config.prefill_original_semantics else None),
                    "prompt": (build_full_prompt(extra_prompt,options.whole_prompt_profile)
                               if prefilled and state["whole_generation"] and config.prefill_original_semantics else
                               build_gray_outpaint_prompt(extra_prompt,scene) if prefilled and state["whole_generation"] and config.prefill_style=="gray" else
                               build_prefilled_full_prompt(extra_prompt,scene) if prefilled and state["whole_generation"] else
                               build_small_overlay_prompt(box,crop,extra_prompt)+scene_instruction(scene)
                                if prefilled and config.prefill_original_semantics and small else
                                build_focused_prompt(region.kind,box,crop,extra_prompt,scene)
                                if prefilled and config.prefill_original_semantics and focused else
                                build_prompt(region.kind,box,crop,extra_prompt)
                                if prefilled and config.prefill_original_semantics else
                                build_prefilled_prompt(box,crop,extra_prompt,scene) if prefilled else
                               build_full_prompt(extra_prompt, options.whole_prompt_profile) if state["whole_generation"] else
                               build_small_overlay_prompt(box,crop,extra_prompt)+scene_instruction(scene)
                               if small else build_focused_prompt(region.kind,box,crop,extra_prompt,scene)
                               if focused else build_prompt(region.kind,box,crop,extra_prompt)),
                    "seed": (options.seed + stable + index) % (2**32),
                    "_context":options.context,"_extra_prompt":extra_prompt,
                    "scene_hint":scene,"core_box":core_box,"prefilled":prefilled,"translucent_hint":hint_audits,
                    "resolution":budget, "size": dimensions(reference,budget),
                })
        except Exception as error:  # noqa: BLE001 - isolate file and GPU failures
            if state is not None:
                state["failed"] = f"{type(error).__name__}: {error}"[:800]
            results[task["relative"]] = {
                "file": task["relative"], "ok": False,
                "error": f"{type(error).__name__}: {error}"[:800],
            }

    from .roi_grouping import coalesce_edits
    edits=coalesce_edits(edits,options.roi_grouping)
    for edit in edits:
        if edit.get("_roi_members"):
            edit["size"]=dimensions(edit["image"],edit["resolution"])

    from .roi_sampling import prepare_roi_sampling
    prepare_roi_sampling(edits,config)
    buckets = defaultdict(list)
    # Disable bucketing => adjacent compatible edits only; never mix different layouts.
    run, previous = 0, None
    for edit in edits:
        if edit["state"]["failed"]:
            continue
        schedule=edit.get("_roi_uniform_steps",0)
        layout=(edit["resolution"],edit["size"],schedule)
        if layout != previous:
            run += 1
        previous = layout
        key = ((edit["resolution"],edit["size"],schedule) if config.batch_bucketing
               else (run,edit["resolution"],edit["size"],schedule))
        buckets[key].append(edit)
    phase_offload=bool(phase_offload or (phase_manage and config.quality_phase_offload_above
                        and any(e["resolution"]>config.quality_phase_offload_above for e in edits)))
    if phase_offload and edits and not str(pipe._quality_encoder.device).startswith("cpu"):
        from .quality_loading import park_quality_encoder
        transfer_started=time.monotonic()
        pipe._quality_encoder=park_quality_encoder(pipe._quality_encoder)
        torch.cuda.empty_cache()
        quality_transfer_seconds+=time.monotonic()-transfer_started
    prefill_stats={"cached_images":0,"oom_splits":0,"fallback_images":0}
    if cuda and config.prompt_encoder=="lean" and config.encoder_batch_size>1 and edits:
        from .batching import preencode_window
        prefill_stats=preencode_window(pipe,[e for e in edits if not e["state"]["failed"]],config.encoder_batch_size)
    oom_retries = 0
    quality_oom_splits = 0
    quality_validation_splits = 0

    def execute(group):
        nonlocal oom_retries
        failure = None
        try:
            patches = independent_batch(pipe,group,group[0]["resolution"],config.kv_cache)
            if len(patches) != len(group):
                raise RuntimeError("Pipeline returned wrong image count")
            for edit, patch in zip(group, patches):
                # Store patches; apply in detection order even when buckets reorder edits.
                edit["patch"] = patch
                edit.pop("_encoded_prompt",None)
                edit["state"]["batches"].append(len(group))
            return
        except Exception as error:  # noqa: BLE001 - isolate file and GPU failures
            failure = f"{type(error).__name__}: {error}"[:800]
            is_oom = isinstance(error, torch.cuda.OutOfMemoryError)
            split = len(group) > 1 and (not is_oom or config.oom_split_batch)
            if is_oom:
                oom_retries += 1
                if split:
                    batch_limits[(group[0]["resolution"],group[0]["size"])] = len(group) // 2
        # Release exception traceback and transient tensors before retrying.
        torch.cuda.empty_cache()
        if split:
            middle = len(group) // 2
            execute(group[:middle])
            execute(group[middle:])
        else:
            for edit in group:
                edit["state"]["failed"] = failure

    for bucket in buckets.values():
        offset = 0
        while offset < len(bucket):
            maximum = (config.small_batch_size if config.small_batch_size and bucket[0]["resolution"] <= 512
                       else min(config.batch_size,config.large_batch_size) if config.large_batch_size and bucket[0]["resolution"]>1024
                       else config.batch_size)
            size = min(maximum, batch_limits.get((bucket[0]["resolution"],bucket[0]["size"]),maximum))
            execute(bucket[offset:offset + size])
            offset += size
    for edit in edits:
        edit.pop("_encoded_prompt",None)
        state = edit["state"]
        if not state["failed"]:
            if state["whole_generation"]:
                from PIL import Image
                state["result"] = edit["patch"].convert("RGB").resize(
                    state["source"].size, Image.Resampling.LANCZOS
                )
                if state["protected_regions"] is not None:
                    state["result"] = composite_protected(
                        state["source"], state["result"], state["composition_regions"], options.feather, options.repair_blending,
                        core_boxes=state["composition_cores"],
                    )
            else:
                if edit.get("_roi_members"):
                    from .roi_grouping import blend_group
                    state["result"]=blend_group(state["result"],edit["patch"],edit,options.feather,
                        options.repair_blending,reference_original=state["source"],exclude_regions=state["regions"])
                else:
                    state["result"] = blend_patch(
                        state["result"],edit["patch"],edit["box"],edit["crop"],options.feather,options.repair_blending,
                        reference_original=state["source"],exclude_regions=state["regions"],core_box=edit.get("core_box")
                    )
    for state in states:
        task = state["task"]
        result = {
            "file": task["relative"], "ok": not bool(state["failed"]),
            "regions": [r.model_dump() for r in state["regions"]],
            "detection_seconds": state["detection_seconds"], "batch_sizes": state["batches"],
            "background_preserved": state["protected_regions"] is not None or options.scope == "regions",
            "first_pass_strategy_version":"2026-11-01.source-details-core-v2",
            "first_pass_mode":options.first_pass_mode,
            "whole_location_guidance":bool(state["whole_generation"] and options.whole_prompt_profile=="located" and state["protected_regions"]),
            "processing_path":("whole_direct" if state["whole_generation"] and state["protected_regions"] is None
                               else "whole_composite" if state["whole_generation"] else "region_repair"),
            "source_detail_views":detail_counts.get(task["relative"],0),
            "source_edge_completion":edge_evidence.get(task["relative"],[]),
            "source_detail_context_requeries":context_evidence.get(task["relative"],[]),
            "model_inferences":sum(e["state"] is state for e in edits),
            "model_seeds":[e["seed"] for e in edits if e["state"] is state],
            "model_sampling":[e.get("_sampling_log",{}) for e in edits if e["state"] is state],
            "roi_grouping":options.roi_grouping,
            "roi_groups":[{"members":[list(m["box"]) for m in e.get("_roi_members",[e])],
                           "crop":list(e["crop"]),"budget":e["resolution"]}
                          for e in edits if e["state"] is state],
            "model_input_budgets":[e["resolution"] for e in edits if e["state"] is state],
            "encoder_generation_transfer":[e.get("_encoder_generation_log") for e in edits if e["state"] is state],
            "whole_prompt_profile":options.whole_prompt_profile,
            "protection_detail_scan":config.protection_detail_scan,
            "protection_core_feather":config.protection_core_feather,
            "source_rim_composition":{"regions":[r.model_dump() for r in state["composition_regions"] or []],
                                      "cores":[{"mask":list(mask),"core":list(core)}
                                               for mask,core in state["composition_cores"].items()]},
            "protected_core_boxes":[{"mask":list(r.box),"core":list(state["protection_cores"][r.box])}
                for r in state["protected_regions"] or [] if r.box in state["protection_cores"]],
            "source_geometry_requery":source_geometry_requeries.get(task["relative"]),
            "source_brief":source_briefs.get(task["relative"]),
            "edit_prompt_sha256":[hashlib.sha256(e["prompt"].encode()).hexdigest() for e in edits if e["state"] is state],
            "model_prefill_batch_sizes":[e.get("_prefill_batch_size",0) for e in edits if e["state"] is state],
            "window_prefill_stats":prefill_stats,
            "model_reference_sizes":[list(e["image"].size) for e in edits if e["state"] is state],
            "gray_conditioning":[e["image"].info["gray_conditioning"] for e in edits if e["state"] is state
                                 and "gray_conditioning" in e["image"].info],
            "prefill_style":config.prefill_style,
            "conditioning_prefill":[e["prefilled"] for e in edits if e["state"] is state],
            "translucent_hint":[r for e in edits if e["state"] is state for r in e.get("translucent_hint",[])],
            "denoising_context": [e.get("_denoising_context_log") for e in edits if e["state"] is state
                                  and e.get("_denoising_context_log")],
            "original_semantic_reference": any(e.get("_semantic_reference") is not None
                                               for e in edits if e["state"] is state),
            "semantic_reference_sizes": [list(e["_semantic_reference"].size)
                if e.get("_semantic_reference") is not None else None for e in edits if e["state"] is state],
            "whole_conditioning_source":("assumed_translucent_body_hint" if any(
                any(r.get("applied") for r in e.get("translucent_hint",[])) for e in edits if e["state"] is state)
                else "source_proven_gray_initialization" if config.prefill_style=="gray" and state["whole_generation"]
                and any(e["prefilled"] for e in edits if e["state"] is state)
                else "source_harmonic_initialization" if state["whole_generation"]
                and any(e["prefilled"] for e in edits if e["state"] is state) else "unmodified_original"),
            "opaque_core_boxes":[e["core_box"] for e in edits if e["state"] is state],
            "source_scene":state["source_scene"],
            "repair_blending": options.repair_blending,
            "harmonization": state["result"].info.get("harmonization"),
            "repair_regions": [r.model_dump() for r in (state["protected_regions"] if state["protected_regions"] is not None
                               else state["regions"] if options.scope == "regions" else [])],
        }
        try:
            if state["failed"]:
                raise RuntimeError(state["failed"])
            output = state["result"]
            from .images import protect_edge_completion_context
            restored=0
            if result["processing_path"]!="whole_direct":
                output,restored=protect_edge_completion_context(
                    state["source"],output,edge_evidence.get(task["relative"],[]))
            result["edge_context_pixels_preserved"]=restored
            if output.size != state["source"].size:
                raise RuntimeError("Output dimensions differ from original")
            rims=source_rim_audits.get(task["relative"],[])
            if rims:
                from .source_panel_rims import restore_proven_rims
                output,count=restore_proven_rims(state["source"],output,rims)
                if count!=sum(r.get("approved") is True for r in rims):
                    raise RuntimeError("Approved source-rim provenance no longer matches the original")
                result["protected_source_rim_count"]=count
            if state["alpha"] is not None:
                output.putalpha(state["alpha"])
            output_size = save_output(output, task["source"], task["destination"],
                                      state["exif"], state["icc"], options)
            result.update(action="restored", output_size=output_size)
        except Exception as error:  # noqa: BLE001 - isolate file and GPU failures
            result.update(ok=False, error=f"{type(error).__name__}: {error}"[:800])
        results[task["relative"]] = result
    for task in tasks:
        results[task["relative"]].setdefault("source_rim_protection",source_rim_audits.get(task["relative"],[]))
    quality_transformer_transfer={}
    with transformer_source_phase(pipe, cuda and config.transformer_phase_offload
                                  and options.quality_check,
                                  execution_device, quality_transformer_transfer,
                                  before_restore=before_transformer_restore):
        if options.quality_check:
            if phase_manage:
                transfer_started=time.monotonic()
                torch.cuda.empty_cache()
                from .quality_loading import restore_quality_encoder
                pipe._quality_encoder=restore_quality_encoder(pipe._quality_encoder,execution_device)
                quality_transfer_seconds+=time.monotonic()-transfer_started
            from .quality import verify_batch
            pending = []
            for task in tasks:
                result = results[task["relative"]]
                if not result["ok"]:
                    result["quality"] = {"status": "skipped", "reason": "processing_failed"}
                elif not cuda:
                    result["quality"] = {"status": "skipped", "reason": "mock_backend"}
                else:
                    pending.append(task)

            def review(group):
                nonlocal quality_oom_splits, quality_validation_splits
                failure, split, is_oom = "", False, False
                try:
                    pairs = [
                        (source_images[t["relative"]], existing_images[t["relative"]]
                         if t["relative"] in existing_images else load_photo(t["destination"])[0])
                        for t in group
                    ]
                    reviews = verify_batch(
                        pairs, pipe._quality_encoder, pipe._quality_processor, execution_device,
                        options.quality_resolution, options.quality_max_tokens,
                        cross_review_allowed=[options.quality_cross_review and not t.get("_cross_review_used") for t in group],
                        editor_encoder=getattr(pipe,"text_encoder",None), editor_processor=getattr(pipe,"processor",None),
                        weights_relation=getattr(pipe,"_quality_weights_relation","unknown"),
                        editor_context=[{"background_preserved":results[t["relative"]].get("background_preserved"),
                                         "repair_regions":results[t["relative"]].get("repair_regions",[]),
                                         "action":results[t["relative"]].get("action"),
                                         "source_scene":results[t["relative"]].get("source_scene","unknown"),
                                         "harmonization":results[t["relative"]].get("harmonization"),
                                         "protected_core_boxes":results[t["relative"]].get("protected_core_boxes",[]),
                                         "source_rim_protection":results[t["relative"]].get("source_rim_protection",[]),
                                         "denoising_context":results[t["relative"]].get("denoising_context",[])} for t in group],
                        reviewer=config.quality_backend, detail_patches=options.quality_detail_patches,
                        rectangle_guard=config.quality_rectangle_guard,
                        cross_context=config.quality_cross_full_context,
                        detection_batch_size=config.detection_batch_size, feather=options.feather,
                        source_regions=[
                            detections[t["relative"]][0][0] if t["relative"] in detections
                            and not isinstance(detections[t["relative"]][0], Exception)
                            else [Region.model_validate(r) for r in t["_repair_provenance"]["regions"]] if t.get("_repair_provenance")
                            else options.regions if options.scope == "regions" and options.preset == "custom"
                            else None for t in group
                        ],
                    )
                    for task, verdict, pair in zip(group, reviews, pairs):
                        size = ([options.output_width, options.output_height] if options.output_width
                                else list(source_images[task["relative"]].size))
                        verdict["display_output_size"] = list(pair[1].size)
                        verdict["dimensions_match"] = verdict["display_output_size"] == size
                        if not verdict["dimensions_match"] and verdict["status"] != "error":
                            verdict.update(status="review", reason=(verdict["reason"]+" 输出尺寸与所选设置不符。").strip())
                        results[task["relative"]]["quality"] = verdict
                    return
                except Exception as error:  # noqa: BLE001 - quality failures must never become passes
                    is_oom = isinstance(error, torch.cuda.OutOfMemoryError)
                    split = len(group) > 1
                    failure = f"{type(error).__name__}: {error}"[:400]
                # Leave the exception handler before retrying: traceback frames can own GPU tensors.
                if is_oom:
                    torch.cuda.empty_cache()
                if split:
                    quality_oom_splits += int(is_oom)
                    quality_validation_splits += int(not is_oom)
                    mid = len(group)//2
                    review(group[:mid])
                    review(group[mid:])
                else:
                    for task in group:
                        results[task["relative"]]["quality"] = {"status":"error","reason":failure}

            for offset in range(0, len(pending), options.quality_batch_size):
                review(pending[offset:offset + options.quality_batch_size])
    for task in tasks:
        result=results[task["relative"]]
        disputed=[v for v in physical_audits.get(task["relative"],[]) if v.get("requires_review")]
        if disputed and result.get("ok"):
            verdict=result.setdefault("quality",{"status":"review","reason":""})
            verdict["pre_source_carrier_guard"]={"status":verdict["status"],"reason":verdict.get("reason")}
            if verdict["status"]!="error":
                verdict["status"]="review"
            verdict["guard_codes"]=list(dict.fromkeys(verdict.get("guard_codes",[])+["source_physical_ambiguity"]))
            broad_dispute=any(v.get("version")=="source-broad-consensus-v1" for v in disputed)
            rim_dispute=any(v.get("version")=="source-rim-identity-v1" for v in disputed)
            gray_dispute=any(v.get("version")=="gray-source-digital-v1" for v in disputed)
            explanation=(" 原图检测范围过大，双模型未达成一致的位置或数字覆盖层证明；已保留未确认范围的原像素，需复核，不能盲目重绘。"
                         if broad_dispute else
                         " 原图信息卡候选的数字身份存在分类分歧；已保留该处原像素，需要复核，不能盲目擦除。"
                         if rim_dispute else
                         " 灰初始化候选缺少双模型完整数字身份源证明；已保留原图，需要复核，不能盲目重绘。"
                         if gray_dispute else
                         " 原图小块文字存在实物载体证据，但双模型未达成完整分类证明；保留该处原图，需复核，不能盲目擦除。")
            verdict["reason"]=(verdict.get("reason","")+explanation).strip()
            verdict["source_carrier_evidence"]=disputed
            from .evidence import refresh_evidence
            refresh_evidence(verdict)
            result["retry_blocked"]="uncertain_source_carrier"
            if result.get("action")=="no_watermark_detected":
                result["action"]="source_carrier_review"
    if cuda:
        torch.cuda.synchronize()
    metrics = {
        "image_adapter":getattr(pipe,"_image_adapter",{}),
        "quality_phase_offload":phase_offload,
        "quality_host_pinned_bytes":getattr(getattr(pipe,"_quality_encoder",None),"_quality_host_pinned_bytes",0),
        "source_transformer_transfer":dict(source_transformer_transfer),
        "quality_transformer_transfer":dict(quality_transformer_transfer),
        "quality_weight_transfer_seconds":round(quality_transfer_seconds/max(1,len(tasks)),4),
        "window_seconds": round(time.monotonic() - start, 3), "window_images": len(tasks),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated()/2**20) if cuda else 0,
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved()/2**20) if cuda else 0,
        "oom_retries": oom_retries, "quality_oom_splits": quality_oom_splits,
        "quality_validation_splits":quality_validation_splits,
    }
    return [results[task["relative"]] | metrics | {"source_physical_vetting":physical_audits.get(task["relative"],[]),
             "gray_source_certificates":gray_certificates.get(task["relative"],[])} for task in tasks]


def process_image(task, options, config, pipe):
    return process_batch([task], options, config, pipe)[0]


def worker_main(index, gpu, settings, model_path, requests, events, load_gate, quality_path=None,
                quality_weights_relation="unknown"):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config = DeploymentConfig.model_validate(settings)
    if config.expandable_segments:
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    try:
        with load_gate:
            if config.protect_source_rims and config.backend!="mock" and quality_weights_relation!="different_weights":
                raise RuntimeError("原图边沿保护需要与编辑器不同的独立检验权重")
            pipe = None if config.backend == "mock" else load_pipeline(config, model_path)
            if pipe is not None and quality_path:
                import torch

                from .quality_loading import load_quality_encoder, load_quality_processor
                torch.cuda.empty_cache()
                pipe._quality_encoder = load_quality_encoder(quality_path,config,pipe._execution_device)
                pipe._quality_processor = load_quality_processor(quality_path)
            elif pipe is not None and config.quality_verifier:
                pipe._quality_encoder, pipe._quality_processor = pipe.text_encoder, pipe.processor
            if pipe is not None:
                pipe._quality_weights_relation = quality_weights_relation
        events.put({"type": "ready", "worker": index,"image_adapter":getattr(pipe,"_image_adapter",{})})
    except Exception as error:  # noqa: BLE001 - isolate file and GPU failures
        traceback.print_exc()
        events.put({"type": "fatal", "worker": index, "error": str(error)[:800]})
        return
    batch_limits = {}
    while True:
        request = requests.get()
        if request is None:
            return
        try:
            from .retry import process_attempts
            results = process_attempts(
                request["tasks"], ProcessingConfig.model_validate(request["options"]), config, pipe,
                batch_limits, request["job"], process_batch,
            )
        except Exception as error:  # noqa: BLE001 - isolate file and GPU failures
            traceback.print_exc()
            results = [
                {"file": task["relative"], "ok": False,
                 "error": f"{type(error).__name__}: {error}"[:800]}
                for task in request["tasks"]
            ]
            if config.backend != "mock":
                import torch
                torch.cuda.empty_cache()
        events.put({"type": "results", "worker": index, "job": request["job"], "results": results})
