"""Read-only paired verifier benchmark with frozen SOURCE masks and unchanged outputs."""
from __future__ import annotations

import argparse
import collections
import errno
import hashlib
import json
import multiprocessing as mp
import os
import queue
import re
import subprocess
import time
from copy import deepcopy
from pathlib import Path

PROFILES={
    "4b-bf16":("Qwen--Qwen3-VL-4B-Instruct","none"),
    "8b-nf4":("Qwen--Qwen3-VL-8B-Instruct","nf4"),
    "32b-nf4":("Qwen--Qwen3-VL-32B-Instruct","nf4"),
    "35-4b-bf16":("Qwen--Qwen3.5-4B","none"),
    "35-9b-bf16":("Qwen--Qwen3.5-9B","none"),
}


def frozen_job_cases(state_root, job, files=None):
    from watermark_studio.config import Region
    folder = Path(state_root) / job
    summary = json.loads((folder/"summary.json").read_text())
    if summary["state"] != "completed":
        raise ValueError("Completed source job required")
    rows = [json.loads(line) for line in (folder/"results.jsonl").read_text().splitlines()]
    if len(rows) != summary["total"] or len({r["file"] for r in rows}) != len(rows):
        raise ValueError("Incomplete or repeated source results")
    source_root, output_root = Path(summary["input_dir"]).resolve(), Path(summary["output_dir"]).resolve()
    selected=None if files is None else set(files)
    if selected is not None and (not selected or len(selected)!=len(files) or
                                 not selected.issubset({r["file"] for r in rows})):
        raise ValueError("Explicit source subset must be nonempty, unique and available")
    cases = []
    for row in sorted(rows, key=lambda r:r["file"]):
        name = Path(row["file"])
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("Every paired image must stay inside its frozen source root")
        if selected is not None and row["file"] not in selected:
            continue
        if not row.get("ok"):
            raise ValueError("Every selected pair must have a valid successful generation")
        source, output = (source_root/name).resolve(), (output_root/name).resolve()
        if not source.is_relative_to(source_root) or not output.is_relative_to(output_root):
            raise ValueError("Pair path escapes source/output roots")
        if not source.is_file() or not output.is_file():
            raise ValueError("A source/output pair is missing")
        q = row.get("quality", {})
        if row.get("processing_path")=="whole_direct":
            # No compositing mask does not mean the original has no watermark.
            # Only the independent source scan is valid provenance here.
            raw=q.get("source_regions")
        else:
            raw = row.get("repair_regions", q.get("source_regions"))
        if not isinstance(raw, list) or len(raw)>8:
            raise ValueError("Missing bounded ORIGINAL source-mask snapshot")
        regions = [Region.model_validate(r).model_dump() for r in raw]
        scene = row.get("source_scene", q.get("scene_context", "unknown"))
        scene = "unknown" if scene is None else scene
        if scene not in {"outdoor","indoor","document","unknown"}:
            raise ValueError("Untrusted source scene category")
        cases.append({"file":str(name),"source":str(source),"output":str(output),
                      "regions":regions,"scene":scene,"cohort":"source_job",
                      "generation_attempts":row.get("quality_attempts",1),
                      "baseline_status":q.get("status"),"frozen_quality":q,"source_cores":row.get("protected_core_boxes",[]),
                      "source_rim_protection":row.get("source_rim_protection",[]),
                      "denoising_context":row.get("denoising_context",[]),"preserved":row.get("background_preserved",False),
                      "frozen_vetoes":[g for g in q.get("guard_codes",[]) if g!="identical"]})
    return cases


def known_control_cases(root):
    from PIL import Image

    from watermark_studio.config import Region
    records = json.loads((root/"fixture.json").read_text())
    cases=[]
    positions=((.02,.66),(.60,.03),(.30,.38),(.03,.03),(.60,.67))
    for record in records:
        name=record["file"]
        source, output = root/"input"/name, root/"output"/name
        with Image.open(source) as image:
            w,h=image.size
        if record["kind"]=="identity":
            regions=[]
        else:
            index=int(name.split("-")[0])
            x,y=positions[index%5]
            left,top=int(x*w)/w,int(y*h)/h
            right,bottom=int((x+.37)*w)/w,int(min(1,y+.25)*h)/h
            # Includes rectangle endpoint and the same conservative safety margin.
            px,py=.008+(right-left)*.06,.008+(bottom-top)*.06
            regions=[Region(kind="panel",box=(max(0,left-px),max(0,top-py),
                min(1,right+px+1/w),min(1,bottom+py+1/h))).model_dump()]
        cases.append({"file":name,"source":str(source),"output":str(output),"regions":regions,
                      "scene":"document" if record["background"]=="receipt" else "outdoor",
                      "cohort":"controls","expected":record["expected"],"kind":record["kind"],"preserved":True,
                      "source_cores":[] if not regions else [{"mask":regions[0]["box"],
                          "core":[left,top,min(1,right+1/w),min(1,bottom+1/h)]}]})
    return cases


def grade_worker(gpu, cases, model, mode, editor_model, editor_processor, relation, cross, batch, rectangle_guard, full_context, paired_context, resolution, text_free_consensus, frozen_disputes, text_free_scope, inspection_profile, proof_profile, proof_identity, uniform_guard, ocr_guard, ready, start):
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
    try:
        import torch
        from transformers import Qwen3VLProcessor
        from transformers.utils import logging
        logging.disable_progress_bar()

        from watermark_studio.config import DeploymentConfig, Region
        from watermark_studio.images import load_photo
        from watermark_studio.quality import verify_batch
        from watermark_studio.quality_loading import load_quality_encoder, load_quality_processor
        torch.set_num_threads(1)
        cfg=DeploymentConfig(quality_verifier=True,quality_backend="independent",quality_quantization=mode)
        encoder=load_quality_encoder(model,cfg,"cuda:0")
        processor=load_quality_processor(model)
        editor=None
        ep=None
        allow=cross and relation=="different_weights"
        if allow:
            plain=cfg.model_copy(update={"quality_quantization":"none"})
            editor=load_quality_encoder(editor_model,plain,"cuda:0")
            ep=Qwen3VLProcessor.from_pretrained(editor_processor,local_files_only=True)
        proof_validator=proof_processor=None
        if proof_profile=="35-4b-bf16":
            proof_path=Path(model).parents[2]/"Qwen--Qwen3.5-4B/snapshots/master"
            proof_validator=load_quality_encoder(proof_path,cfg,"cuda:0")
            proof_processor=load_quality_processor(proof_path)
        text_detector=None
        if ocr_guard:
            from watermark_studio.text_presence import TargetTextPresence
            text_detector=TargetTextPresence()
        ready.put({"type":"ready","gpu":gpu})
        start.wait()
        def grade(group,contexts=None):
            variants=contexts if contexts is not None else ((False,True) if paired_context else (full_context,))
            for context in variants:
                failure=None
                try:
                    pairs=[(load_photo(c["source"])[0],load_photo(c["output"])[0]) for c in group]
                    began=time.monotonic()
                    if frozen_disputes:
                        values=[deepcopy(c["frozen_quality"]) for c in group]
                    else:
                        values=verify_batch(pairs,encoder,processor,"cuda:0",resolution=resolution,
                            detail_patches=4,detection_batch_size=2,
                            source_regions=[[Region.model_validate(r) for r in c["regions"]] for c in group],
                            editor_context=[{"source_scene":c["scene"],"protected_core_boxes":c.get("source_cores",[]),"background_preserved":c.get("preserved",False),
                                "source_rim_protection":c.get("source_rim_protection",[]),
                                "denoising_context":c.get("denoising_context",[])} for c in group],
                            cross_review_allowed=[allow]*len(group),editor_encoder=editor,editor_processor=ep,
                            reviewer="paired_"+mode,weights_relation=relation,rectangle_guard=rectangle_guard,
                            cross_context=context,inspection_profile=inspection_profile)
                    if len(values)!=len(group):
                        raise RuntimeError("Verifier response count differs from frozen pairs")
                    torch.cuda.synchronize()
                    initial_statuses=[("review" if c.get("frozen_vetoes") and v["status"]!="error" else v["status"]) for c,v in zip(group,values)]
                    for case,value in zip(group,values):
                        if text_free_consensus and not case.get("frozen_vetoes"):
                            from watermark_studio.text_free_review import reconsider_text_free
                            pair_index=group.index(case)
                            reconsider_text_free(*pairs[pair_index],value,editor,ep,encoder,processor,
                                "cuda:0",{"background_preserved":case.get("preserved",False),
                                "source_scene":case["scene"],"protected_core_boxes":case.get("source_cores",[])},
                                relation,resolution,proof_scope=text_free_scope,
                                proof_validator=proof_validator,proof_processor=proof_processor,
                                proof_identity=proof_identity,uniform_guard=uniform_guard,text_detector=text_detector)
                        if case.get("frozen_vetoes"):
                            if value["status"]!="error":
                                value["status"]="review"
                            value["guard_codes"]=sorted(set(value.get("guard_codes",[])+case["frozen_vetoes"]))
                            value["reason"]=(value.get("reason","")+" 保留冻结候选的不可豁免约束。").strip()
                            if value.get("cross_review",{}).get("status")=="corrected":
                                value["cross_review"]["status"]="blocked_frozen_veto"
                    torch.cuda.synchronize()
                    elapsed=time.monotonic()-began
                    ready.put({"type":"rows","rows":[c|{"quality":v,"initial_verdict_status":initial_status,"context_variant":"full" if context else "local",
                        "variant_group_seconds":round(elapsed/len(group),4)} for c,v,initial_status in zip(group,values,initial_statuses)]})
                    continue
                except Exception as error:  # noqa: BLE001 - retry only this variant and release traceback first
                    failure=f"{type(error).__name__}: {error}"[:300]
                torch.cuda.empty_cache()
                if len(group)>1:
                    middle=len(group)//2
                    grade(group[:middle],(context,));grade(group[middle:],(context,))
                else:
                    ready.put({"type":"rows","rows":[group[0]|{
                        "context_variant":"full" if context else "local",
                        "quality":{"status":"error","reason":failure}}]})
        for at in range(0,len(cases),batch):
            grade(cases[at:at+batch])
    except Exception as error:  # noqa: BLE001 - report child failure before cleanup
        ready.put({"type":"fatal","gpu":gpu,"error":f"{type(error).__name__}: {error}"[:400]})


def file_hashes(cases):
    result={}
    for case in cases:
        for key in ("source","output"):
            with open(case[key],"rb") as stream:
                result[case[key]]=hashlib.file_digest(stream,"sha256").hexdigest()
    return result



def require_disjoint_server_gpus(health, requested, allow):
    """Explicit test opt-in, proved against live reserved physical GPU indices."""
    if not allow:
        raise RuntimeError("Stop the inference server before the standalone verifier benchmark")
    reserved = health.get("gpus")
    if (not isinstance(reserved, list) or not reserved
            or any(type(gpu) is not int or gpu < 0 for gpu in reserved)
            or len(set(reserved)) != len(reserved) or not health.get("code_revision")):
        raise RuntimeError("Cannot prove live server GPU reservation")
    if set(reserved) & set(requested):
        raise RuntimeError("Verifier benchmark overlaps live inference server GPUs")
    return {"active": True, "gpus": sorted(reserved), "code_revision": health["code_revision"]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-job")
    parser.add_argument("--files",nargs="+",help="Explicit successful source subset; omissions are reported")
    parser.add_argument("--controls-only",action="store_true",help="Grade only known control pairs; no candidate source job")
    parser.add_argument("--tag",required=True)
    parser.add_argument("--profile",choices=list(PROFILES),required=True)
    parser.add_argument("--gpus",default="0,1,2,3,4,5,6,7")
    parser.add_argument("--allow-disjoint-server",action="store_true",
                        help="Allow a live server only when its reported physical GPUs are all disjoint; default refuses")
    parser.add_argument("--quality-batch-size",type=int,choices=[1,2,3,4],default=2)
    parser.add_argument("--quality-resolution",type=int,choices=[512,768,1024,1280],default=768)
    parser.add_argument("--text-free-proof-profile",choices=["primary","35-4b-bf16"],default="primary")
    parser.add_argument("--text-free-ocr-guard",action="store_true")
    parser.add_argument("--text-free-uniform-guard",action="store_true")
    parser.add_argument("--inspection-profile",choices=["legacy","defects"],default="legacy")
    parser.add_argument("--first-snapshot",action="store_true",help="Validate an immutable first snapshot, then freshly grade the same pixels")
    parser.add_argument("--text-free-scope",choices=["global","targets"],default="global")
    parser.add_argument("--frozen-disputes",action="store_true",help="Only adjudicate SHA-validated immutable first verdicts; do not rerun first VQA")
    parser.add_argument("--text-free-consensus",action="store_true",help="Experimental evidence-based adjudication; cannot override frozen vetoes")
    parser.add_argument("--cross-review",action="store_true")
    parser.add_argument("--rectangle-guard",action="store_true")
    parser.add_argument("--cross-full-context",action="store_true")
    parser.add_argument("--paired-context",action="store_true")
    parser.add_argument("--select-review",action="store_true",help="Frozen source final unaccepted files only")
    parser.add_argument("--controls",type=Path)
    parser.add_argument("--state-dir",type=Path,default=Path.home()/".local/state/watermark-studio")
    parser.add_argument("--models-root",type=Path,default=Path.home()/".cache/watermark-models/models")
    args=parser.parse_args()
    if args.text_free_proof_profile!="primary" and (not args.text_free_consensus or args.profile!="4b-bf16"):
        parser.error("Additional proof weights require canonical4b text-free consensus")
    if args.text_free_proof_profile!="primary" and not args.text_free_ocr_guard:
        parser.error("The native audit disproved third-model text absence alone; --text-free-ocr-guard is required")
    if args.text_free_ocr_guard and not args.text_free_consensus:
        parser.error("OCR absence veto requires text-free consensus")
    if args.text_free_uniform_guard and not args.text_free_consensus:
        parser.error("The optional uniform-core veto requires text-free consensus")
    if args.frozen_disputes and args.inspection_profile!="legacy":
        parser.error("Cached first verdicts cannot be relabelled with a fresh inspection protocol")
    if args.first_snapshot and (args.frozen_disputes or args.controls_only):
        parser.error("--first-snapshot requires fresh independent inspection; no cached disputes/controls-only")
    if args.text_free_consensus and not args.cross_review:
        parser.error("--text-free-consensus requires --cross-review")
    if args.frozen_disputes and (not args.text_free_consensus or args.controls or args.controls_only or args.paired_context or args.profile!="4b-bf16"):
        parser.error("--frozen-disputes requires text-free consensus, a first snapshot and 4b-bf16; no controls/paired variants")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+",args.tag):
        parser.error("Simple tag required")
    if args.controls_only:
        if args.source_job or args.files or not args.controls:
            parser.error("--controls-only requires --controls and excludes --source-job")
    elif not args.source_job or not re.fullmatch(r"[a-zA-Z0-9_-]+" if args.frozen_disputes or args.first_snapshot else r"[a-f0-9]{32}",args.source_job):
        parser.error("Exact completed source job ID required")
    gpus=[int(v) for v in args.gpus.split(",")]
    if len(set(gpus))!=len(gpus) or not gpus or min(gpus)<0:
        parser.error("Distinct nonnegative GPU indices required")
    import urllib.error
    import urllib.request
    server_isolation={"active":False}
    try:
        with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as response:
            server_health=json.load(response)
    except urllib.error.URLError as error:
        if getattr(error.reason,"errno",None)!=errno.ECONNREFUSED:
            raise RuntimeError("Cannot establish that the inference server is stopped") from error
    else:
        server_isolation=require_disjoint_server_gpus(server_health,gpus,args.allow_disjoint_server)
    code_revision=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    target=args.state_dir/"verifier-benchmarks"/args.tag/args.profile
    target.mkdir(parents=True,exist_ok=False)
    cases=known_control_cases(args.controls) if args.controls else []
    if not args.controls_only:
        source_cases=frozen_job_cases(args.state_dir,args.source_job,args.files)
        cases += [c for c in source_cases if not args.select_review or c["baseline_status"]!="passed"]
    expected_total=len(cases)*(2 if args.paired_context else 1)
    before=file_hashes(cases)
    repository,mode=PROFILES[args.profile]
    model=args.models_root/repository/"snapshots/master"
    editor_root=args.models_root/"Qwen--Qwen-Image-2.1-Turbo/snapshots/master"
    from watermark_studio.model_identity import compare_model_weights
    identity=compare_model_weights(editor_root/"text_encoder",model)
    proof_identity=None
    if args.text_free_proof_profile=="35-4b-bf16":
        proof_path=args.models_root/"Qwen--Qwen3.5-4B/snapshots/master"
        proof_identity=compare_model_weights(editor_root/"text_encoder",proof_path)
        base_proof=compare_model_weights(model,proof_path)
        if proof_identity["relation"]!="different_weights" or base_proof["relation"]!="different_weights":
            raise ValueError("Additional proof must differ canonically from editor and primary verifier")
    snapshot_validation=None
    if args.frozen_disputes:
        from watermark_studio.frozen_candidates import validate_first_snapshot
        validate_first_snapshot(args.state_dir/args.source_job,identity)
    elif args.first_snapshot:
        from watermark_studio.frozen_candidates import validate_fresh_inspection
        snapshot_validation=validate_fresh_inspection(args.state_dir/args.source_job,identity)
    ctx=mp.get_context("spawn")
    events,start=ctx.Queue(),ctx.Event()
    workers=[]
    began=time.monotonic()
    try:
        for rank,gpu in enumerate(gpus):
            process=ctx.Process(target=grade_worker,args=(gpu,cases[rank::len(gpus)],str(model),mode,
                str(editor_root/"text_encoder"),str(editor_root/"processor"),identity["relation"],
                args.cross_review,args.quality_batch_size,args.rectangle_guard,
                args.cross_full_context,args.paired_context,args.quality_resolution,args.text_free_consensus,args.frozen_disputes,args.text_free_scope,args.inspection_profile,args.text_free_proof_profile,proof_identity,args.text_free_uniform_guard,args.text_free_ocr_guard,events,start))
            process.start();workers.append(process)
        pending=len(workers)
        while pending:
            event=events.get(timeout=300)
            if event["type"]!="ready":
                raise RuntimeError(event)
            pending-=1
        load_seconds=round(time.monotonic()-began,3)
        began=last=time.monotonic();start.set()
        results=[]
        with (target/"results.jsonl").open("w") as stream:
            while len(results)<expected_total:
                try:
                    event=events.get(timeout=2)
                except queue.Empty:
                    if (any(p.exitcode not in (None,0) for p in workers)
                            or all(p.exitcode is not None for p in workers)):
                        raise RuntimeError("Verifier worker exited unexpectedly")
                    continue
                if event["type"]=="fatal":
                    raise RuntimeError(event)
                for row in event["rows"]:
                    results.append(row);stream.write(json.dumps(row,ensure_ascii=False)+"\n");stream.flush()
                if time.monotonic()-last>=30:
                    print(json.dumps({"profile":args.profile,"completed":len(results),"total":expected_total}),flush=True)
                    last=time.monotonic()
        seconds=round(time.monotonic()-began,3)
        if len(results)!=expected_total or len({(r["cohort"],r["file"],r.get("context_variant")) for r in results})!=expected_total:
            raise RuntimeError("Repeated verifier response")
        for process in workers:
            process.join(timeout=15)
        if before!=file_hashes(cases):
            raise RuntimeError("Source or candidate changed during the read-only benchmark")
        cohorts={}
        for cohort in sorted({c["cohort"] for c in cases}):
            variants=("local","full") if args.paired_context else ("full" if args.cross_full_context else "local",)
            for variant in variants:
                selected=[r for r in results if r["cohort"]==cohort and r["context_variant"]==variant]
                label=f"{cohort}/{variant}" if args.paired_context else cohort
                cohorts[label]={"images":len(selected),
                    "counts":dict(collections.Counter(r["quality"]["status"] for r in selected)),
                    "initial_counts":dict(collections.Counter(r.get("initial_verdict_status",r["quality"]["status"]) for r in selected)),
                    "text_free_corrected":sum(r["quality"].get("text_free_review",{}).get("status")=="corrected" for r in selected),
                    "model_seconds_sum":round(sum(r.get("variant_group_seconds",0) for r in selected),4),
                    "cross_corrected":sum(r["quality"].get("cross_review",{}).get("status")=="corrected" for r in selected)}
                if cohort=="controls":
                    cohorts[label].update(
                        clean_refused=sum(r["expected"]=="passed" and r["quality"]["status"]!="passed" for r in selected),
                        new_false_accept_files=[r["file"] for r in selected if r["expected"]=="review" and r.get("initial_verdict_status")!="passed" and r["quality"]["status"]=="passed"],
                        new_clean_refusal_files=[r["file"] for r in selected if r["expected"]=="passed" and r.get("initial_verdict_status")=="passed" and r["quality"]["status"]!="passed"],
                        defect_passed=sum(r["expected"]=="review" and r["quality"]["status"]=="passed" for r in selected),
                        false_accept_files=[r["file"] for r in selected if r["expected"]=="review" and r["quality"]["status"]=="passed"],
                        false_refuse_files=[r["file"] for r in selected if r["expected"]=="passed" and r["quality"]["status"]!="passed"])
        if server_isolation["active"]:
            with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as response:
                latest=require_disjoint_server_gpus(json.load(response),gpus,True)
            if latest!=server_isolation:
                raise RuntimeError("Live server GPU reservation changed during verifier evaluation")
        result={"server_gpu_isolation":server_isolation,"profile":args.profile,"source_job":args.source_job,"source_files":args.files,
                 "code_revision":code_revision,"images":len(cases),"verifications":expected_total,
                "cross_full_context":args.cross_full_context,"paired_context":args.paired_context,
                "inspection_profile":args.inspection_profile,"first_snapshot":args.first_snapshot,
                "fresh_snapshot_validation":snapshot_validation,
                "text_free_proof_profile":args.text_free_proof_profile,"proof_identity":proof_identity,
                "text_free_uniform_guard":args.text_free_uniform_guard,"text_free_ocr_guard":args.text_free_ocr_guard,
                "quality_resolution":args.quality_resolution,"text_free_consensus":args.text_free_consensus,"frozen_disputes":args.frozen_disputes,"text_free_scope":args.text_free_scope,
                "select_review":args.select_review,"rectangle_guard":args.rectangle_guard,
                "source_mask_origin":"frozen_original_source_snapshot_and_synthetic_annotation",
                "weight_identity":identity,"cross_review_requested":args.cross_review,
                "cross_review_effective":args.cross_review and identity["relation"]=="different_weights",
                "seconds":seconds,"load_seconds":load_seconds,"cohorts":cohorts,"outputs_unchanged":True,"sources_unchanged":True}
        (target/"summary.json").write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
    finally:
        start.set()
        for process in workers:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)


if __name__=="__main__":
    main()
