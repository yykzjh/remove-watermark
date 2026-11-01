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
from pathlib import Path

PROFILES={
    "4b-bf16":("Qwen--Qwen3-VL-4B-Instruct","none"),
    "8b-nf4":("Qwen--Qwen3-VL-8B-Instruct","nf4"),
    "35-4b-bf16":("Qwen--Qwen3.5-4B","none"),
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
        raw = row.get("repair_regions", q.get("source_regions"))
        if not isinstance(raw, list) or len(raw)>8:
            raise ValueError("Missing bounded ORIGINAL source-mask snapshot")
        regions = [Region.model_validate(r).model_dump() for r in raw]
        scene = row.get("source_scene", q.get("scene_context", "unknown"))
        if scene not in {"outdoor","indoor","document","unknown"}:
            raise ValueError("Untrusted source scene category")
        cases.append({"file":str(name),"source":str(source),"output":str(output),
                      "regions":regions,"scene":scene,"cohort":"source_job",
                      "generation_attempts":row.get("quality_attempts",1)})
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
                      "cohort":"controls","expected":record["expected"],"kind":record["kind"]})
    return cases


def grade_worker(gpu, cases, model, mode, editor_model, editor_processor, relation, cross, batch, boundary_views, native_verdicts, artifact_question, physical_guidance, ready, start):
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
        ready.put({"type":"ready","gpu":gpu})
        start.wait()
        def grade(group):
            try:
                pairs=[(load_photo(c["source"])[0],load_photo(c["output"])[0]) for c in group]
                values=verify_batch(pairs,encoder,processor,"cuda:0",resolution=768,
                    detail_patches=4,detection_batch_size=2,
                    source_regions=[[Region.model_validate(r) for r in c["regions"]] for c in group],
                    editor_context=[{"source_scene":c["scene"]} for c in group],
                    cross_review_allowed=[allow]*len(group),editor_encoder=editor,editor_processor=ep,
                    reviewer="paired_"+mode,weights_relation=relation,boundary_views=boundary_views,native_verdicts=native_verdicts,artifact_question=artifact_question,physical_guidance=physical_guidance)
                if len(values)!=len(group):
                    raise RuntimeError("Verifier response count differs from frozen pairs")
                ready.put({"type":"rows","rows":[c|{"quality":v} for c,v in zip(group,values)]})
            except Exception as error:  # noqa: BLE001 - isolate model and GPU failures
                torch.cuda.empty_cache()
                if len(group)>1:
                    middle=len(group)//2;grade(group[:middle]);grade(group[middle:])
                else:
                    ready.put({"type":"rows","rows":[group[0]|{"quality":{
                        "status":"error","reason":f"{type(error).__name__}: {error}"[:300]}}]})
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


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-job")
    parser.add_argument("--files",nargs="+",help="Explicit successful source subset; omissions are reported")
    parser.add_argument("--controls-only",action="store_true",help="Grade only known control pairs; no candidate source job")
    parser.add_argument("--tag",required=True)
    parser.add_argument("--profile",choices=list(PROFILES),required=True)
    parser.add_argument("--gpus",default="0,1,2,3,4,5,6,7")
    parser.add_argument("--quality-batch-size",type=int,choices=[1,2,3,4],default=2)
    parser.add_argument("--cross-review",action="store_true")
    parser.add_argument("--boundary-views",action="store_true")
    parser.add_argument("--native-verdicts",action="store_true")
    parser.add_argument("--artifact-question",action="store_true")
    parser.add_argument("--physical-guidance",action="store_true")
    parser.add_argument("--controls",type=Path)
    parser.add_argument("--state-dir",type=Path,default=Path.home()/".local/state/watermark-studio")
    parser.add_argument("--models-root",type=Path,default=Path.home()/".cache/watermark-models/models")
    args=parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+",args.tag):
        parser.error("Simple tag required")
    if args.controls_only:
        if args.source_job or args.files or not args.controls:
            parser.error("--controls-only requires --controls and excludes --source-job")
    elif not args.source_job or not re.fullmatch(r"[a-f0-9]{32}",args.source_job):
        parser.error("Exact completed source job ID required")
    gpus=[int(v) for v in args.gpus.split(",")]
    if len(set(gpus))!=len(gpus) or not gpus or min(gpus)<0:
        parser.error("Distinct nonnegative GPU indices required")
    import urllib.error
    import urllib.request
    try:
        urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2).close()
    except urllib.error.URLError as error:
        if getattr(error.reason,"errno",None)!=errno.ECONNREFUSED:
            raise RuntimeError("Cannot establish that the inference server is stopped") from error
    else:
        raise RuntimeError("Stop the inference server before the standalone verifier benchmark")
    code_revision=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    target=args.state_dir/"verifier-benchmarks"/args.tag/args.profile
    target.mkdir(parents=True,exist_ok=False)
    cases=known_control_cases(args.controls) if args.controls else []
    if not args.controls_only:
        cases+=frozen_job_cases(args.state_dir,args.source_job,args.files)
    before=file_hashes(cases)
    repository,mode=PROFILES[args.profile]
    model=args.models_root/repository/"snapshots/master"
    editor_root=args.models_root/"Qwen--Qwen-Image-2.1-Turbo/snapshots/master"
    from watermark_studio.model_identity import compare_model_weights
    identity=compare_model_weights(editor_root/"text_encoder",model)
    ctx=mp.get_context("spawn")
    events,start=ctx.Queue(),ctx.Event()
    workers=[]
    began=time.monotonic()
    try:
        for rank,gpu in enumerate(gpus):
            process=ctx.Process(target=grade_worker,args=(gpu,cases[rank::len(gpus)],str(model),mode,
                str(editor_root/"text_encoder"),str(editor_root/"processor"),identity["relation"],
                args.cross_review,args.quality_batch_size,args.boundary_views,args.native_verdicts,args.artifact_question,args.physical_guidance,events,start))
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
            while len(results)<len(cases):
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
                    print(json.dumps({"profile":args.profile,"completed":len(results),"total":len(cases)}),flush=True)
                    last=time.monotonic()
        seconds=round(time.monotonic()-began,3)
        if len({(r["cohort"],r["file"]) for r in results})!=len(cases):
            raise RuntimeError("Repeated verifier response")
        for process in workers:
            process.join(timeout=15)
        if before!=file_hashes(cases):
            raise RuntimeError("Source or candidate changed during the read-only benchmark")
        cohorts={}
        for cohort in sorted({c["cohort"] for c in cases}):
            selected=[r for r in results if r["cohort"]==cohort]
            cohorts[cohort]={"images":len(selected),"counts":dict(collections.Counter(r["quality"]["status"] for r in selected)),
                "cross_corrected":sum(r["quality"].get("cross_review",{}).get("status")=="corrected" for r in selected)}
            if cohort=="controls":
                cohorts[cohort].update(
                    clean_refused=sum(r["expected"]=="passed" and r["quality"]["status"]!="passed" for r in selected),
                    defect_passed=sum(r["expected"]=="review" and r["quality"]["status"]=="passed" for r in selected))
        result={"profile":args.profile,"source_job":args.source_job,"source_files":args.files,
                 "code_revision":code_revision,"images":len(cases),"boundary_views":args.boundary_views,"native_verdicts":args.native_verdicts,"artifact_question":args.artifact_question,"physical_guidance":args.physical_guidance,
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
