"""Reproduce source-only detection recovery on frozen failures; requires idle GPUs."""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--job",required=True)
    parser.add_argument("--files",nargs="+",required=True)
    parser.add_argument("--gpu",type=int,required=True)
    parser.add_argument("--resolution",type=int,default=1280)
    parser.add_argument("--detection-batch-size",type=int,default=2)
    args=parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{32}",args.job) or args.gpu<0:
        parser.error("Exact job ID and explicit nonnegative GPU required")
    try:
        urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2).close()
    except urllib.error.URLError as error:
        if getattr(error.reason,"errno",None)!=errno.ECONNREFUSED:
            raise RuntimeError("Cannot confirm the inference service is stopped") from error
    else:
        raise RuntimeError("Stop the inference service before the GPU source probe")
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    import torch
    from transformers import Qwen3VLProcessor
    from transformers.utils import logging

    from watermark_studio.config import DeploymentConfig
    from watermark_studio.detection import locate_watermarks_batch, refine_source_details
    from watermark_studio.images import load_photo
    from watermark_studio.quality_loading import load_quality_encoder

    logging.disable_progress_bar()
    torch.set_num_threads(1)
    folder=Path.home()/".local/state/watermark-studio"/args.job
    summary=json.loads((folder/"summary.json").read_text())
    if summary["state"]!="completed":
        raise ValueError("Completed frozen source job required")
    known={json.loads(row)["file"] for row in (folder/"results.jsonl").read_text().splitlines()}
    if len(set(args.files))!=len(args.files) or not set(args.files).issubset(known):
        raise ValueError("Each explicit source must be unique and belong to the frozen job")
    root=Path(summary["input_dir"]).resolve()
    paths=[(root/name).resolve() for name in args.files]
    if any(not p.is_relative_to(root) or not p.is_file() for p in paths):
        raise ValueError("Sources must stay inside the frozen input root")
    fingerprints=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    photos=[load_photo(str(p))[0] for p in paths]
    model=Path.home()/".cache/watermark-models/models/Qwen--Qwen-Image-2.1-Turbo/snapshots/master"
    cfg=DeploymentConfig(quality_verifier=True,quality_backend="independent")
    started=time.monotonic()
    encoder=load_quality_encoder(model/"text_encoder",cfg,"cuda:0")
    processor=Qwen3VLProcessor.from_pretrained(model/"processor",local_files_only=True)
    load_seconds=round(time.monotonic()-started,3)
    started=time.monotonic()
    global_found=locate_watermarks_batch(photos,encoder,processor,"cuda:0",
                                        args.resolution,args.detection_batch_size,True)
    edges,contexts={},{}
    found,views=refine_source_details(photos,global_found,encoder,processor,"cuda:0",
        args.resolution,args.detection_batch_size,complete=True,edge_audit=edges,context_audit=contexts)
    records=[]
    for index,(name,result) in enumerate(zip(args.files,found)):
        record={"file":name,"detail_views":views[index],"context_requeries":contexts.get(index,[]),
                "edge_completion":edges.get(index,[])}
        if isinstance(result,Exception):
            record["error"]=str(result)[:400]
        else:
            record["regions"]=[r.model_dump() for r in result[0]]
        records.append(record)
    unchanged=fingerprints==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    assert unchanged,"Source probe modified an input file"
    report={"job":args.job,"gpu":args.gpu,"load_seconds":load_seconds,
            "detection_seconds":round(time.monotonic()-started,3),"sources_unchanged":unchanged,
            "no_diffusion_candidate_generated":True,"records":records}
    target=folder/("source-context-probe-"+str(time.time_ns())+".json")
    target.write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=="__main__":
    main()
