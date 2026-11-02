"""Read-only single-purpose faint-matte calibration; never alters verdicts or images."""
from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import time
import urllib.request
from collections import Counter
from pathlib import Path


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--controls",type=Path,required=True)
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--tag",required=True)
    a=p.parse_args()
    if a.gpu<0 or not a.tag.replace("-","").replace("_","").isalnum():
        p.error("Explicit GPU and new simple tag required")
    helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as response:
        isolation=helper["require_disjoint_server_gpus"](json.load(response),[a.gpu],True)
    cases=[c for c in helper["known_control_cases"](a.controls)
        if c["kind"] in ("identity","exact_removal","exact_jpeg","shadow_18","shadow_32")]
    if len(cases)!=45:
        p.error("Fixed 27 clean and 18 shadow challenge pairs required")
    hashes=helper["file_hashes"](cases)
    revision=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    work=a.controls.parent/a.tag;work.mkdir(exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu);os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
    import torch
    from transformers import Qwen3VLProcessor

    from watermark_studio.config import DeploymentConfig, Region
    from watermark_studio.images import load_photo
    from watermark_studio.model_identity import compare_model_weights
    from watermark_studio.quality_loading import load_quality_encoder, load_quality_processor
    from watermark_studio.shadow_assessment import assess_shadow, shadow_chats
    torch.set_num_threads(1);root=Path.home()/".cache/watermark-models/models"
    eroot=root/"Qwen--Qwen-Image-2.1-Turbo/snapshots/master"
    vroot=root/"Qwen--Qwen3-VL-4B-Instruct/snapshots/master"
    identity=compare_model_weights(eroot/"text_encoder",vroot)
    if identity["relation"]!="different_weights":
        raise ValueError("Independent canonical checkpoints required")
    cfg=DeploymentConfig(quality_verifier=True,quality_backend="independent",quality_quantization="none")
    began=time.monotonic()
    editor=load_quality_encoder(eroot/"text_encoder",cfg,"cuda:0")
    ep=Qwen3VLProcessor.from_pretrained(eroot/"processor",local_files_only=True)
    verifier=load_quality_encoder(vroot,cfg,"cuda:0");vp=load_quality_processor(vroot)
    load_seconds=time.monotonic()-began;began=time.monotonic();rows=[]
    with (work/"results.jsonl").open("x") as stream:
        for c in cases:
            before=load_photo(c["source"])[0];after=load_photo(c["output"])[0]
            chats=shadow_chats(before,after,[Region.model_validate(r) for r in c["regions"]])
            row={k:c[k] for k in ("file","kind","expected")};row["approved"]=False
            for label,model,processor in (("editor",editor,ep),("independent",verifier,vp)):
                try:row[label]=assess_shadow(chats,model,processor,"cuda:0")
                except Exception as error:row[label]={"error":str(error)[:200]}  # noqa: BLE001
            states=[row[k].get("repair_state") for k in ("editor","independent")]
            row["both_defect"]=all(x=="确认有缺陷" for x in states)
            row["either_defect"]=any(x=="确认有缺陷" for x in states)
            rows.append(row);stream.write(json.dumps(row,ensure_ascii=False)+"\n");stream.flush()
            print(json.dumps({"completed":len(rows),"total":len(cases)}),flush=True)
    if hashes!=helper["file_hashes"](cases) or revision!=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip():
        raise ValueError("Source, candidate or code changed")
    with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as response:
        if isolation!=helper["require_disjoint_server_gpus"](json.load(response),[a.gpu],True):
            raise ValueError("Server GPU reservation changed during diagnosis")
    report={"scope":"Diagnostic only; no approvals, guard overrides or image modifications",
        "revision":revision,"total":len(rows),"seconds":round(time.monotonic()-began,3),
        "load_seconds":round(load_seconds,3),"weight_identity":identity,"server_isolation":isolation,
        "both_defect":dict(Counter(r["kind"] for r in rows if r["both_defect"])),
        "either_defect":dict(Counter(r["kind"] for r in rows if r["either_defect"])),
        "false_flags_both":sum(r["both_defect"] and r["expected"]=="passed" for r in rows),
        "false_flags_either":sum(r["either_defect"] and r["expected"]=="passed" for r in rows),
        "sources_and_candidates_unchanged":True}
    (work/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False))


if __name__=="__main__":
    main()
