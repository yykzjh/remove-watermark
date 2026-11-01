"""Read-only source identity calibration; no diffusion generation or output approval."""
import argparse
import hashlib
import json
import os
import runpy
import subprocess
import time
import urllib.request
from pathlib import Path


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--controls",type=Path,required=True)
    p.add_argument("--physical-from",type=Path,nargs="+",required=True)
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.gpu<0 or a.output.exists():
        p.error("Explicit GPU and new report required")
    helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as response:
        server=helper["require_disjoint_server_gpus"](json.load(response),[a.gpu],True)
    revision=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    items={}
    for case in helper["known_control_cases"](a.controls):
        for region in case["regions"]:
            key=(hashlib.sha256(Path(case["source"]).read_bytes()).hexdigest(),tuple(region["box"]))
            cores={tuple(c["mask"]):c["core"] for c in case.get("source_cores",[])}
            items.setdefault(key,{"source":case["source"],"region":region,
                "core":cores.get(tuple(region["box"]),region["box"]),
                "expected_digital":True,"cohort":"planted_digital_overlay"})
    for report in a.physical_from:
        for case in map(json.loads,report.read_text().splitlines()):
            dispute=case["quality"].get("cross_review",{})
            for finding in dispute.get("localization",{}).get("findings",[]):
                if finding.get("issue")!="physical_text":
                    continue
                box=finding["box"];key=(hashlib.sha256(Path(case["source"]).read_bytes()).hexdigest(),tuple(box))
                if key in items:
                    raise ValueError("Contradictory source reference labels")
                items[key]={"source":case["source"],"region":{"kind":"panel","box":box},
                    "core":box,"expected_digital":False,"cohort":"native_physical_candidate",
                    "reference_report":str(report),"reference_claim":finding["observation"]}
    cases=list(items.values())
    if not any(not c["expected_digital"] for c in cases):
        raise ValueError("Missing native physical negative controls")
    paths={c["source"] for c in cases}
    def digests():
        return {name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in paths}
    hashes=digests()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
    import torch
    from transformers import Qwen3VLProcessor

    from watermark_studio.config import DeploymentConfig, Region
    from watermark_studio.images import load_photo
    from watermark_studio.model_identity import compare_model_weights
    from watermark_studio.quality_loading import load_quality_encoder, load_quality_processor
    from watermark_studio.source_digital_proof import certify_gray_regions
    torch.set_num_threads(1)
    root=Path.home()/".cache/watermark-models/models"
    editor_path=root/"Qwen--Qwen-Image-2.1-Turbo/snapshots/master/text_encoder"
    verifier_path=root/"Qwen--Qwen3-VL-4B-Instruct/snapshots/master"
    identity=compare_model_weights(editor_path,verifier_path)
    if identity["relation"]!="different_weights":
        raise ValueError("Independent checkpoints required")
    cfg=DeploymentConfig(quality_verifier=True,quality_backend="independent",quality_quantization="none")
    start=time.monotonic()
    editor=load_quality_encoder(editor_path,cfg,"cuda:0")
    ep=Qwen3VLProcessor.from_pretrained(editor_path.parent/"processor",local_files_only=True)
    verifier=load_quality_encoder(verifier_path,cfg,"cuda:0");vp=load_quality_processor(verifier_path)
    load_seconds=time.monotonic()-start
    start=time.monotonic();rows=[]
    for case in cases:
        source=load_photo(case["source"])[0];region=Region.model_validate(case["region"])
        proof=certify_gray_regions(source,[region],{region.box:tuple(case["core"])},[],editor,ep,
            verifier,vp,"cuda:0",identity["relation"])[0]
        rows.append(case|{"proof":proof})
        print(json.dumps({"completed":len(rows),"total":len(cases)}),flush=True)
    assert digests()==hashes and revision==subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as response:
        assert server==helper["require_disjoint_server_gpus"](json.load(response),[a.gpu],True)
    summary={"scope":"Source identity development calibration; native negative candidates are reviewed references, not blind labels",
        "code_revision":revision,"weight_identity":identity,"server_isolation":server,
        "seconds":round(time.monotonic()-start,3),"load_seconds":round(load_seconds,3),"source_files_unchanged":True,
        "planted_digital_count":sum(c["expected_digital"] for c in rows),
        "planted_digital_approved":sum(c["expected_digital"] and c["proof"]["approved"] for c in rows),
        "native_physical_count":sum(not c["expected_digital"] for c in rows),
        "native_physical_approved":sum(not c["expected_digital"] and c["proof"]["approved"] for c in rows),"rows":rows}
    a.output.write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!="rows"}),flush=True)


if __name__=="__main__":
    main()
