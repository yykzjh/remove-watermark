"""Read-only independent side-band proof calibration; never submit image jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import subprocess
import time
import urllib.request
from collections import Counter
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def labelled_proposal(photo,mask,core,native):
    width,height=photo.size
    return {"protocol":"source_card_side_rim_v1","side":"right","mask":list(mask),"core":list(core),
        "source_size":list(photo.size),"uncertainty_gap_pixels":8,"card_edge_pixel":native[0]-8,
        "restore_pixels":list(native),"restore_box":[v/s for v,s in zip(native,(width,height,width,height))],
        "approved":False}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--mode",choices=["side","top","band"],default="side")
    p.add_argument("--proposals",type=Path,required=True)
    p.add_argument("--controls",type=Path,required=True)
    p.add_argument("--physical-fixtures",type=Path)
    p.add_argument("--files",nargs="+",help="Explicit frozen original source subset")
    p.add_argument("--max-source-cases",type=int,default=6)
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--rank",type=int,default=0)
    p.add_argument("--shards",type=int,default=1)
    p.add_argument("--tag",required=True)
    a=p.parse_args()
    if (not 0<=a.rank<a.shards<=4 or not 1<=a.max_source_cases<=64 or a.gpu<0
            or not a.tag.replace("-","").replace("_","").isalnum()):
        p.error("Bounded explicit GPU, shard, development limit and new simple tag required")
    h=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    from watermark_studio.images import load_photo, pixel_box
    data=json.loads(a.proposals.read_text())
    if data.get("version")!=1 or data.get("quality_permission") is not False:
        p.error("Unapproved immutable source-geometry artifact required")
    cases=[]
    if a.files and (len(set(a.files))!=len(a.files) or not set(a.files).issubset({c["file"] for c in data["cases"]})):
        p.error("Existing unique frozen source filenames required")
    pool=[c for c in data["cases"] if not a.files or c["file"] in set(a.files)]
    selected=[]
    for c in pool:
        targets=c["proposals"]
        if a.mode=="top":
            targets=[{"mask":item["mask"],"core":item["core"]} for item in c["source_cores"]
                if any(r["kind"]=="panel" and tuple(r["box"])==tuple(item["mask"])
                    and (r["box"][2]-r["box"][0])*(r["box"][3]-r["box"][1])>=.035 for r in c["regions"])]
        if targets:
            selected.append(c|{"proposals":targets})
    selected=selected[:a.max_source_cases]
    for c in selected:
        if digest(c["source"])!=c["source_file_sha256"]:
            raise ValueError("Frozen proposal source file changed")
        if hashlib.sha256(load_photo(c["source"])[0].tobytes()).hexdigest()!=c["source_rgb_sha256"]:
            raise ValueError("Frozen source proposal RGB changed")
        for prop in c["proposals"]:
            cases.append({"cohort":"native_candidate","file":c["file"],"source":c["source"],
                "proposal":prop,"expected_clear":None,"expected_digital":None})
    for c in h["known_control_cases"](a.controls):
        if c.get("kind")!="retained_panel":
            continue
        source=load_photo(c["source"])[0];width,height=source.size
        core=c["source_cores"][0]["core"];mask=c["regions"][0]["box"]
        l,t,r,b=pixel_box(core,source.size);cw=r-l
        # Exact fixture footprint. Labels are evaluation only and never enter prompts.
        clear=(r+8,t,min(width,r+32),b)
        gray=(l+max(8,round(cw*.05)),t+round(height*.19),
              r-max(8,round(cw*.05)),min(b-8,t+round(height*.215)))
        blue=(l+round(cw*.75),t+max(3,round(height*.01)),
              l+round(cw*.95),t+max(3,round(height*.01))+max(8,round(height*.02)))
        for label,native,expected in (("clear_photo",clear,True),("gray_backing",gray,False),("blue_backing",blue,False)):
            if min(native[2]-native[0],native[3]-native[1])<8:
                continue
            cases.append({"cohort":label,"file":c["file"],"source":c["source"],
                "proposal":labelled_proposal(source,mask,core,native),
                "expected_clear":expected,"expected_digital":True})
    if a.physical_fixtures:
        for c in json.loads(a.physical_fixtures.read_text()):
            source=load_photo(c["source"])[0];l,t,r,b=pixel_box(c["box"],source.size)
            native=(r+8,t,min(source.width,r+32),b)
            if min(native[2]-native[0],native[3]-native[1])<8:
                continue
            cases.append({"cohort":"native_physical_object","file":str(c["source"]),"source":c["source"],
                "proposal":labelled_proposal(source,c["box"],c["box"],native),
                "expected_clear":None,"expected_digital":False})
    if a.mode=="top":
        identities={}
        for c in cases:
            key=(c["source"],tuple(c["proposal"]["mask"]))
            identities.setdefault(key,c|{"expected_clear":None,
                "cohort":"synthetic_digital_card" if c["expected_digital"] is True else c["cohort"]})
        cases=list(identities.values())
    cases=cases[a.rank::a.shards]
    hashes={c["source"]:digest(c["source"]) for c in cases}
    with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as response:
        server=h["require_disjoint_server_gpus"](json.load(response),[a.gpu],True)
    revision=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    work=Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031")/a.tag
    work.mkdir(exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
    import torch
    from transformers import Qwen3VLProcessor

    from watermark_studio.config import DeploymentConfig, Region
    from watermark_studio.model_identity import compare_model_weights
    from watermark_studio.quality_loading import load_quality_encoder, load_quality_processor
    from watermark_studio.source_panel_rims import inspect_photo_rims, propose_blue_card_rims
    from watermark_studio.source_side_proof import diagnose_unoccluded_band, inspect_side_rims
    from watermark_studio.text_presence import TargetTextPresence
    torch.set_num_threads(1)
    root=Path.home()/".cache/watermark-models/models"
    ep_path=root/"Qwen--Qwen-Image-2.1-Turbo/snapshots/master"
    vp_path=root/"Qwen--Qwen3-VL-4B-Instruct/snapshots/master"
    identity=compare_model_weights(ep_path/"text_encoder",vp_path)
    if identity["relation"]!="different_weights":
        raise ValueError("Canonical independent checkpoints required")
    cfg=DeploymentConfig(quality_verifier=True,quality_backend="independent",quality_quantization="none")
    began=time.monotonic()
    editor=load_quality_encoder(ep_path/"text_encoder",cfg,"cuda:0")
    ep=Qwen3VLProcessor.from_pretrained(ep_path/"processor",local_files_only=True)
    verifier=load_quality_encoder(vp_path,cfg,"cuda:0");vp=load_quality_processor(vp_path)
    ocr=TargetTextPresence();load_seconds=time.monotonic()-began;began=time.monotonic()
    rows=[]
    for c in cases:
        photo=load_photo(c["source"])[0]
        if a.mode=="top":
            prop=c["proposal"]
            proposed=propose_blue_card_rims(photo,[Region(kind="panel",box=prop["mask"])],
                [{"mask":prop["mask"],"core":prop["core"]}])
            proof=(inspect_photo_rims(photo,proposed,editor,ep,verifier,vp,"cuda:0",
                identity["relation"],ocr)[0] if proposed else
                {"approved":False,"reason":"no_source_top_geometry"})
        elif a.mode=="band":
            proof=diagnose_unoccluded_band(photo,c["proposal"],editor,ep,verifier,vp,"cuda:0",identity["relation"],ocr)
        else:
            proof=inspect_side_rims(photo,[c["proposal"]],editor,ep,verifier,vp,"cuda:0",identity["relation"],ocr)[0]
        rows.append(c|{"proof":proof,"source_file_sha256":hashes[c["source"]]})
        print(json.dumps({"completed":len(rows),"total":len(cases)}),flush=True)
    if hashes!={name:digest(name) for name in hashes} or revision!=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip():
        raise RuntimeError("Source or code changed during read-only calibration")
    with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=2) as response:
        assert server==h["require_disjoint_server_gpus"](json.load(response),[a.gpu],True)
    summary={"scope":"Source proof calibration only; no image restoration or production job",
        "mode":a.mode,"revision":revision,"evaluations":len(rows),"images":len({c["source"] for c in rows}),"rank":a.rank,"shards":a.shards,
        "seconds":round(time.monotonic()-began,3),"load_seconds":round(load_seconds,3),
        "cohorts":dict(Counter(c["cohort"] for c in rows)),
        "approved":dict(Counter(c["cohort"] for c in rows if c["proof"]["approved"])),
        "band_consensus":dict(Counter(c["cohort"] for c in rows if c["proof"].get("band_consensus"))),
        "false_clear_bands":sum(c["proof"].get("band_consensus",False) and c["expected_clear"] is False for c in rows),
        "false_approvals":sum(c["proof"]["approved"] and
            (c["expected_clear"] is False or c["expected_digital"] is False) for c in rows),
        "sources_unchanged":True,"weight_identity":identity,"rows":rows}
    (work/"report.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (work/"sources.json").write_text(json.dumps(list(hashes),ensure_ascii=False))
    print(json.dumps({k:v for k,v in summary.items() if k not in {"rows","weight_identity"}}))


if __name__=="__main__":
    main()
