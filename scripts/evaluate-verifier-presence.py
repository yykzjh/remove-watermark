"""Read-only full held-out predicate evaluation; never a native-image quality pass rate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import subprocess
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--target",type=Path,required=True)
    p.add_argument("--gpu",type=int,choices=[0,1,2,3],required=True)
    p.add_argument("--adapter",type=Path)
    p.add_argument("--adapter-sha256",default="")
    p.add_argument("--base-only",action="store_true")
    p.add_argument("--limit",type=int,default=0)
    a=p.parse_args()
    if bool(a.adapter)!=bool(a.adapter_sha256) or (not a.base_only and not a.adapter) or (a.base_only and a.adapter):
        p.error("Choose base-only or an explicitly SHA-pinned presence adapter")
    assert not a.target.exists() and 0<=a.limit<=2000
    from watermark_studio.training_reservation import offline_reservation
    reservation=offline_reservation(a.gpu)
    helper=runpy.run_path(str(Path(__file__).with_name("train-verifier-presence-lora.py")))
    data,manifest_sha=helper["validate_manifest"](a.manifest)
    rows=[row for row in data["rows"] if row["split"]=="validation"]
    total=len(rows)
    if a.limit:rows=rows[:a.limit]
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
    import torch
    from transformers.utils import logging

    from watermark_studio.config import DeploymentConfig, Region
    from watermark_studio.images import load_photo
    from watermark_studio.model_identity import compare_model_weights
    from watermark_studio.quality import _structured_vqa
    from watermark_studio.quality_loading import load_quality_encoder, load_quality_processor
    from watermark_studio.quality_presence import FIELDS, PROMPT, output_chat
    from watermark_studio.training_features import digest

    logging.disable_progress_bar();torch.set_num_threads(1)
    cache=Path.home()/".cache/watermark-models/models"
    model_path=cache/"Qwen--Qwen3-VL-4B-Instruct/snapshots/master"
    identity=compare_model_weights(cache/"Qwen--Qwen-Image-2.1-Turbo/snapshots/master/text_encoder",model_path)
    assert identity["relation"]=="different_weights"
    model=load_quality_encoder(model_path,DeploymentConfig(quality_verifier=True,quality_backend="independent"),"cuda:0")
    processor=load_quality_processor(model_path)
    if a.adapter:
        from peft import PeftModel
        info=json.loads((a.adapter/"presence-task.json").read_text())
        if (digest(a.adapter/"adapter_model.safetensors")!=a.adapter_sha256 or
                info["weights_sha256"]!=a.adapter_sha256 or
                info["prompt_sha256"]!=hashlib.sha256(PROMPT.encode()).hexdigest() or
                info.get("base_identity",{}).get("verifier_sha256")!=identity["verifier_sha256"]):
            raise ValueError("Adapter weight/base/protocol mismatch")
        model=PeftModel.from_pretrained(model,a.adapter,is_trainable=False,local_files_only=True)
        model.eval()
    a.target.mkdir(parents=True)
    revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    began=time.monotonic();results=[]
    with (a.target/"results.jsonl").open("x") as output:
        for index,row in enumerate(rows):
            chat=output_chat(load_photo(row["before"])[0],load_photo(row["after"])[0],
                [Region.model_validate(r) for r in row["regions"]],detail_layout=row.get("detail_layout"))
            values,seconds=_structured_vqa([chat],FIELDS,model,processor,"cuda:0",96)
            record={"id":row["id"],"parent_kind":row["parent_kind"],"overlay_present":row["overlay_present"],
                "uncertain":row["uncertain"],"prediction":values[0],"seconds":seconds}
            results.append(record);output.write(json.dumps(record)+"\n");output.flush()
            if (index+1)%16==0:
                print(json.dumps({"done":index+1,"total":len(rows)}),flush=True)
    unsafe=[r["id"] for r in results if (r["overlay_present"] or r["uncertain"]) and
            not r["prediction"]["overlay_present"] and not r["prediction"]["uncertain"]]
    missed=[r["id"] for r in results if r["overlay_present"] and not r["prediction"]["overlay_present"]]
    refused=[r["id"] for r in results if not r["overlay_present"] and not r["uncertain"] and
             (r["prediction"]["overlay_present"] or r["prediction"]["uncertain"])]
    assert digest(a.manifest)==manifest_sha
    assert revision==subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    assert reservation==offline_reservation(a.gpu)
    report={"state":"completed_full_validation" if len(rows)==total else "completed_subset",
        "rows":len(rows),"validation_total":total,"unsafe_clear_files":unsafe,
        "overlay_false_negative_files":missed,"clean_refused_files":refused,
        "seconds":time.monotonic()-began,"manifest_sha256":manifest_sha,"revision":revision,
        "adapter_sha256":a.adapter_sha256,"base_only":a.base_only,"weight_identity":identity,
        "source_pixels_changed":False,"additional_image_generations":0,"production_adopted":False,
        "scope":"Source-disjoint synthetic/weak-label raw predicate evaluation. Not native dewatermark quality or 1000-image qualification."}
    (a.target/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if not k.endswith("_files")}|
                     {"unsafe_clear":len(unsafe),"overlay_false_negatives":len(missed),
                      "clean_refused":len(refused)}),flush=True)


if __name__=="__main__":
    main()
