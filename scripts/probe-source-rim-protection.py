"""Read-only source-rim protection experiment; first candidates stay immutable."""
import argparse
import json
import os
import re
import time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--gpu",type=int,required=True)
p.add_argument("--source-job",required=True)
p.add_argument("--candidate-job",required=True)
p.add_argument("--files",nargs="+",required=True)
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
if a.output.exists() or a.gpu<0 or not 1<=len(set(a.files))==len(a.files)<=8 or any(
        not re.fullmatch(r"[a-f0-9]{32}",j) for j in [a.source_job,a.candidate_job]):
    p.error("Fresh experiment directory, explicit GPU, bounded unique cases required")
os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch

from watermark_studio.config import DeploymentConfig, Region
from watermark_studio.frozen_candidates import bounded, digest
from watermark_studio.images import load_photo
from watermark_studio.model_identity import compare_model_weights
from watermark_studio.quality import repair_boundary_warnings, repair_texture_warnings
from watermark_studio.quality_loading import load_quality_encoder, load_quality_processor
from watermark_studio.rectangle_phase import rectangle_phase
from watermark_studio.source_panel_rims import (
    inspect_photo_rims,
    propose_blue_card_rims,
    restore_proven_rims,
)
from watermark_studio.text_presence import TargetTextPresence

torch.set_num_threads(1)
assert torch.cuda.mem_get_info()[0]>32*2**30,"Use an idle GPU"
root=Path.home()/".local/state/watermark-studio"
def job_read(job):
    summary=json.loads((root/job/"summary.json").read_text())
    assert summary["state"]=="completed"
    rows={r["file"]:r for r in map(json.loads,(root/job/"results.jsonl").read_text().splitlines())}
    assert len(rows)==summary["total"] and set(a.files)<=rows.keys()
    assert all(rows[n].get("ok") and rows[n].get("quality_attempts",1)==1
               and rows[n].get("model_inferences")==1 for n in a.files)
    return summary,rows
baseline,old=job_read(a.source_job);candidate,current=job_read(a.candidate_job)
models=Path.home()/".cache/watermark-models/models"
editor=models/"Qwen--Qwen-Image-2.1-Turbo/snapshots/master"
validator=models/"Qwen--Qwen3-VL-4B-Instruct/snapshots/master"
identity=compare_model_weights(editor/"text_encoder",validator)
assert identity["relation"]=="different_weights"
began=time.monotonic()
config=DeploymentConfig(quality_verifier=True,quality_backend="independent")
encoder=load_quality_encoder(editor/"text_encoder",config,"cuda:0")
from transformers import Qwen3VLProcessor

processor=Qwen3VLProcessor.from_pretrained(editor/"processor",local_files_only=True)
judge=load_quality_encoder(validator,config,"cuda:0")
judge_processor=load_quality_processor(validator)
detector=TargetTextPresence()
load_seconds=time.monotonic()-began
a.output.mkdir(parents=True);records=[]
for index,name in enumerate(a.files):
    row=current[name];original=bounded(Path(baseline["input_dir"]),name)
    source_pair=bounded(Path(candidate["input_dir"]),name);generated=bounded(Path(candidate["output_dir"]),name)
    before_hash=digest(original);after_hash=digest(generated)
    assert before_hash==digest(source_pair) and old[name]["repair_regions"]==row["repair_regions"]
    photo=load_photo(original)[0];first=load_photo(generated)[0]
    regions=[Region.model_validate(r) for r in row["repair_regions"]]
    proposals=propose_blue_card_rims(photo,regions,row.get("protected_core_boxes",[]))
    at=time.monotonic()
    proofs=inspect_photo_rims(photo,proposals,encoder,processor,judge,judge_processor,
                             "cuda:0",identity["relation"],detector)
    restored,count=restore_proven_rims(photo,first,proofs)
    effective=[];effective_cores=[];photo_checks=[]
    for region in regions:
        proof=next((r for r in proofs if r["mask"]==list(region.box) and r.get("approved")),None)
        target=Region.model_validate(proof["guard_region"]) if proof else region
        effective.append(target)
        core=next((v["core"] for v in row.get("protected_core_boxes",[]) if v["mask"]==list(region.box)),None)
        if core:
            effective_cores.append({"mask":list(target.box),"core":[core[0],max(core[1],target.box[1]),core[2],core[3]]})
        if proof:
            # Integer bounds are authoritative and avoid a floating extra row.
            box=tuple(proof["restore_pixels"])
            photo_checks.append({"box":list(box),"source_pixels_exact":
                photo.crop(box).tobytes()==restored.crop(box).tobytes()})
    cue=lambda image,rs,cs,reference=photo:{"boundary":repair_boundary_warnings(image,rs,before=reference),
        "texture":repair_texture_warnings(image,rs,before=reference,core_boxes=cs),
        "phase":rectangle_phase(reference,image,rs)}
    before_cues=cue(first,regions,row.get("protected_core_boxes",[]))
    after_cues=cue(restored,effective,effective_cores)
    output=a.output/f"{index:03d}.png";restored.save(output,compress_level=1)
    assert before_hash==digest(original) and after_hash==digest(generated)
    records.append({"file":name,"source":str(original),"candidate":str(generated),"output":str(output),
        "source_sha256":before_hash,"candidate_sha256":after_hash,"proofs":proofs,
        "restored_rims":count,"photo_checks":photo_checks,"seconds":round(time.monotonic()-at,4),
        "original_cues":before_cues,"precise_boundary_cues":after_cues,
        "original_harmonization":row.get("harmonization"),
        "source_regions":row["repair_regions"],"effective_guard_regions":[r.model_dump() for r in effective],
        "effective_guard_cores":effective_cores,"native_size":list(photo.size)})
report={"scope":"experimental original-photo rim protection; not approved quality or fresh generation",
        "code_revision":__import__("subprocess").check_output(["git","rev-parse","--short","HEAD"],text=True).strip(),
        "identity":identity,"load_seconds":round(load_seconds,3),"cases":len(records),
        "source_and_first_candidate_unchanged":True,"rows":records}
(a.output/"report.json").write_text(json.dumps(report,indent=2))
print(json.dumps({"cases":len(records),"load_seconds":report["load_seconds"],"cases_restored":sum(
    r["restored_rims"]>0 for r in records),"rows":[{"file":r["file"],"seconds":r["seconds"],
    "proposals":len(r["proofs"]),"restored":r["restored_rims"],"photo_checks":r["photo_checks"],
    "original_cue_counts":{k:v["count"] for k,v in r["original_cues"].items()},
    "precise_cue_counts":{k:v["count"] for k,v in r["precise_boundary_cues"].items()},
    "proofs":[{k:p.get(k) for k in ["reason","editor","independent","ocr"]} for p in r["proofs"]]} for r in records]}),flush=True)
