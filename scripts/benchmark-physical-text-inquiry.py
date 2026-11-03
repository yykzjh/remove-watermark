"""Read-only native/control inquiry calibration; never a new thousand-image success claim."""
import argparse
import collections
import copy
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

p=argparse.ArgumentParser(__doc__)
p.add_argument("--gpu",type=int,required=True)
p.add_argument("--manifest",type=Path)
p.add_argument("--controls",type=Path)
p.add_argument("--frozen-baseline-job")
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
assert a.gpu in (0,1,2,3) and not a.output.exists()
assert (a.manifest is None)!=(a.controls is None)
os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch

from watermark_studio.config import DeploymentConfig, Region
from watermark_studio.grounded_review import inspect_grounded  # noqa: F401 - eagerly pin the implementation
from watermark_studio.images import load_photo
from watermark_studio.model_identity import compare_model_weights
from watermark_studio.physical_text_review import reconsider
from watermark_studio.quality import verify_batch
from watermark_studio.quality_loading import load_quality_encoder, load_quality_processor
from watermark_studio.text_preservation import OutputTextPreservation

spec=importlib.util.spec_from_file_location("verifier_benchmark",Path(__file__).with_name("benchmark-verifier-known-source.py"))
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
cases=helper.manifest_cases(a.manifest) if a.manifest else helper.known_control_cases(a.controls)
assert len(cases)<=1000
hashes=helper.file_hashes(cases)
baseline={}
if a.frozen_baseline_job:
    baseline={r["file"]:r for r in map(json.loads,
        (Path.home()/".local/state/watermark-studio"/a.frozen_baseline_job/"results.jsonl").read_text().splitlines())}
    assert {c["file"] for c in cases}<=baseline.keys()
revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
torch.set_num_threads(1)
cfg=DeploymentConfig(quality_verifier=True,quality_backend="independent",quality_quantization="none")
models=Path.home()/".cache/watermark-models/models"
editor_path=models/"Qwen--Qwen-Image-2.1-Turbo/snapshots/master"
qa_path=models/"Qwen--Qwen3-VL-4B-Instruct/snapshots/master"
identity=compare_model_weights(editor_path/"text_encoder",qa_path)
assert identity["relation"]=="different_weights"
a.output.mkdir(parents=True)
load_start=time.monotonic()
editor=load_quality_encoder(editor_path/"text_encoder",cfg,"cuda:0")
ep=load_quality_processor(editor_path/"processor")
validator=load_quality_encoder(qa_path,cfg,"cuda:0")
vp=load_quality_processor(qa_path)
preserver=OutputTextPreservation();preserver.detector._engine()
print(json.dumps({"phase":"loaded","revision":revision,"gpu":a.gpu,"seconds":time.monotonic()-load_start}),flush=True)
start=last=time.monotonic();rows=[]
with (a.output/"results.jsonl").open("w") as stream:
    for case in cases:
        before,after=load_photo(case["source"])[0],load_photo(case["output"])[0]
        context=helper.case_context(case)
        if baseline:
            verdict=copy.deepcopy(baseline[case["file"]]["quality"])
        else:
            verdict=verify_batch([(before,after)],validator,vp,"cuda:0",
                reviewer="independent",resolution=768,detail_patches=4,
                source_regions=[[Region.model_validate(r) for r in case["regions"]]],
                editor_context=[context],rectangle_guard=True)[0]
        initial=copy.deepcopy(verdict)
        reconsider(before,after,verdict,editor,ep,validator,vp,"cuda:0",context,
            identity["relation"],preserver)
        row={"file":case["file"],"expected":case.get("expected"),"cohort":case["cohort"],
            "initial_quality":initial,"quality":verdict}
        rows.append(row);stream.write(json.dumps(row,ensure_ascii=False)+"\n");stream.flush()
        if time.monotonic()-last>=20:
            print(json.dumps({"phase":"progress","done":len(rows),"total":len(cases),
                "corrected":sum(r["quality"].get("physical_text_review",{}).get("status")=="corrected" for r in rows)}),flush=True)
            last=time.monotonic()
seconds=time.monotonic()-start
assert hashes==helper.file_hashes(cases)
report={"scope":"read-only verifier development ablation; NOT a first-pass thousand-image evaluation",
    "images":len(rows),"weight_identity":identity,"revision":revision,"gpu":a.gpu,"seconds":seconds,
    "initial_counts":dict(collections.Counter(r["initial_quality"]["status"] for r in rows)),
    "final_counts":dict(collections.Counter(r["quality"]["status"] for r in rows)),
    "corrected":sum(r["quality"].get("physical_text_review",{}).get("status")=="corrected" for r in rows),
    "new_known_defect_misses":[r["file"] for r in rows if r.get("expected")=="review"
        and r["quality"]["status"]=="passed" and r["initial_quality"]["status"]!="passed"],
    "known_defect_misses":[r["file"] for r in rows if r.get("expected")=="review" and r["quality"]["status"]=="passed"],
    "clean_vetoes":[r["file"] for r in rows if r.get("expected")=="passed" and r["quality"]["status"]!="passed"],
    "sources_outputs_unchanged":True,"frozen_baseline_job":a.frozen_baseline_job}
(a.output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False),flush=True)
