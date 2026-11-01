"""Read-only first-candidate SR screen; no retries or original modifications."""
import argparse
import json
import os
import time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--gpu",type=int,required=True)
p.add_argument("--snapshot",type=Path,required=True)
p.add_argument("--weights",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--limit",type=int,default=8)
p.add_argument("--resolution",type=int,choices=[768,1024,1280,1536],default=1024)
a=p.parse_args()
if a.output.exists() or not 1<=a.limit<=64 or a.gpu<0:
    p.error("Fresh destination, explicit GPU, bounded cohort required")
os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
import torch

from watermark_studio.compact_sr import GENERAL_SHA256, CompactSR
from watermark_studio.config import Region
from watermark_studio.frozen_candidates import digest, record_digest
from watermark_studio.images import composite_protected, load_photo
from watermark_studio.quality import repair_texture_warnings

torch.set_num_threads(1)
assert torch.cuda.mem_get_info()[0]>8*2**30
summary=json.loads((a.snapshot/"summary.json").read_text())
rows=[json.loads(line) for line in (a.snapshot/"results.jsonl").read_text().splitlines()]
manifest={r["file"]:r for r in json.loads((a.snapshot/"manifest.json").read_text())}
chosen=[r for r in rows if "repair_texture_collapse" in r["quality"].get("guard_codes",[])
        and "source_physical_ambiguity" not in r["quality"].get("guard_codes",[])]
chosen.sort(key=lambda r:r["quality"].get("repair_texture",{}).get("count",0),reverse=True)
a.output.mkdir(parents=True)
at=time.monotonic()
engine=CompactSR(a.weights)
load_seconds=time.monotonic()-at
records=[]
for index,row in enumerate(chosen[:a.limit]):
    name=row["file"];expected=manifest[name]
    source=Path(summary["input_dir"])/name;candidate=Path(summary["output_dir"])/name
    assert digest(source)==expected["source_sha256"]
    assert digest(candidate)==expected["candidate_sha256"]
    assert record_digest(row)==expected["record_sha256"]
    before=load_photo(source)[0];result=load_photo(candidate)[0]
    regions=[Region.model_validate(r) for r in row["repair_regions"]]
    cores={tuple(item["mask"]):item["core"] for item in row.get("protected_core_boxes",[])}
    torch.cuda.synchronize();at=time.monotonic()
    enhanced=engine.enhance(result,reference_resolution=a.resolution)
    torch.cuda.synchronize();sr_seconds=time.monotonic()-at
    # Already harmonized first result: do not run another colour fit or erase its
    # recorded raw-generation drift diagnostics. Preserve those in the record.
    enhanced=composite_protected(before,enhanced,regions,12,"feather",core_boxes=cores)
    enhanced.info["harmonization"]=row.get("harmonization")
    finding=repair_texture_warnings(enhanced,regions,before=before,inset_search=True,core_boxes=row.get("protected_core_boxes",[]))
    dest=a.output/f"{index:03d}.png";enhanced.save(dest,compress_level=1)
    assert enhanced.size==before.size
    assert digest(source)==expected["source_sha256"]
    assert digest(candidate)==expected["candidate_sha256"]
    records.append({"file":name,"output":str(dest),"source_sha256":expected["source_sha256"],
        "first_candidate_sha256":expected["candidate_sha256"],"enhanced_sha256":digest(dest),
        "native_size":list(before.size),"sr_seconds":round(sr_seconds,4),
        "previous_texture":row["quality"].get("repair_texture"),"new_texture":finding,
        "original_harmonization":row.get("harmonization"),"regions":row["repair_regions"],
        "protected_core_boxes":row.get("protected_core_boxes",[]),"source":str(source),
        "first_candidate":str(candidate),"source_scene":row.get("source_scene")})
report={"scope":"experimental postprocessing, not quality-approved; no new Qwen generation",
        "weights_sha256":GENERAL_SHA256,"code_revision":__import__("subprocess").check_output(
            ["git","rev-parse","--short","HEAD"],text=True).strip(),
        "reference_resolution":a.resolution,"load_seconds":round(load_seconds,4),
        "cases":len(records),"source_and_first_candidates_unchanged":True,"rows":records}
(a.output/"report.json").write_text(json.dumps(report,indent=2))
print(json.dumps({"cases":len(records),"load_seconds":round(load_seconds,3),
    "sr_seconds":round(sum(r["sr_seconds"] for r in records),3),
    "texture_clear":sum(not r["new_texture"]["count"] for r in records),
    "source_and_first_candidates_unchanged":True}),flush=True)
