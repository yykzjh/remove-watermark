"""Read-only numeric replay using the exact opacity cores already used for ROI composition."""
import argparse
import concurrent.futures
import json
from pathlib import Path

from watermark_studio.config import Region
from watermark_studio.images import load_photo
from watermark_studio.quality import repair_texture_warnings
from watermark_studio.rectangle_phase import rectangle_phase_tight


def recorded_cores(row,size):
    regions=row.get("regions",[])
    values=row.get("opaque_core_boxes",[])
    if (row.get("processing_path")!="region_repair" or len(values)!=len(regions)
            or any(row.get("conditioning_prefill",[]))):
        return []
    out=[]
    for region,core in zip(regions,values):
        if core is None:continue
        w,h=size;l,t,r,b=region["box"]
        c=[v/s for v,s in zip(core,(w,h,w,h))]
        if l<=c[0]<c[2]<=r and t<=c[1]<c[3]<=b:
            out.append({"mask":region["box"],"core":c})
    return out


def run_case(args):
    row,source,target=args
    before=load_photo(source)[0];after=load_photo(target)[0]
    cores=recorded_cores(row,before.size)
    regions=[Region.model_validate(v) for v in row.get("repair_regions",[])]
    old=row.get("quality",{})
    texture=repair_texture_warnings(after,regions,before=before,core_boxes=cores)
    phase=rectangle_phase_tight(before,after,regions,cores)
    return {"file":row["file"],"old_status":old.get("status"),"model_status":old.get("model_status"),
            "old_guards":old.get("guard_codes",[]),"cores":cores,
            "old_texture":old.get("repair_texture",{}).get("count",0),"texture":texture["count"],
            "supports":texture["source_reference_support"],
            "old_phase":old.get("repair_rectangle_phase",{}).get("count",0),"phase":phase["count"]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--job",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--only-review",action="store_true")
    a=p.parse_args()
    assert not a.output.exists() and 1<=a.workers<=8
    root=Path.home()/".local/state/watermark-studio"/a.job
    summary=json.loads((root/"summary.json").read_text())
    rows=[json.loads(x) for x in (root/"results.jsonl").read_text().splitlines()]
    if a.only_review:rows=[r for r in rows if r.get("quality",{}).get("status")=="review"]
    tasks=[(r,str(Path(summary["input_dir"])/r["file"]),str(Path(summary["output_dir"])/r["file"])) for r in rows]
    with concurrent.futures.ProcessPoolExecutor(a.workers) as pool:
        results=list(pool.map(run_case,tasks))
    report={"job":a.job,"scope":"numeric measurement only; no approval or image edits",
        "cases":len(results),"with_recorded_cores":sum(bool(r["cores"]) for r in results),
        "texture_cleared":sum(r["old_texture"]>0 and r["texture"]==0 for r in results),
        "texture_added":sum(r["old_texture"]==0 and r["texture"]>0 for r in results),
        "phase_cleared":sum(r["old_phase"]>0 and r["phase"]==0 for r in results),
        "phase_added":sum(r["old_phase"]==0 and r["phase"]>0 for r in results),
        "rows":results}
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!="rows"}),flush=True)


if __name__=="__main__":main()
