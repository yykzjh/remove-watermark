"""CPU-only ablation of texture statistics on fixed source-mask controls."""
from __future__ import annotations

import argparse
import inspect
import json
import math
import runpy
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--controls",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--caps",type=float,nargs="+",default=[8,12,16,24,32])
    args=parser.parse_args()
    if args.output.exists() or any(not math.isfinite(v) or not 0<v<=255 for v in args.caps):
        parser.error("Fresh output and finite caps in (0,255] required")
    from watermark_studio import quality
    from watermark_studio.config import Region
    from watermark_studio.images import load_photo

    helper=runpy.run_path(str(Path(__file__).with_name("benchmark-verifier-known-source.py")))
    cases=helper["known_control_cases"](args.controls)
    before=helper["file_hashes"](cases)
    source=inspect.getsource(quality.repair_texture_warnings)
    token="stats.append(float(tile.mean()))"
    if source.count(token)!=1:
        raise ValueError("Unexpected texture implementation; no implicit patch")
    variants=[("base",quality.repair_texture_warnings)]
    for cap in args.caps:
        namespace=dict(vars(quality))
        compiled=compile(source.replace(token,f"stats.append(float(np.minimum(tile,{cap!r}).mean()))"),
                         "<offline-texture-ablation>","exec")
        exec(compiled,namespace)  # noqa: S102 - checked project source and finite numeric constants only
        variants.append((f"cap_{cap:g}",namespace["repair_texture_warnings"]))
    values={name:[] for name,_ in variants}
    started=time.monotonic()
    for case in cases:
        original=load_photo(case["source"])[0]
        result=load_photo(case["output"])[0]
        regions=[Region.model_validate(r) for r in case["regions"]]
        for name,function in variants:
            check=function(result,regions,12,before=original)
            values[name].append({"file":case["file"],"kind":case["kind"],"expected":case["expected"],
                                 "warned":bool(check["count"])})
    assert before==helper["file_hashes"](cases),"CPU inspection changed source or controlled outputs"
    base={r["file"] for r in values["base"] if r["warned"] and r["expected"]=="review"}
    summary={}
    for name,rows in values.items():
        warnings={r["file"] for r in rows if r["warned"] and r["expected"]=="review"}
        summary[name]={"clean_warned":[r["file"] for r in rows if r["warned"] and r["expected"]=="passed"],
                       "defect_warned":len(warnings),"lost_defect_warnings":sorted(base-warnings),
                       "new_defect_warnings":sorted(warnings-base)}
    report={"scope":"texture guard only, not full acceptance or classification",
            "images":len(cases),"seconds":round(time.monotonic()-started,3),
            "no_gpu_model":True,"sources_and_outputs_unchanged":True,"summary":summary,"rows":values}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!="rows"}),flush=True)

if __name__=="__main__":
    main()
