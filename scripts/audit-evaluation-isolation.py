"""Audit frozen evaluation source pixels against discoverable history; never infer or edit images."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from watermark_studio.images import EXTENSIONS
from watermark_studio.source_fingerprints import FingerprintIndex, fingerprint_file


def source_values(values):
    if not isinstance(values,list):raise TypeError("Source provenance must be a list")
    result=[]
    for value in values:
        path=value.get("path") if isinstance(value,dict) else value
        if not isinstance(path,str):raise TypeError("Source provenance entry lacks a path")
        result.append(path)
    return result


def discover(root,sealed,state,unexposed_reservations=()):
    reservations={sealed,*unexposed_reservations}
    paths=set();catalogs=[];unresolved=[]
    for f in sorted(root.glob("*/sources.json")):
        if f.parent in reservations:continue
        paths.update(source_values(json.loads(f.read_text())));catalogs.append(str(f))
    for f in sorted(root.glob("*/source-selection.json")):
        if f.parent in reservations:continue
        data=json.loads(f.read_text());catalogs.append(str(f))
        if data.get("sources"):
            paths.update(source_values(data["sources"]));continue
        for name in data.get("source_sha256",{}):
            options=[Path(name)] if Path(name).is_absolute() else [root/"八标照片集"/name,f.parent/"input"/name]
            selected=next((str(p) for p in options if p.is_file()),None)
            if selected:paths.add(selected)
            else:unresolved.append({"catalog":str(f),"path":name})
    for f in sorted(state.glob("*/summary.json")):
        if not re.fullmatch("[a-f0-9]{32}",f.parent.name):continue
        data=json.loads(f.read_text());input_dir=data.get("input_dir");results=f.parent/"results.jsonl"
        if not input_dir or not results.exists():continue
        catalogs.append(str(f))
        for line in results.read_text().splitlines():
            row=json.loads(line);name=row.get("file")
            if isinstance(name,str):paths.add(str(Path(input_dir)/name))
    photos=sorted({str(Path(p).resolve()) for p in paths if Path(p).suffix.lower() in EXTENSIONS})
    missing=[p for p in photos if not Path(p).is_file()]
    return [p for p in photos if Path(p).is_file()],catalogs,unresolved,missing


def validate_unexposed_parent(root,sealed,state,parent):
    """Skip only an immutable, never-inferred reservation directly bound to the new selection."""
    parent=parent.resolve(strict=True)
    if parent.parent!=root or parent==sealed:
        raise ValueError("Reservation parent must be a separate sibling")
    if {p.name for p in parent.iterdir()}!={"input","sources.json","source-selection.json"}:
        raise ValueError("Parent contains artifacts beyond a source-only reservation")
    raw=(parent/"source-selection.json").read_bytes()
    current=json.loads((sealed/"source-selection.json").read_text())
    if current.get("refinement",{}).get("parent_selection_sha256")!=hashlib.sha256(raw).hexdigest():
        raise ValueError("Parent must match the new selection immutable lineage")
    for f in state.glob("*/summary.json"):
        if not re.fullmatch("[a-f0-9]{32}",f.parent.name):continue
        input_dir=json.loads(f.read_text()).get("input_dir")
        if input_dir and Path(input_dir).resolve().is_relative_to(parent):
            raise ValueError("Parent was exposed to an API job; it must remain historical")
    return parent


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--sealed",type=Path,required=True)
    p.add_argument("--target",type=Path,required=True)
    p.add_argument("--state-dir",type=Path,default=Path.home()/".local/state/watermark-studio")
    p.add_argument("--extra-history",type=Path)
    p.add_argument("--known-history-gaps",type=Path)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--unexposed-reservation-parent",type=Path,
        help="Direct immutable parent of a pre-inference alias refinement; API exposure is forbidden")
    a=p.parse_args()
    root=a.root.resolve(strict=True);sealed=a.sealed.resolve(strict=True)
    if not 1<=a.workers<=8 or sealed.parent!=root or a.target.resolve().parent!=root:
        p.error("Bounded remote fingerprint audit under the experiment root required")
    selection=sealed/"source-selection.json";raw_selection=selection.read_bytes();data=json.loads(raw_selection)
    sources=source_values(data["sources"])
    if len(sources)<1000 or len(sources)>4096:raise ValueError("Frozen >=1000-source selection required")
    frozen=[]
    for path in sources:
        path=Path(path).resolve(strict=True);relative=path.relative_to(root/"八标照片集").as_posix()
        expected=data["source_sha256"].get(relative) or data["source_sha256"].get(str(path))
        if not isinstance(expected,str) or not re.fullmatch("[a-f0-9]{64}",expected):
            raise ValueError("Every frozen source needs its exact original encoded SHA")
        frozen.append((str(path),expected))
    if len({p for p,_ in frozen})!=len(frozen) or len({s for _,s in frozen})!=len(frozen):
        raise ValueError("Frozen selection encoded identities already repeat")
    parents=([validate_unexposed_parent(root,sealed,a.state_dir,a.unexposed_reservation_parent)]
        if a.unexposed_reservation_parent else [])
    prior,catalogs,unresolved,missing=discover(root,sealed,a.state_dir,parents)
    history_gaps=json.loads(a.known_history_gaps.read_text()) if a.known_history_gaps else []
    if not isinstance(history_gaps,list) or len(history_gaps)>128:
        raise ValueError("Bounded explicit unresolved history ledger required")
    if len(prior)>50000:raise ValueError("Historical inventory unexpectedly exceeds bound")
    a.target.mkdir(exist_ok=False);work=a.target.resolve();began=time.monotonic()
    index=FingerprintIndex();errors=[];matches=[];encoded=set();sealed_encoded=set();scanned=0
    def calculate(item):
        partition,path,expected=item
        try:
            result=fingerprint_file(path)
            if expected and result["encoded_sha256"]!=expected:raise ValueError("Frozen encoded source bytes changed")
            return {**result,"partition":partition},None
        except Exception as error:  # noqa: BLE001 - incomplete audits never become qualification
            return None,{"partition":partition,"path":path,"error":str(error)[:240]}
    def append(value,stream):
        stream.write(json.dumps(value,ensure_ascii=False)+"\n");stream.flush()
    with (work/"fingerprints.jsonl").open("x") as stream:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for result,error in pool.map(calculate,[("history",path,None) for path in prior]):
                scanned+=1
                if error:errors.append(error)
                else:
                    append(result,stream)
                    if result["encoded_sha256"] not in encoded:
                        index.add(result);encoded.add(result["encoded_sha256"])
                if scanned%200==0:print(json.dumps({"stage":"history_hashes","completed":scanned,"total":len(prior)}),flush=True)
        if a.extra_history:
            if a.extra_history.stat().st_size>100000:raise ValueError("Extra fingerprints exceed bound")
            extra=json.loads(a.extra_history.read_text())
            if not isinstance(extra,list) or len(extra)>32:raise ValueError("Bounded extra source fingerprints required")
            for value in extra:
                value={**value,"partition":"extra_history"}
                # Validate descriptor size and required match fields without opening an image.
                if len(value["small_rgb16_base64"])>1100 or len(value["phash64"])!=16:
                    raise ValueError("Invalid extra fingerprint shape")
                index.add(value);append(value,stream)
        history_unique=len({v["encoded_sha256"] for v in index.values})
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for number,(result,error) in enumerate(pool.map(calculate,[("sealed",path,sha) for path,sha in frozen]),1):
                if error:errors.append(error)
                else:
                    for other,relation in index.matches(result):
                        matches.append({"sealed_path":result["path"],"sealed_sha256":result["encoded_sha256"],
                            "other_path":other["path"],"other_partition":other["partition"],**relation})
                    index.add(result);sealed_encoded.add(result["encoded_sha256"]);append(result,stream)
                if number%100==0:print(json.dumps({"stage":"sealed_hashes","completed":number,"total":len(frozen)}),flush=True)
    if selection.read_bytes()!=raw_selection:raise ValueError("Frozen selection changed during audit")
    report={"version":1,"state":"completed","qualified":not (matches or errors or missing or unresolved or history_gaps),
        "selection_sha256":hashlib.sha256(raw_selection).hexdigest(),"sealed_sources":len(frozen),
        "sealed_encoded_unique":len(sealed_encoded),"historical_paths":len(prior),"historical_unique_encoded":history_unique,
        "catalogs":len(catalogs),"unexposed_reservation_parents":[str(v) for v in parents],"unresolved_catalog_paths":len(unresolved),"missing_history_paths":len(missing),
        "fingerprint_errors":len(errors),"known_history_gaps":len(history_gaps),"exact_matches":sum(r["relation"] in ("encoded_equal","decoded_rgb_equal") for r in matches),
        "near_duplicate_candidates":sum(r["relation"].startswith("near_") for r in matches),
        "seconds":round(time.monotonic()-began,3),"model_calls":0,"images_changed":0,"previews_created":0,
        "scope":"Fingerprint audit of discoverable sources only; near candidates are not quality labels or automatic replacements"}
    for name,value in [("report.json",report),("matches.json",matches),("errors.json",errors),
            ("missing.json",missing),("unresolved.json",unresolved),("known-history-gaps.json",history_gaps),("catalogs.json",catalogs)]:
        (work/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
