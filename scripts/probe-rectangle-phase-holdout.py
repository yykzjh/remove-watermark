"""New randomized artifact-only controls; original dataset files stay immutable."""
import argparse
import hashlib
import io
import json
import random
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageDraw

from watermark_studio.detection import parse_detection
from watermark_studio.images import load_photo
from watermark_studio.rectangle_phase import (
    rectangle_phase,
    rectangle_phase_core,
    rectangle_phase_inset,
    rectangle_phase_sharp,
    rectangle_phase_tight,
)

p=argparse.ArgumentParser()
p.add_argument("--source-job",required=True)
p.add_argument("--images",type=int,default=64)
p.add_argument("--seed",type=int,default=202611022)
p.add_argument("--cue",choices=["phase","uniform"],default="phase")
p.add_argument("--core-boundaries",action="store_true")
p.add_argument("--sharp-core",action="store_true",help="Offline original smoothing with fixed native-source hypotheses; no approval")
p.add_argument("--inset-core",action="store_true",help="Offline fixed safety-padding offset; preserves every v5 veto")
p.add_argument("--tight-core",action="store_true",help="Additional narrow source-core search; offline only")
p.add_argument("--output",type=Path,required=True)
p.add_argument("--exclude-probes",type=Path,nargs="*",default=[])
a=p.parse_args()
if sum((a.core_boundaries,a.tight_core,a.inset_core,a.sharp_core))>1:
    p.error("Choose one phase diagnostic")
if a.output.exists() or not 8<=a.images<=128:
    p.error("Fresh output, 8..128 backgrounds required")
folder=Path.home()/".local/state/watermark-studio"/a.source_job
summary=json.loads((folder/"summary.json").read_text())
assert summary["state"]=="completed"
root=Path(summary["input_dir"]).resolve()
rows=[json.loads(x) for x in (folder/"results.jsonl").read_text().splitlines()]
rng=random.Random(a.seed)
rng.shuffle(rows)
records=[];fingerprints={};used=set()
excluded={r["source_sha256"] for path in a.exclude_probes for r in json.loads(path.read_text())["rows"]}
started=time.monotonic()
for row in rows:
    if len(used)>=a.images:
        break
    path=(root/row["file"]).resolve()
    assert path.is_relative_to(root)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest in fingerprints.values() or digest in excluded:
        continue
    base=load_photo(str(path))[0]
    base.thumbnail((1024,1024),Image.Resampling.LANCZOS)
    known=row.get("repair_regions")
    if known is None:
        continue
    chosen=None
    for _ in range(128):
        width,height=rng.uniform(.28,.42),rng.uniform(.23,.34)
        left,top=rng.uniform(.015,1-width-.015),rng.uniform(.015,1-height-.015)
        box=(left,top,left+width,top+height)
        if any(max(box[0],r["box"][0])<min(box[2],r["box"][2]) and
               max(box[1],r["box"][1])<min(box[3],r["box"][3]) for r in known):
            continue
        chosen=box
        break
    if chosen is None:
        continue
    used.add(row["file"]);fingerprints[str(path)]=digest
    native=tuple(round(v*s) for v,s in zip(chosen,(base.width,base.height,base.width,base.height)))
    normalized=[round(1000*v/s) for v,s in zip(native,(base.width,base.height,base.width,base.height))]
    regions=parse_detection(json.dumps({"watermarks":[{"kind":"panel","box":normalized}]}))
    source=base.copy()
    draw=ImageDraw.Draw(source)
    draw.rectangle(native,fill=(220,220,220))
    draw.rectangle((native[0],native[1],native[2],native[1]+(native[3]-native[1])//3),fill=(30,100,220))
    kinds=("exact","jpeg","flat_fill") if a.cue=="uniform" else ("exact","jpeg","shadow_18","shadow_32","shadow_48")
    for kind in kinds:
        target=base.copy()
        if kind=="jpeg":
            stream=io.BytesIO();target.save(stream,format="JPEG",quality=90,subsampling=0)
            target=Image.open(io.BytesIO(stream.getvalue())).convert("RGB")
        elif kind=="flat_fill":
            ImageDraw.Draw(target).rectangle(native,fill=(128,128,128))
        elif kind.startswith("shadow_"):
            layer=Image.new("RGBA",base.size,(0,0,0,0))
            ImageDraw.Draw(layer).rectangle(native,fill=(0,0,0,int(kind.split("_")[1])))
            target=Image.alpha_composite(base.convert("RGBA"),layer).convert("RGB")
        core=[native[0]/base.width,native[1]/base.height,
              min(1,(native[2]+1)/base.width),min(1,(native[3]+1)/base.height)]
        if a.cue=="uniform":
            from watermark_studio.uniform_core import uniform_core_fill
            result=uniform_core_fill(target,regions,[{"mask":regions[0].box,"core":core}])
        else:
            result=(rectangle_phase_sharp(source,target,regions,[{"mask":regions[0].box,"core":core}]) if a.sharp_core else
                    rectangle_phase_inset(source,target,regions,[{"mask":regions[0].box,"core":core}]) if a.inset_core else
                    rectangle_phase_tight(source,target,regions,[{"mask":regions[0].box,"core":core}]) if a.tight_core else
                    rectangle_phase_core(source,target,regions,[{"mask":regions[0].box,"core":core}])
                    if a.core_boundaries else rectangle_phase(source,target,regions))
        records.append({"file":row["file"],"source_sha256":digest,"kind":kind,
                        "artifact_added":kind.startswith("shadow_") or kind=="flat_fill","cue":result})
assert len(used)==a.images
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in fingerprints.items())
by_kind={kind:{"total":sum(r["kind"]==kind for r in records),
               "warned":sum(r["kind"]==kind and bool(r["cue"]["count"]) for r in records)}
         for kind in sorted({r["kind"] for r in records})}
report={"code_revision":subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip(),"scope":"randomized new artifact-only controls, not complete watermark-removal truth",
        "cue":a.cue,"core_boundaries":a.core_boundaries,"tight_core":a.tight_core,"inset_core":a.inset_core,"sharp_core":a.sharp_core,
        "seed":a.seed,"backgrounds":len(used),"cases":len(records),
        "seconds":round(time.monotonic()-started,3),"source_inputs_unchanged":True,
        "by_kind":by_kind,"clean_warned":[r["file"]+":"+r["kind"] for r in records if not r["artifact_added"] and r["cue"]["count"]],
        "rows":records}
a.output.write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!="rows"}),flush=True)
