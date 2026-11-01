"""New randomized artifact-only controls; original dataset files stay immutable."""
import argparse
import hashlib
import io
import json
import random
import time
from pathlib import Path

from PIL import Image, ImageDraw

from watermark_studio.detection import parse_detection
from watermark_studio.images import load_photo
from watermark_studio.rectangle_phase import rectangle_phase, rectangle_phase_core

p=argparse.ArgumentParser()
p.add_argument("--source-job",required=True)
p.add_argument("--images",type=int,default=64)
p.add_argument("--seed",type=int,default=202611022)
p.add_argument("--core-boundaries",action="store_true")
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
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
started=time.monotonic()
for row in rows:
    if len(used)>=a.images:
        break
    path=(root/row["file"]).resolve()
    assert path.is_relative_to(root)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest in fingerprints.values():
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
    for kind in ("exact","jpeg","shadow_18","shadow_32","shadow_48"):
        target=base.copy()
        if kind=="jpeg":
            stream=io.BytesIO();target.save(stream,format="JPEG",quality=90,subsampling=0)
            target=Image.open(io.BytesIO(stream.getvalue())).convert("RGB")
        elif kind.startswith("shadow_"):
            layer=Image.new("RGBA",base.size,(0,0,0,0))
            ImageDraw.Draw(layer).rectangle(native,fill=(0,0,0,int(kind.split("_")[1])))
            target=Image.alpha_composite(base.convert("RGBA"),layer).convert("RGB")
        core=[native[0]/base.width,native[1]/base.height,
              min(1,(native[2]+1)/base.width),min(1,(native[3]+1)/base.height)]
        result=(rectangle_phase_core(source,target,regions,[{"mask":regions[0].box,"core":core}])
                if a.core_boundaries else rectangle_phase(source,target,regions))
        records.append({"file":row["file"],"source_sha256":digest,"kind":kind,
                        "artifact_added":kind.startswith("shadow_"),"cue":result})
assert len(used)==a.images
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in fingerprints.items())
by_kind={kind:{"total":sum(r["kind"]==kind for r in records),
               "warned":sum(r["kind"]==kind and bool(r["cue"]["count"]) for r in records)}
         for kind in sorted({r["kind"] for r in records})}
report={"scope":"randomized new artifact-only controls, not complete watermark-removal truth",
        "core_boundaries":a.core_boundaries,
        "seed":a.seed,"backgrounds":len(used),"cases":len(records),
        "seconds":round(time.monotonic()-started,3),"source_inputs_unchanged":True,
        "by_kind":by_kind,"clean_warned":[r["file"]+":"+r["kind"] for r in records if not r["artifact_added"] and r["cue"]["count"]],
        "rows":records}
a.output.write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!="rows"}),flush=True)
