"""CPU ablation on controlled known masks; not an end-to-end semantic score."""
import argparse
import collections
import json
from pathlib import Path

from watermark_studio.detection import parse_detection
from watermark_studio.images import load_photo
from watermark_studio.quality import repair_boundary_warnings, repair_texture_warnings

parser=argparse.ArgumentParser()
parser.add_argument("--tag",required=True)
args=parser.parse_args()
root=Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031")/("adversarial-"+args.tag)
records=json.loads((root/"fixture.json").read_text())
results=[]
for record in records:
    before=load_photo(root/"input"/record["file"])[0]
    after=load_photo(root/"output"/record["file"])[0]
    index=int(record["file"].split("-")[0])
    x,y=((.02,.66),(.60,.03),(.30,.38),(.03,.03),(.60,.67))[index%5]
    regions=[] if record["kind"]=="identity" else parse_detection(json.dumps({"watermarks":[{
        "kind":"panel","box":[round(x*1000),round(y*1000),round((x+.37)*1000),round(min(1,y+.25)*1000)]}]}))
    seam=repair_boundary_warnings(after,regions,before=before,inset_search=True)
    texture=repair_texture_warnings(after,regions,before=before,inset_search=True)
    results.append(record|{"seam":seam["count"],"texture":texture["count"]})
summary={kind:{"samples":sum(r["kind"]==kind for r in results),
               "seam":sum(bool(r["seam"]) for r in results if r["kind"]==kind),
               "texture":sum(bool(r["texture"]) for r in results if r["kind"]==kind)}
         for kind in sorted({r["kind"] for r in results})}
(root/"numeric-candidate.json").write_text(json.dumps(results,indent=2))
print(json.dumps(summary),flush=True)
