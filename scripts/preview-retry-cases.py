"""Few native diagnostic crops for failed retries; no source/output mutation."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from watermark_studio.images import load_photo

p=argparse.ArgumentParser()
p.add_argument("--job",required=True)
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
assert not a.output.exists()
state=Path.home()/".local/state/watermark-studio"/a.job
s=json.loads((state/"summary.json").read_text())
rows=[r for r in map(json.loads,(state/"results.jsonl").read_text().splitlines())
      if r.get("quality_attempts",1)>1 and r["quality"]["status"]!="passed"]
assert 1<=len(rows)<=4
canvas=Image.new("RGB",(1280,440*len(rows)),"#202020");draw=ImageDraw.Draw(canvas);audit=[]
for i,r in enumerate(rows):
    before=load_photo(str(Path(s["input_dir"])/r["file"]))[0]
    first_path=Path(r["first_rejected_candidate"])
    assert hashlib.sha256(first_path.read_bytes()).hexdigest()==r["first_rejected_sha256"]
    first=load_photo(str(first_path))[0]
    q=r["attempt_history"][0]["quality"];box=q["repair_texture"]["suspect_segments"][0]["box"]
    x=(box[0]+box[2])*before.width/2;y=(box[1]+box[3])*before.height/2
    edge=min(600,before.width,before.height)
    l=max(0,min(before.width-edge,round(x-edge/2)));t=max(0,min(before.height-edge,round(y-edge/2)))
    crop=(l,t,l+edge,t+edge);top=i*440
    draw.text((8,top+4),f"{i}: source | first candidate | source native | first native ; best attempt={r.get('best_attempt')}",fill="white")
    for j,photo in enumerate([before,first]):
        thumb=ImageOps.contain(photo,(240,400));canvas.paste(thumb,(j*240,top+28))
        detail=ImageOps.contain(photo.crop(crop),(400,400));canvas.paste(detail,(480+j*400,top+28))
    audit.append({"file":r["file"],"crop":crop,"cue":box,"best_attempt":r.get("best_attempt")})
canvas.save(a.output,quality=72,optimize=True)
a.output.with_suffix(".json").write_text(json.dumps(audit,ensure_ascii=False,indent=2))
print(json.dumps({"bytes":a.output.stat().st_size,"cases":len(rows)}))
