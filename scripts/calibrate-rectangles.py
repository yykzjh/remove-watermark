"""Known-target CPU calibration; reports candidates, never invents semantic accuracy."""
import json
from pathlib import Path

from watermark_studio.config import Region
from watermark_studio.images import load_photo
from watermark_studio.repair_checks import rectangular_repairs

root=Path("/mnt/sdb/zjh/baidu-download/1mQf6F6g-20261031/adversarial-duel-controls-v5")
metadata=json.loads((root/"result.json").read_text())
rows={r["file"]:r for r in map(json.loads,(Path.home()/".local/state/watermark-studio"/metadata["job"]/"results.jsonl").read_text().splitlines())}
records=json.loads((root/"fixture.json").read_text())
for record in records:
    before=load_photo(root/"input"/record["file"])[0];after=load_photo(root/"output"/record["file"])[0]
    regions=[Region.model_validate(r) for r in rows[record["file"]].get("quality",{}).get("source_regions",[])]
    result=rectangular_repairs(before,after,regions)
    record["rectangle"]=result
(root/"rectangle-calibration.json").write_text(json.dumps(records,indent=2))
print(json.dumps({kind:{"total":sum(r["kind"]==kind for r in records),
    "flagged":sum(r["kind"]==kind and r["rectangle"]["count"]>0 for r in records)}
    for kind in sorted({r["kind"] for r in records})}),flush=True)
