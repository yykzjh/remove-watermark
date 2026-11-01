"""Source-only resolution risk features; never alter edits, masks or QA decisions."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .surface_reference import source_rims


def source_resolution_features(photo, regions, cores, *, scene, base_resolution=1024):
    """Measure visible source margins, excluding every unknown overlay core."""
    panel_areas = [(r.box[2]-r.box[0])*(r.box[3]-r.box[1])
                   for r in regions if r.kind == "panel"]
    area = max(panel_areas, default=0.)
    density = math.sqrt(photo.width*photo.height)/base_resolution
    result = {"method": "visible_source_detail_v1", "eligible": False,
              "panel_area": round(area, 6), "native_density": round(density, 6),
              "tiles": 0, "detail75": None, "score": None}
    if scene != "outdoor" or not .035 <= area < .35:
        return result | {"reason": "not_regular_outdoor_panel"}
    image = photo.convert("RGB")
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    bands = source_rims(np.asarray(image, dtype=np.float32), regions, cores)
    tiles = [tile for index, sides in bands.items() if regions[index].kind == "panel"
             for side in sides for tile in side["tiles"] if tile is not None]
    if not tiles:
        return result | {"reason": "no_visible_source_rims"}
    detail = float(np.percentile([tile["detail"] for tile in tiles], 75))
    score = max(1., density)*min(detail, 20.)*math.sqrt(area)
    result.update(eligible=density > 1.25, tiles=len(tiles), detail75=round(detail, 6),
                  score=round(score, 6), reason="source_only_candidate_risk")
    return result
