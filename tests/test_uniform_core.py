import numpy as np
from PIL import Image, ImageDraw

from watermark_studio.config import Region
from watermark_studio.uniform_core import uniform_core_fill


def test_new_neutral_paint_block_is_found_without_treating_flat_scene_as_a_defect():
    rng=np.random.default_rng(2)
    base=Image.fromarray(rng.integers(40,210,(600,800,3),dtype=np.uint8))
    region=Region(kind="panel",box=(.17,.17,.73,.83))
    core=[.2,.2,.7,.8]
    provenance=[{"mask":list(region.box),"core":core}]
    paint=base.copy()
    ImageDraw.Draw(paint).rectangle((160,120,559,479),fill=(128,128,128))
    before=paint.tobytes()
    assert uniform_core_fill(paint,[region],provenance)["count"]==1
    assert uniform_core_fill(base,[region],provenance)["count"]==0
    assert uniform_core_fill(Image.new("RGB",base.size,(128,128,128)),[region],provenance)["count"]==0
    assert paint.tobytes()==before
    for colour in [(255,255,255),(0,0,0),(20,60,180)]:
        other=base.copy();ImageDraw.Draw(other).rectangle((160,120,559,479),fill=colour)
        assert uniform_core_fill(other,[region],provenance)["count"]==0
    assert uniform_core_fill(paint,[region],[{"mask":core,"core":core}])["count"]==0
