import numpy as np
from PIL import Image, ImageDraw

from watermark_studio.config import Region
from watermark_studio.images import composite_protected
from watermark_studio.quality import repair_boundary_warnings


def test_harmonic_blending_removes_colour_cast_without_using_overlay_pixels():
    yy, xx = np.indices((384, 512))
    clean_array = np.stack([110+xx%32, 130+yy%24, 145+(xx+yy)%16], axis=2).astype(np.uint8)
    clean = Image.fromarray(clean_array)
    original = clean.copy()
    region = Region(box=(0, .5, .4, 1))
    ImageDraw.Draw(original).rectangle((0,192,204,383),fill=(20,70,220))
    generated = Image.fromarray((clean_array.astype(np.int16)-[25,15,20]).astype(np.uint8))
    output = composite_protected(original,generated,[region],feather=0,blending="harmonic")
    actual = np.asarray(output).astype(np.float32)
    assert np.abs(actual[220:380,:180]-clean_array[220:380,:180]).mean() < 1.5
    assert np.array_equal(actual[:180],np.asarray(original)[:180])
    assert np.array_equal(actual[:,220:],np.asarray(original)[:,220:])


def test_seam_guard_rejects_dark_rectangle_and_accepts_harmonized_result():
    clean = Image.new("RGB",(512,384),(140,140,140))
    original = clean.copy()
    region = Region(box=(0,.45,.35,1))
    ImageDraw.Draw(original).rectangle((0,173,179,383),fill=(20,60,210))
    generated = Image.new("RGB",clean.size,(80,80,80))
    old = composite_protected(original,generated,[region],12,"feather")
    new = composite_protected(original,generated,[region],12,"harmonic")
    assert repair_boundary_warnings(old,[region])["count"] > 0
    assert repair_boundary_warnings(new,[region])["count"] == 0


def test_no_reference_does_not_use_watermark_as_colour_reference():
    from watermark_studio.images import harmonize_repair
    original = Image.new("RGB",(80,60),"blue")
    generated = Image.new("RGB",original.size,"green")
    actual = harmonize_repair(original,generated,[Region(box=(0,0,1,1))])
    assert actual.tobytes() == generated.tobytes()

def test_targeted_details_cover_tiny_code_instead_of_only_large_panel():
    from watermark_studio.quality import output_detail_views
    photo = Image.new("RGB",(4000,3000),"gray")
    panel = Region(box=(0,.7,.4,1))
    logo = Region(box=(.85,.97,1,1))
    views, boxes = output_detail_views(photo,photo,4,[panel,logo])
    assert len(views) == 4 and len(boxes) == 4
    assert boxes[0][0] <= .85 and boxes[0][2] == 1
    assert boxes[0][1] <= .97 and boxes[0][3] == 1
    assert views[0].width > views[0].height * 3
    assert max(views[0].size) <= 512
