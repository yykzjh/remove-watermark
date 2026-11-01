import numpy as np
from PIL import Image, ImageDraw

from watermark_studio.boundary_views import boundary_ribbons
from watermark_studio.config import Region


def test_ribbons_do_not_invent_exterior_for_clipped_panel_or_mutate_output():
    image=Image.new("RGB",(800,600),"green")
    ImageDraw.Draw(image).rectangle((0,350,400,599),fill=(20,80,20))
    original=image.tobytes()
    canvas,boxes=boundary_ribbons(image,[Region(box=(0,.58,.5,1))])
    assert {x["side"] for x in boxes}=={"top","right"}
    assert canvas.width*canvas.height<=640*384
    assert image.tobytes()==original
    assert all(0<=v<=1 for x in boxes for v in x["box"])


def test_no_diagnostic_without_large_known_source_overlay():
    assert boundary_ribbons(Image.new("RGB",(200,200)),[])==(None,[])
    assert boundary_ribbons(Image.new("RGB",(200,200)),[Region(box=(.8,.9,1,1))])==(None,[])


def test_original_output_boundary_remains_visible_in_normal_column():
    image=Image.new("RGB",(800,600),(140,140,140))
    ImageDraw.Draw(image).rectangle((160,120,640,480),fill=(120,120,120))
    canvas,boxes=boundary_ribbons(image,[Region(box=(.2,.2,.8,.8))])
    assert len(boxes)==4
    colours={tuple(x) for x in np.asarray(canvas.crop((0,0,320,96))).reshape(-1,3)}
    assert (140,140,140) in colours and (120,120,120) in colours
