import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from watermark_studio.config import Region
from watermark_studio.detection import parse_detection
from watermark_studio.rectangle_phase import rectangle_phase


def fixture(box):
    clean=Image.new("RGB",(800,600),(140,140,140))
    native=tuple(round(v*s) for v,s in zip(box,(800,600,800,600)))
    source=clean.copy()
    ImageDraw.Draw(source).rectangle(native,fill=(30,100,220))
    regions=parse_detection(json.dumps({"watermarks":[{"kind":"panel","box":[round(v*1000) for v in box]}]}))
    dark=clean.copy()
    ImageDraw.Draw(dark).rectangle(native,fill=(120,120,120))
    return clean,source,dark,regions


@pytest.mark.parametrize("box",[(.2,.2,.7,.8),(0,.55,.4,1)])
def test_coherent_rectangle_is_flagged_but_clean_and_original_shadows_are_retained(box):
    clean,source,dark,regions=fixture(box)
    before,after=source.tobytes(),dark.tobytes()
    result=rectangle_phase(source,dark,regions)
    assert result["count"]==1
    assert rectangle_phase(source,clean,regions)["count"]==0
    assert rectangle_phase(dark,dark.copy(),regions)["count"]==0
    assert before==source.tobytes() and after==dark.tobytes()
    json.dumps(result,allow_nan=False)


def test_natural_gradient_and_single_real_material_edge_do_not_make_a_closed_rectangle():
    yy,xx=np.mgrid[:600,:800]
    value=(90+xx*.06+yy*.04).astype(np.uint8)
    clean=Image.fromarray(np.repeat(value[...,None],3,axis=2))
    source=clean.copy()
    ImageDraw.Draw(source).rectangle((160,120,560,480),fill=(30,100,220))
    regions=parse_detection('{"watermarks":[{"kind":"panel","box":[200,200,700,800]}]}')
    assert rectangle_phase(source,clean,regions)["count"]==0


def test_contaminated_exterior_and_non_panel_masks_do_not_supply_evidence():
    _clean,source,dark,regions=fixture((.2,.2,.7,.8))
    neighbour=Region(kind="logo",box=(.75,.25,.78,.45))
    assert rectangle_phase(source,dark,[*regions,neighbour])["count"]==0
    assert rectangle_phase(source,dark,[Region(kind="logo",box=(.2,.2,.7,.8))])["count"]==0
