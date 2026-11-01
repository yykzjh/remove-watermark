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


@pytest.mark.parametrize("shade,expected_sign",[(120,-1),(160,1)])
def test_joint_search_checks_both_tone_directions_once_per_panel(shade,expected_sign):
    _clean,source,_dark,regions=fixture((.2,.2,.7,.8))
    result=Image.new("RGB",source.size,(140,140,140))
    ImageDraw.Draw(result).rectangle((160,120,560,480),fill=(shade,shade,shade))
    cue=rectangle_phase(source,result,regions)
    assert cue["count"]==1
    assert cue["findings"][0]["joint_sign"]==expected_sign
    assert cue["method"]=="source_panel_phase_v2_joint"


def test_core_search_only_adds_vetoes_with_valid_source_provenance():
    from watermark_studio.rectangle_phase import rectangle_phase_core
    clean,source,dark,regions=fixture((.2,.2,.7,.8))
    core={"mask":list(regions[0].box),"core":[.2,.2,.7,.8]}
    base=rectangle_phase(source,dark,regions)
    extended=rectangle_phase_core(source,dark,regions,[core])
    assert {tuple(v["box"]) for v in base["findings"]} <= {tuple(v["box"]) for v in extended["findings"]}
    assert rectangle_phase_core(source,clean,regions,[core])["count"]==0
    assert rectangle_phase_core(source,dark,regions,[{"mask":core["mask"],"core":[0,0,1,1]}])["findings"]==base["findings"]
    assert rectangle_phase_core(source,dark,regions,[{"core":[float("nan"),0,1,1]}])["findings"]==base["findings"]


def test_sign_searches_share_immutable_pixel_preparation_per_scale(monkeypatch):
    import watermark_studio.rectangle_phase as phase
    _clean,source,dark,regions=fixture((.2,.2,.7,.8))
    original=phase._prepare_phase
    calls=[]
    def prepare(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(phase,"_prepare_phase",prepare)
    core={"mask":list(regions[0].box),"core":[.2,.2,.7,.8]}
    assert phase.rectangle_phase_core(source,dark,regions,[core])["count"]==1
    assert len(calls)==2



@pytest.mark.parametrize("box",[(.2,.2,.7,.8),(0,.55,.4,1)])
def test_tight_core_never_discards_old_vetoes_or_changes_original_pixels(box):
    from watermark_studio.rectangle_phase import rectangle_phase_core, rectangle_phase_tight
    clean,source,dark,regions=fixture(box)
    hashes=source.tobytes(),dark.tobytes()
    core={"mask":list(regions[0].box),"core":list(box)}
    original=rectangle_phase_core(source,dark,regions,[core])
    extra=rectangle_phase_tight(source,dark,regions,[core])
    assert {tuple(v["box"]) for v in original["findings"]} <= {tuple(v["box"]) for v in extra["findings"]}
    assert rectangle_phase_tight(source,clean,regions,[core])["count"]==0
    assert rectangle_phase_tight(dark,dark.copy(),regions,[core])["count"]==0
    assert (source.tobytes(),dark.tobytes())==hashes
    json.dumps(extra,allow_nan=False)


def test_tight_search_does_not_use_unknown_or_outside_core_coordinates():
    from watermark_studio.rectangle_phase import rectangle_phase_core, rectangle_phase_tight
    _,source,dark,regions=fixture((.2,.2,.7,.8))
    for bad in ([{"mask":list(regions[0].box),"core":[0,0,1,1]}],[{"core":[float("nan"),0,1,1]}],[]):
        assert rectangle_phase_tight(source,dark,regions,bad)["findings"]==rectangle_phase_core(source,dark,regions,bad)["findings"]


def test_tight_and_broad_core_reuse_the_same_fixed_blur_arrays(monkeypatch):
    import watermark_studio.rectangle_phase as phase
    _,source,dark,regions=fixture((.2,.2,.7,.8))
    core={"mask":list(regions[0].box),"core":[.2,.2,.7,.8]}
    original=phase._prepare_phase;calls=[]
    def prepare(*args,**kwargs):
        calls.append(kwargs.get("blur_radius"))
        return original(*args,**kwargs)
    monkeypatch.setattr(phase,"_prepare_phase",prepare)
    result=phase.rectangle_phase_tight(source,dark,regions,[core])
    assert result["count"]==1 and len(calls)==2 and calls.count(3.)==1


def test_offline_fixed_padding_hypothesis_preserves_every_v5_veto_and_original_pixel():
    from watermark_studio.rectangle_phase import rectangle_phase_inset, rectangle_phase_tight
    clean,source,dark,regions=fixture((.2,.2,.7,.8))
    cores=[{"mask":list(regions[0].box),"core":[.2,.2,.7,.8]}]
    before=source.tobytes(),dark.tobytes()
    old=rectangle_phase_tight(source,dark,regions,cores)
    new=rectangle_phase_inset(source,dark,regions,cores)
    assert all(f in new["findings"] for f in old["findings"])
    assert rectangle_phase_inset(source,clean,regions,cores)["count"]==0
    assert rectangle_phase_inset(dark,dark.copy(),regions,cores)["count"]==0
    assert before==(source.tobytes(),dark.tobytes())
