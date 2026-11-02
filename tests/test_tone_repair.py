import numpy as np
from PIL import Image

from watermark_studio.config import Region
from watermark_studio.tone_repair import repair_tone_steps


def test_tone_repair_removes_injected_rectangle_without_changing_outside_pixels():
    y,x=np.mgrid[:384,:512]
    clean=130+.04*x+.025*y+np.random.default_rng(5).normal(0,2,(384,512))
    rgb=np.clip(np.rint(np.repeat(clean[...,None],3,axis=2)),0,255).astype(np.uint8)
    box=(96,80,416,320);l,t,r,b=box
    source=rgb.copy();source[t:b,l:r]=np.rint(.5*rgb[t:b,l:r]+.5*235).astype(np.uint8)
    damaged=rgb.copy();damaged[t:b,l:r]-=18
    region=Region(kind="panel",box=(l/512,t/384,r/512,b/384))
    fixed,audit=repair_tone_steps(Image.fromarray(source),Image.fromarray(damaged),[region],
        [{"mask":region.box,"core":region.box}])
    assert audit["applied"] and audit["quality_permission"] is False
    actual=np.asarray(fixed);inside=np.zeros((384,512),bool);inside[t:b,l:r]=True
    assert np.array_equal(actual[~inside],damaged[~inside])
    assert np.abs(actual[inside].astype(float)-rgb[inside]).mean()<1.
    assert audit["final_phase_count"]<audit["initial_phase_count"]


def test_tone_repair_does_not_modify_known_clean_or_no_edit_frames():
    rng=np.random.default_rng(77);rgb=rng.integers(60,210,(384,512,3),dtype=np.uint8)
    region=Region(kind="panel",box=(.18,.2,.82,.84))
    photo=Image.fromarray(rgb)
    for regions in [[],[region]]:
        fixed,audit=repair_tone_steps(photo,photo,regions)
        assert not audit["applied"] and fixed.tobytes()==photo.tobytes()


def test_real_existing_shadow_crossing_digital_panel_is_preserved():
    y,x=np.mgrid[:384,:512]
    base=140+.03*x+.02*y+np.random.default_rng(71).normal(0,2,(384,512))
    clean=np.rint(np.repeat(base[...,None],3,axis=2)).astype(np.uint8)
    clean[40:350,50:465]-=18
    source=clean.copy();source[80:320,96:416]=np.rint(.5*clean[80:320,96:416]+.5*235).astype(np.uint8)
    region=Region(kind="panel",box=(96/512,80/384,416/512,320/384))
    fixed,audit=repair_tone_steps(Image.fromarray(source),Image.fromarray(clean),[region],
        [{"mask":region.box,"core":region.box}])
    assert not audit["applied"] and fixed.tobytes()==Image.fromarray(clean).tobytes()
