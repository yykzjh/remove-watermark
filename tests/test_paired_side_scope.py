import hashlib

import pytest
from PIL import Image, ImageDraw

from watermark_studio.config import Region
from watermark_studio.source_card_edges import propose_card_side_rims
from watermark_studio.source_panel_rims import propose_blue_card_rims
from watermark_studio.source_side_proof import paired_band_scope


def fixture():
    source=Image.new("RGB",(1000,800),(80,110,60));draw=ImageDraw.Draw(source)
    draw.rectangle((30,480,450,539),fill=(30,110,220))
    draw.rectangle((30,540,464,739),fill=(220,220,220))
    region=Region(kind="panel",box=(0,.54,.51,.995))
    core={"mask":list(region.box),"core":[.022,.592,.459,.933]}
    anchor=propose_blue_card_rims(source,[region],[core])[0]
    yes={"digital_card":True,"rim_clear":True,"uncertain":False,"area":"none"}
    anchor.update(approved=True,editor=yes,independent=yes,weights_relation="different_weights",
        ocr={"status":"clear"},source_rgb_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
        source_rim_sha256=hashlib.sha256(source.crop(tuple(anchor["restore_pixels"])).tobytes()).hexdigest())
    return source,region,anchor


def test_paired_scope_accepts_only_exterior_whole_band_or_native_subset():
    source,region,anchor=fixture()
    props=propose_card_side_rims(source,[region],[anchor],verify_body_sides=True)
    right=next(p for p in props if p["side"]=="right")
    assert paired_band_scope(source,right,anchor)["quality_permission"] is False
    sub=right|{"restore_pixels":[right["restore_pixels"][0]+1,600,right["restore_pixels"][2]-1,700]}
    assert paired_band_scope(source,sub,anchor)["candidate_pixels"]==sub["restore_pixels"]
    # A gray-body strip beyond the narrower blue header is still covered.
    bad=right|{"restore_pixels":[452,560,464,700]}
    with pytest.raises(ValueError,match="exterior band"):
        paired_band_scope(source,bad,anchor)


def test_paired_scope_rejects_stale_rgb_changed_identity_or_missing_body_evidence():
    source,region,anchor=fixture()
    prop=propose_card_side_rims(source,[region],[anchor],verify_body_sides=True)[0]
    changed=source.copy();changed.putpixel((900,10),(1,2,3))
    with pytest.raises(ValueError,match="complete source-card"):
        paired_band_scope(changed,prop,anchor)
    with pytest.raises(ValueError,match="cannot change"):
        paired_band_scope(source,prop|{"core":[.1,.6,.5,.9]},anchor)
    # A strong header and one body edge do not establish the full matte extent.
    ImageDraw.Draw(source).rectangle((0,540,29,739),fill=(220,220,220))
    anchor["source_rgb_sha256"]=hashlib.sha256(source.tobytes()).hexdigest()
    with pytest.raises(ValueError,match="exterior band"):
        paired_band_scope(source,prop,anchor)
