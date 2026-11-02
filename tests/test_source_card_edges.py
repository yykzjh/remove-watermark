import hashlib

import numpy as np
from PIL import Image, ImageDraw

from watermark_studio.config import Region
from watermark_studio.source_card_edges import propose_card_side_rims
from watermark_studio.source_panel_rims import propose_blue_card_rims, restore_proven_rims


def fixture():
    source=Image.new("RGB",(1000,800),(80,110,60))
    draw=ImageDraw.Draw(source)
    draw.rectangle((30,480,450,539),fill=(30,110,220))
    draw.rectangle((30,540,450,739),fill=(220,220,220))
    region=Region(kind="panel",box=(0,.54,.51,.995))
    core={"mask":list(region.box),"core":[.022,.592,.459,.933]}
    anchor=propose_blue_card_rims(source,[region],[core])[0]
    yes={"digital_card":True,"rim_clear":True,"uncertain":False,"area":"none"}
    anchor.update(approved=True,editor=yes,independent=yes,weights_relation="different_weights",
        ocr={"status":"clear"},source_rim_sha256=hashlib.sha256(
            source.crop(tuple(anchor["restore_pixels"])).tobytes()).hexdigest())
    return source,region,anchor


def test_side_proposals_follow_visible_card_edges_but_grant_no_restoration():
    source,region,anchor=fixture()
    props=propose_card_side_rims(source,[region],[anchor])
    assert {p["side"] for p in props}=={"left","right","bottom"}
    for p in props:
        expected={"left":30,"right":451,"bottom":740}[p["side"]]
        assert abs(p["card_edge_pixel"]-expected)<=2
        l,t,r,b=p["restore_pixels"]
        # Every proposed original pixel is outside the true synthetic card.
        assert not (max(l,30)<min(r,451) and max(t,480)<min(b,740))
        assert not p["approved"] and "editor" not in p
    candidate=Image.new("RGB",source.size,"black")
    result,count=restore_proven_rims(source,candidate,props)
    assert count==0 and result.tobytes()==candidate.tobytes()


def test_other_overlay_blocks_a_candidate_clear_band():
    source,region,anchor=fixture()
    other=Region(kind="logo",box=(0,.8,.025,.98))
    props=propose_card_side_rims(source,[region,other],[anchor])
    assert "left" not in {p["side"] for p in props}
    assert "right" in {p["side"] for p in props}


def test_missing_stale_or_nonblue_anchor_never_supplies_side_proposals():
    source,region,anchor=fixture()
    assert not propose_card_side_rims(source,[region],[])
    assert not propose_card_side_rims(source,[region],[anchor|{"source_rim_sha256":"bad"}])
    changed=source.copy();ImageDraw.Draw(changed).rectangle((30,480,450,539),fill=(140,140,140))
    assert not propose_card_side_rims(changed,[region],[anchor])
    assert np.asarray(source)[500,100,2]==220


def test_body_edge_guard_does_not_infer_gray_width_from_blue_header():
    source,region,anchor=fixture()
    # The gray body extends beyond the blue header, inside the search budget.
    ImageDraw.Draw(source).rectangle((451,540,464,739),fill=(220,220,220))
    props=propose_card_side_rims(source,[region],[anchor],verify_body_sides=True)
    right=next(p for p in props if p["side"]=="right")
    assert abs(right["card_edge_pixel"]-465)<=2
    assert right["restore_pixels"][0]>=471
    assert right["source_geometry"]["body_sides"]["right"]["edge_phase_support"]>=.75
    assert right["approved"] is False


def test_missing_body_edge_withholds_side_proposals_instead_of_trusting_header():
    source,region,anchor=fixture()
    # Wider than the bounded search: fail closed, do not keep the header-only edge.
    ImageDraw.Draw(source).rectangle((451,540,520,739),fill=(220,220,220))
    props=propose_card_side_rims(source,[region],[anchor],verify_body_sides=True)
    assert not any(p["side"]=="right" for p in props)
