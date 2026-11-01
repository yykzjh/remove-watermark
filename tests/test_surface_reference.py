import numpy as np
from PIL import Image, ImageDraw

from watermark_studio.config import Region
from watermark_studio.quality import repair_texture_warnings
from watermark_studio.surface_reference import matching_smooth_run, source_rims


def scene(smooth=True):
    rng=np.random.default_rng(417)
    base=np.array([109,142,176],dtype=float)
    values=np.clip(base+rng.normal(0,20,(800,1000,1)),0,255).astype(np.uint8)
    clean=Image.fromarray(values)
    if smooth:
        ImageDraw.Draw(clean).rectangle((0,480,999,799),fill=tuple(base.astype(int)))
    source=clean.copy()
    mask=Region(kind="panel",box=(.02,.58,.48,.98))
    core=(.05,.62,.45,.95)
    ImageDraw.Draw(source).rectangle((50,496,450,760),fill=(30,100,220))
    after=clean.copy()
    ImageDraw.Draw(after).rectangle((20,480,479,783),fill=(114,147,181))
    return source,after,[mask],[{"mask":list(mask.box),"core":list(core)}]


def test_original_smooth_surface_is_not_compared_only_to_adjacent_rough_material():
    before,after,regions,cores=scene()
    hashes=(before.tobytes(),after.tobytes())
    assert repair_texture_warnings(after,regions,before=before)["count"]>0
    corrected=repair_texture_warnings(after,regions,before=before,core_boxes=cores)
    assert corrected["count"]==0
    proof=corrected["source_reference_support"]
    assert proof and all(p["visible_source_pixels"]>=96 for p in proof)
    assert all("reference_side" in p and "side" in p for p in proof)
    assert hashes==(before.tobytes(),after.tobytes())


def test_actual_flattening_of_original_textured_surface_keeps_texture_veto():
    before,after,regions,cores=scene(smooth=False)
    assert repair_texture_warnings(after,regions,before=before)["count"]>0
    assert repair_texture_warnings(after,regions,before=before,core_boxes=cores)["count"]>0


def test_unknown_or_invalid_source_core_never_supplies_a_reference():
    before,after,regions,_cores=scene()
    for cores in ([],[{"mask":regions[0].box,"core":[0,0,1,1]}],
                  [{"mask":[0,0,.1,.1],"core":[0,0,.05,.05]}]):
        result=repair_texture_warnings(after,regions,before=before,core_boxes=cores)
        assert result["count"]>0 and not result["source_reference_support"]


def test_wrong_repair_colour_cannot_borrow_an_unrelated_smooth_reference():
    before,after,regions,cores=scene()
    ImageDraw.Draw(after).rectangle((20,480,479,783),fill=(135,140,145))
    assert repair_texture_warnings(after,regions,before=before,core_boxes=cores)["count"]>0


def test_reference_run_needs_three_contiguous_visible_source_tiles():
    tile={"box":[0,0,.1,.1],"detail":.1,"spread":1,"rgb":[110,140,170]}
    assert matching_smooth_run([{"side":"left","tiles":[tile,tile,None,tile,tile]}],
                               np.array([110,140,170])) is None
    assert matching_smooth_run([{"side":"left","tiles":[tile,tile,tile]}],
                               np.array([110,140,170])) is not None


def test_rim_reference_pixels_must_not_overlap_any_other_overlay_core():
    rgb=np.full((800,1000,3),140,dtype=np.float32)
    region=Region(kind="panel",box=(.1,.1,.8,.9))
    core={"mask":list(region.box),"core":[.14,.14,.76,.86]}
    other=Region(kind="logo",box=(.12,.14,.14,.86))
    refs=source_rims(rgb,[region,other],[core])
    left=next(side for side in refs[0] if side["side"]=="left")
    assert left["tiles"] and all(tile is None for tile in left["tiles"])


def test_tight_scope_only_uses_explicit_clear_band_outside_mask():
    before,_after,_regions,cores=scene()
    tight=Region(kind="panel",box=(.02,.62,.48,.98))
    tight_cores=[{"mask":list(tight.box),"core":cores[0]["core"]}]
    rgb=np.asarray(before,dtype=np.float32)
    top=lambda refs:next((s for s in refs[0] if s["side"]=="top"),None)
    assert top(source_rims(rgb,[tight],tight_cores)) is None
    clear=(.02,.58,.48,.62)
    refs=source_rims(rgb,[tight],tight_cores,verified_clear_boxes=[clear])
    support=matching_smooth_run([top(refs)],np.array([109,142,176]))
    assert support["visible_source_pixels"]==96
    assert support["reference_scopes"]==["verified_clear_source_band"]
    # A short, invalid, or noncontaining strip cannot establish 96 clear pixels.
    for band in ((.05,.60,.10,.62),[0,0,1,2],(.02,.619,.48,.62)):
        refs=source_rims(rgb,[tight],tight_cores,verified_clear_boxes=[band])
        assert matching_smooth_run([top(refs)] if top(refs) else [],np.array([109,142,176])) is None


def test_clear_band_cannot_override_other_unknown_overlay_or_colour():
    before,_after,_regions,cores=scene()
    tight=Region(kind="panel",box=(.02,.62,.48,.98))
    other=Region(kind="logo",box=(.02,.60,.48,.619))
    tight_cores=[{"mask":list(tight.box),"core":cores[0]["core"]}]
    refs=source_rims(np.asarray(before,dtype=np.float32),[tight,other],tight_cores,
                     verified_clear_boxes=[(.02,.58,.48,.62)])
    top=next(s for s in refs[0] if s["side"]=="top")
    assert all(t is None for t in top["tiles"])
    refs=source_rims(np.asarray(before,dtype=np.float32),[tight],tight_cores,
                     verified_clear_boxes=[(.02,.58,.48,.62)])
    assert matching_smooth_run(refs[0],np.array([180,30,30])) is None
