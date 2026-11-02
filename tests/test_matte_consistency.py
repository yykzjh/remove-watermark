import numpy as np
from PIL import Image

from watermark_studio.config import Region
from watermark_studio.matte_consistency import matte_consistency


def photos(legitimate_shadow=False,artifact_shadow=False):
    rng=np.random.default_rng(11)
    clean=rng.integers(60,190,size=(256,320,3)).astype(np.float64)
    mask=np.zeros((256,320),dtype=bool);mask[40:240,32:288]=True
    if legitimate_shadow:clean[mask]-=18
    before=clean.copy();before[mask]=.75*clean[mask]+.25*230
    after=clean.copy()
    if artifact_shadow:after[mask]-=18
    return [Image.fromarray(np.rint(photo).astype(np.uint8)) for photo in (before,after)]


def test_correlated_faint_shadow_has_impossible_matte_but_existing_shadow_does_not():
    regions=[Region(kind="panel",box=(.1,40/256,.9,240/256))]
    clean=matte_consistency(*photos(),regions)
    artifact=matte_consistency(*photos(artifact_shadow=True),regions)
    physical=matte_consistency(*photos(legitimate_shadow=True),regions)
    assert clean["observable_tiles"]>0 and clean["count"]==0
    assert artifact["count"]==1 and artifact["quality_permission"] is False
    assert physical["observable_tiles"]>0 and physical["count"]==0


def test_unchanged_overlay_and_uncorrelated_scene_cannot_be_approved_by_matte():
    before,after=photos();regions=[Region(kind="panel",box=(.1,40/256,.9,240/256))]
    unchanged=matte_consistency(before,before,regions)
    shuffled=np.asarray(after).copy();np.random.default_rng(1).shuffle(shuffled,axis=0)
    unrelated=matte_consistency(before,Image.fromarray(shuffled),regions)
    assert unchanged["count"]==0 and unchanged["observable_tiles"]==0
    assert unrelated["count"]==0
    assert unchanged["quality_permission"] is False and unrelated["quality_permission"] is False
