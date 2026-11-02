import random

import numpy as np
import pytest
from PIL import Image, ImageFont

from watermark_studio.config import Region
from watermark_studio.training_pairs import clean_crop, render_pair


def test_clean_crop_avoids_all_source_hypotheses_and_safety_margin():
    regions=[Region(kind="panel",box=(0,.65,.45,1)),Region(kind="logo",box=(.8,.9,1,1))]
    crop=clean_crop((4000,3000),regions,aspect=3/4)
    assert crop is not None
    l,t,r,b=crop
    for region in regions:
        x0,y0,x1,y1=region.box
        assert not (max(l,max(0,(x0-.04)*4000))<min(r,min(4000,(x1+.04)*4000)) and
            max(t,max(0,(y0-.04)*3000))<min(b,min(3000,(y1+.04)*3000)))
    assert abs((r-l)/(b-t)-3/4)<.001
    assert clean_crop((1000,800),[Region(kind="panel",box=(0,0,1,1))]) is None


def test_synthetic_mask_contains_every_changed_pixel_and_identity_is_exact(monkeypatch):
    fallback=ImageFont.load_default()
    monkeypatch.setattr("watermark_studio.training_pairs.ImageFont.truetype",lambda *a,**k:fallback)
    # The portable test font may lack CJK; draw geometry still proves the mask contract.
    clean=Image.new("RGB",(576,448),(80,120,60));before=clean.tobytes()
    for variant in range(4):
        edited,mask,_=render_pair(clean,variant,random.Random(42),"unused")
        changed=np.any(np.asarray(edited)!=np.asarray(clean),axis=2)
        assert not np.any(changed&(np.asarray(mask)==0))
        assert clean.tobytes()==before
        if variant==3:
            assert not changed.any() and edited.tobytes()==before
    with pytest.raises(ValueError):
        render_pair(clean,4,random.Random(42),"unused")
