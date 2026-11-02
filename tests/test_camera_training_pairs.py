import random

import numpy as np
import pytest
from PIL import Image, ImageFont

from watermark_studio.training_pairs import render_pair, render_profile_pair


def test_versioned_camera_overlays_replay_exact_masks_and_do_not_change_legacy(monkeypatch):
    fallback=ImageFont.load_default()
    monkeypatch.setattr("watermark_studio.training_pairs.ImageFont.truetype",lambda *a,**k:fallback)
    clean=Image.fromarray(np.random.default_rng(5).integers(0,256,(448,576,3),dtype=np.uint8))
    before=clean.tobytes()
    for variant in range(4):
        for seed in (7,19,2026):
            image,mask,meta,layer=render_profile_pair(clean,variant,random.Random(seed),"unused",
                profile="camera-v2",return_overlay=True)
            assert clean.tobytes()==before
            assert image.tobytes()==Image.alpha_composite(clean.convert("RGBA"),layer).convert("RGB").tobytes()
            changed=np.any(np.asarray(image)!=np.asarray(clean),axis=2)
            assert not np.any(changed&(np.asarray(mask)==0))
            assert meta["profile"]=="camera-v2"
            if variant==3:assert not changed.any()
            if variant==0:assert meta["header_height"]>0 and meta["rows"]>=4
        old=render_pair(clean,variant,random.Random(7),"unused")[0]
        dispatched=render_profile_pair(clean,variant,random.Random(7),"unused")[0]
        assert old.tobytes()==dispatched.tobytes()
    with pytest.raises(ValueError):
        render_profile_pair(clean,0,random.Random(7),"unused",profile="unknown")
