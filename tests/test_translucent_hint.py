import hashlib

import numpy as np
import pytest
from PIL import Image, ImageDraw

from watermark_studio.config import Region
from watermark_studio.source_panel_rims import propose_blue_card_rims
from watermark_studio.translucent_hint import translucent_body_hint


def fixture():
    rng = np.random.default_rng(28)
    gray = rng.integers(55, 175, (600,800), dtype=np.uint8)
    clean = np.repeat(gray[...,None], 3, axis=2)
    pixels = clean.copy()
    pixels[400:460,20:351] = (30,110,220)
    pixels[460:590,20:351] = np.rint(.55*clean[460:590,20:351]+.45*255).astype(np.uint8)
    source = Image.fromarray(pixels)
    region = Region(kind="panel", box=(0,.59,.49,1))
    core = {"mask": list(region.box), "core": [0,.62,.46,.99]}
    row = propose_blue_card_rims(source, [region], [core])[0]
    yes = {"digital_card":True,"rim_clear":True,"uncertain":False,"area":"none"}
    row.update(approved=True, editor=yes, independent=yes, weights_relation="different_weights",
               ocr={"status":"clear"}, source_rim_sha256=hashlib.sha256(
                   source.crop(tuple(row["restore_pixels"])).tobytes()).hexdigest())
    return source, clean, row


def test_known_transmission_hint_recovers_real_structure_and_preserves_every_other_pixel():
    source, clean, row = fixture()
    original = source.tobytes()
    hint, audit = translucent_body_hint(source, [row], .55)
    assert audit[0]["applied"] and not audit[0]["quality_permission"]
    l,t,r,b = audit[0]["body_pixels"]
    before = np.asarray(source)
    after = np.asarray(hint)
    outside = np.ones(before.shape[:2], bool); outside[t:b,l:r] = False
    assert np.array_equal(after[outside], before[outside])
    assert np.abs(after[t+16:b-16,l+16:r-16].astype(float)-
                  clean[t+16:b-16,l+16:r-16].astype(float)).mean() < 1
    assert source.tobytes() == original and hint.size == source.size


def test_missing_or_stale_proof_cannot_initialize_a_body():
    source, _, row = fixture()
    assert translucent_body_hint(source, [row|{"approved":False}])[0] is source
    _, audit = translucent_body_hint(source, [row|{"source_rim_sha256":"stale"}])
    assert not audit[0]["applied"]


def test_opaque_uniform_body_does_not_supply_recoverable_photo_structure():
    source, _, row = fixture()
    ImageDraw.Draw(source).rectangle((20,460,350,589), fill=(225,225,225))
    _, audit = translucent_body_hint(source, [row])
    assert not audit[0]["applied"]


@pytest.mark.parametrize("beta", [0., .9, float("nan")])
def test_invalid_transmission_cannot_silently_change_a_reference(beta):
    source, _, row = fixture()
    with pytest.raises(ValueError):
        translucent_body_hint(source, [row], beta)
