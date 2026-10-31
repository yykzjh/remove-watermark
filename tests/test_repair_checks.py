import json
import warnings

import numpy as np
import pytest
from PIL import Image, ImageDraw

from watermark_studio.detection import parse_detection
from watermark_studio.repair_checks import rectangular_repairs


def region(box):
    return parse_detection(json.dumps({"watermarks":[{"kind":"panel","box":[round(v*1000) for v in box]}]}))


@pytest.mark.parametrize("box",[(.2,.4,.7,.9),(.02,.7,.4,.98)])
def test_coherent_neutral_tone_step_is_detected_even_when_detector_padding_clips_edges(box):
    source=Image.new("RGB",(600,450),(100,100,100));after=source.copy()
    native=tuple(round(v*s) for v,s in zip(box,(600,450,600,450)))
    ImageDraw.Draw(source).rectangle(native,fill="blue")
    ImageDraw.Draw(after).rectangle(native,fill=(93,93,93))
    result=rectangular_repairs(source,after,region(box))
    assert result["count"]>0
    json.dumps(result,allow_nan=False)
    assert rectangular_repairs(after,after.copy(),region(box))["count"]==0


def test_linear_real_illumination_is_not_misread_as_a_mask_rectangle():
    yy,xx=np.mgrid[:450,:600]
    gray=(90+xx*.05+yy*.04).astype(np.uint8)
    clean=Image.fromarray(np.repeat(gray[...,None],3,axis=2))
    source=clean.copy();ImageDraw.Draw(source).rectangle((120,180,420,405),fill="blue")
    assert rectangular_repairs(source,clean,region((.2,.4,.7,.9)))["count"]==0


def test_profile_slices_are_bounded_and_all_evidence_is_finite():
    rng=np.random.default_rng(7)
    photo=Image.fromarray(rng.integers(0,256,(450,600,3),dtype=np.uint8))
    source=photo.copy();ImageDraw.Draw(source).rectangle((120,180,420,405),fill="blue")
    with warnings.catch_warnings():
        warnings.simplefilter("error",RuntimeWarning)
        result=rectangular_repairs(source,photo,region((.2,.4,.7,.9)))
    json.dumps(result,allow_nan=False)
