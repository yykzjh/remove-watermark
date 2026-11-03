import runpy
from pathlib import Path

import pytest

helper=runpy.run_path(str(Path(__file__).parents[1]/"scripts/probe-ocr-bound-counterfactual.py"))


def test_native_word_maps_from_its_actual_crop_without_mutation():
    word={"box":[.25,.2,.75,.4],"quad":[[.25,.2],[.75,.2],[.75,.4],[.25,.4]],
          "confidence":.99,"transcript_sha256":"example"}
    mapped=helper["global_word"](word,(100,200,300,400),(1000,1000))
    assert mapped["box"]==pytest.approx([.15,.24,.25,.28])
    from watermark_studio.evidence import valid_box
    assert valid_box(mapped["box"])==pytest.approx(mapped["box"])
    assert all(type(value) is float for value in mapped["box"])
    assert all(type(value) is float for point in mapped["quad"] for value in point)
    assert word["box"]==[.25,.2,.75,.4]
    assert helper["inside_fraction"](mapped,[.1,.2,.3,.4])==pytest.approx(1)


@pytest.mark.parametrize("crop,size",[
    ((-1,0,20,20),(100,100)),((0,0,120,20),(100,100)),((20,0,10,20),(100,100)),
])
def test_wrong_photo_crop_cannot_provide_word_evidence(crop,size):
    with pytest.raises(ValueError,match="escaped"):
        helper["global_word"]({"quad":[],"box":[]},crop,size)


def test_inconsistent_quad_cannot_mask_a_different_region():
    word={"box":[.1,.1,.8,.8],"quad":[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]}
    with pytest.raises(ValueError,match="disagree"):
        helper["global_word"](word,(0,0,100,100),(100,100))
