import numpy as np
from PIL import Image

from watermark_studio.detail_foreground import detail_support, foreground_challenge


def photo():
    y,x=np.mgrid[:384,:512]
    rgb=np.stack([80+x*.1+15*np.sin(y*.13),100+y*.2+12*np.cos(x*.2),90+x*.15+y*.1],axis=-1)
    return Image.fromarray(np.rint(np.clip(rgb,0,255)).astype(np.uint8))


def test_encoded_counterfactual_labels_only_added_ink_and_plain_body_has_no_ink():
    clean=photo()
    for quality in (0,95):
        source,counter,glyph,meta=foreground_challenge(clean,None,"none",5,quality=quality)
        assert np.array_equal(np.asarray(source),np.asarray(counter)) and not glyph.any()
        source,counter,glyph,meta=foreground_challenge(clean,None,"faint",5,quality=quality)
        difference=np.max(np.abs(np.asarray(source).astype(np.int16)-np.asarray(counter).astype(np.int16)),axis=2)>1
        assert np.array_equal(glyph,difference) and glyph.any()
        assert meta["quality_permission"] is False


def test_missing_text_anchor_is_not_a_body_permission_and_ocr_boxes_are_excluded():
    source,_,_glyph,meta=foreground_challenge(photo(),None,"gray",5)
    l,t,r,b=meta["body_box"];box=(l+4,t+4,r-4,b-4)
    support,proof=detail_support(source,box,{"status":"clear"},profile="ocr8")
    assert not support.any() and proof["quality_permission"] is False
    detection={"status":"text_present","regions":[{"textlike_boxes":[[l/512,t/384,r/512,b/384]]}]}
    support,_=detail_support(source,box,detection,profile="ocr8")
    assert not support.any()


def test_pixel_exclusions_are_not_a_certificate_for_faint_foreground():
    source,_,glyph,meta=foreground_challenge(photo(),None,"faint",99)
    support,proof=detail_support(source,meta["body_box"],{
        "status":"text_present","regions":[{"textlike_boxes":[[.1,.44,.5,.5]]}]},profile="pixel")
    assert (support & glyph).any()
    assert proof["quality_permission"] is False
