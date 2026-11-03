import numpy as np
import pytest
from PIL import Image, ImageDraw

from watermark_studio.roi_supervision import make_roi_supervision
from watermark_studio.training_features import training_prompt


def test_roi_ground_truth_preserves_secondary_overlay_and_context_pixels(monkeypatch):
    monkeypatch.setattr("watermark_studio.roi_supervision.dimensions",lambda *args:(512,416))
    clean=Image.new("RGB",(800,600),(40,100,40));edited=clean.copy()
    draw=ImageDraw.Draw(edited);draw.rectangle((80,260,400,500),fill="red")
    draw.rectangle((460,400,520,440),fill="white")
    source_bytes=edited.tobytes()
    source,target,mask,geometry=make_roi_supervision(edited,clean,(80,260,401,501),512,"panel")
    support=np.asarray(mask)>0
    a,b=np.asarray(source),np.asarray(target)
    assert np.array_equal(a[~support],b[~support])
    assert np.any(a[support]!=b[support])
    assert np.any((b==255).all(axis=2)&(~support))
    assert edited.tobytes()==source_bytes
    assert source.size==target.size==mask.size
    assert all(v%32==0 for v in source.size)
    assert "Remove only" in geometry["prompt"]


def test_roi_feature_prompt_cannot_silently_fall_back_to_whole_frame():
    assert training_prompt({"edit_prompt":"crop-relative focused instruction"},{"prompt_profile":"roi_focused_v1"})=="crop-relative focused instruction"
    with pytest.raises(ValueError,match="explicit bounded"):
        training_prompt({},{"prompt_profile":"roi_focused_v1"})
    with pytest.raises(ValueError,match="Unknown"):
        training_prompt({},{"prompt_profile":"unsupported"})

def test_interpolation_support_scales_for_small_native_logos(monkeypatch):
    monkeypatch.setattr("watermark_studio.roi_supervision.dimensions",lambda *args:(512,704))
    clean=Image.new("RGB",(100,100),(100,100,100));edited=clean.copy()
    draw=ImageDraw.Draw(edited);draw.rectangle((0,0,49,99),fill="red")
    draw.rectangle((51,0,99,99),fill="white")
    source,target,mask,geometry=make_roi_supervision(edited,clean,(0,0,50,100),512,"logo",box_is_padded=True)
    outside=np.asarray(mask)==0
    assert geometry["resampling_radius"]>3
    assert np.array_equal(np.asarray(source)[outside],np.asarray(target)[outside])
    assert "verification code" in geometry["prompt"]
