import numpy as np
import pytest

from watermark_studio.segmentation_probe import mask_truth, prompt_boxes


def test_segmentation_jitter_keeps_native_full_frame_coordinates_and_rejects_collapse():
    assert prompt_boxes((.1,.2,.5,.8),(1000,800))==[[100.,160.,500.,640.]]
    boxes=prompt_boxes((0,.2,.5,1),(1000,800),.01)
    assert len(boxes)==3 and all(0<=b[0]<b[2]<=1000 and 0<=b[1]<b[3]<=800 for b in boxes)
    with pytest.raises(ValueError):
        prompt_boxes((.4,.4,.41,.41),(1000,800),.02)


def test_exact_overlay_truth_cannot_ignore_even_one_missed_pixel():
    truth=np.zeros((10,12),bool);truth[3:7,2:8]=True
    assert mask_truth(truth,truth)["full_coverage"]
    bad=truth.copy();bad[3,2]=False
    report=mask_truth(bad,truth)
    assert report["missed_pixels"]==1 and not report["full_coverage"]
    extra=truth.copy();extra[2,1]=True
    assert mask_truth(extra,truth)["outside_pixels"]==1
    with pytest.raises(ValueError):
        mask_truth(truth.astype(np.uint8),truth)
