import numpy as np

from watermark_studio.exact_metrics import reconstruction_metrics


def test_known_rectangle_shadow_is_reported_without_inventing_a_quality_pass():
    clean=np.full((32,40,3),160,dtype=np.uint8)
    mask=np.zeros((32,40),bool);mask[8:24,10:30]=True
    shadow=clean.copy();shadow[mask]-=12
    exact=reconstruction_metrics(clean,clean,mask)
    defective=reconstruction_metrics(shadow,clean,mask)
    assert exact["mse"]==exact["low_frequency_rmse"]==0
    assert defective["mean_rgb_bias"]==[-12.,-12.,-12.]
    assert defective["mse"]==144 and defective["low_frequency_rmse"]>9
    assert "passed" not in defective
    assert reconstruction_metrics(shadow,clean,~mask)["mse"]==0
    empty=reconstruction_metrics(clean,clean,np.zeros_like(mask))
    assert empty["pixels"]==0 and empty["mse"] is None
