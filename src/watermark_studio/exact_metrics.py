"""Diagnostics with known exact synthetic ground truth; never native QA permissions."""
from __future__ import annotations

import math

import numpy as np


def reconstruction_metrics(actual,expected,mask):
    actual=np.asarray(actual,dtype=np.float32);expected=np.asarray(expected,dtype=np.float32)
    mask=np.asarray(mask)
    if (actual.ndim!=3 or actual.shape[-1]!=3 or actual.shape!=expected.shape or
            mask.shape!=actual.shape[:2] or mask.dtype!=np.bool_ or
            not np.isfinite(actual).all() or not np.isfinite(expected).all()):
        raise ValueError("Aligned finite RGB and exact boolean support required")
    count=int(mask.sum())
    if not count:return {"pixels":0,"mse":None,"psnr":None,"mae":None,
        "mean_rgb_bias":None,"low_frequency_rmse":None}
    difference=actual-expected
    selected=difference[mask];mse=float(np.mean(selected**2))
    # A 7x7 signed-error average retains broad tint/rectangle shadows while
    # suppressing individual texture/edge deviations. Exact target only.
    radius=3;side=2*radius+1
    padded=np.pad(difference,((radius,radius),(radius,radius),(0,0)),mode="reflect")
    integral=np.pad(padded,((1,0),(1,0),(0,0))).cumsum(0).cumsum(1)
    averaged=(integral[side:,side:]-integral[:-side,side:]-
        integral[side:,:-side]+integral[:-side,:-side])/(side*side)
    return {"pixels":count,"mse":mse,"psnr":-10*math.log10(max(mse,1e-8)/255**2),
        "mae":float(np.mean(np.abs(selected))),
        "mean_rgb_bias":[float(v) for v in selected.mean(axis=0)],
        "low_frequency_rmse":float(np.sqrt(np.mean(averaged[mask]**2)))}
