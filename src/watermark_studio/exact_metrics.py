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



def detail_reconstruction_metrics(actual, expected, mask):
    """Known-target texture fidelity: matching sharpness alone is not fidelity."""
    from PIL import Image, ImageFilter

    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    mask = np.asarray(mask)
    # Reuse all shape, range-of-support and finiteness validation.
    reconstruction_metrics(actual, expected, mask)
    if not mask.any():
        return {"pixels": 0, "highpass_rmse": None, "highpass_correlation": None,
                "detail_energy_ratio": None}
    fields = []
    for image in (actual, expected):
        gray = image @ np.array([.2126, .7152, .0722], dtype=np.float32)
        low = np.asarray(Image.fromarray(np.rint(gray).clip(0, 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(2)), dtype=np.float32)
        fields.append((gray - low)[mask].astype(np.float64))
    observed, target = fields
    left, right = observed - observed.mean(), target - target.mean()
    energy = float(np.sqrt(np.mean(target**2)))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return {"pixels": int(mask.sum()),
            "highpass_rmse": float(np.sqrt(np.mean((observed - target)**2))),
            "highpass_correlation": float(np.dot(left, right) / denominator) if denominator > 1e-8 else None,
            "detail_energy_ratio": float(np.sqrt(np.mean(observed**2)) / energy) if energy > 1e-8 else None}
