import runpy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.mark.parametrize("position", range(4))
def test_control_has_exact_gt_and_only_known_overlays_change_pixels(position):
    overlay = runpy.run_path(str(Path(__file__).parents[1]/"scripts/prepare-translucent-controls.py"))["card_overlay"]
    rng = np.random.default_rng(194)
    clean = Image.fromarray(rng.integers(20,210,(600,800,3),dtype=np.uint8))
    original = clean.tobytes()
    planted, info = overlay(clean,position,.45)
    source, result = np.asarray(clean), np.asarray(planted)
    outside = np.ones(source.shape[:2],bool)
    for key in ("panel","footer"):
        l,t,r,b = info[key]
        outside[t:b+1,l:r+1] = False
    assert np.array_equal(source[outside],result[outside]) and clean.tobytes() == original
    assert planted.size == clean.size
    # Clear unprinted point in the card bottom-right, away from the camera footer.
    l,t,r,b = info["panel"]
    x,y = r-10,b-10
    if not (info["footer"][0]<=x<=info["footer"][2] and info["footer"][1]<=y<=info["footer"][3]):
        expected = .45*source[y,x].astype(float)+.55*255
        assert np.max(np.abs(result[y,x].astype(float)-expected)) <= 1


def test_ground_truth_diagnostic_detects_retained_veil_without_demanding_hidden_pixel_identity():
    helper = runpy.run_path(str(Path(__file__).parents[1]/"scripts/prepare-translucent-controls.py"))
    metrics = runpy.run_path(str(Path(__file__).parents[1]/"scripts/benchmark-translucent-truth.py"))["known_body_metrics"]
    rng = np.random.default_rng(195)
    clean = Image.fromarray(rng.integers(30,210,(600,800,3),dtype=np.uint8))
    marked, geometry = helper["card_overlay"](clean,0,.45)
    correct = metrics(clean,marked,clean,geometry)
    retained = metrics(clean,marked,marked,geometry)
    assert correct["transmitted_mae"] == 0 and correct["transmitted_luma_correlation"] > .99999
    assert retained["transmitted_mae"] > 40 and retained["planted_background_mae"] > 40
    assert correct["transmitted_pixels"] == retained["transmitted_pixels"]
