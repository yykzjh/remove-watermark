import numpy as np
import pytest
from PIL import Image

from watermark_studio.translucent_hint import roi_body_conditioning


def test_transmitted_detail_hint_preserves_every_pixel_outside_its_audited_scope():
    rng = np.random.default_rng(42)
    truth = rng.integers(60, 170, (320, 480, 3), dtype=np.uint8)
    # Exact neutral foreground, known transparency. This is a conditioning
    # fixture, not a claim about any native photo's unknown alpha.
    body = np.rint(truth[100:290, 70:300]*.45 + 255*.55).astype(np.uint8)
    original = truth.copy()
    original[100:290, 70:300] = body
    photo = Image.fromarray(original)
    result, audit = roi_body_conditioning(photo, (60, 80, 310, 300), .45)
    assert audit["applied"]
    assert audit["quality_permission"] is False and audit["final_pixels"] is False
    l, t, r, b = audit["body_pixels"]
    changed = np.any(np.asarray(result) != original, axis=2)
    allowed = np.zeros(changed.shape, bool)
    allowed[t:b, l:r] = True
    assert changed.any() and not changed[~allowed].any()
    inner = np.asarray(result)[t+16:b-16, l+16:r-16].astype(float)
    assert np.abs(inner-truth[t+16:b-16, l+16:r-16]).mean() < 1
    assert np.array_equal(np.asarray(photo), original)


def test_flat_card_contains_no_reliable_transmitted_detail_and_keeps_exact_original():
    original = Image.new("RGB", (400, 300), (220, 220, 220))
    result, audit = roi_body_conditioning(original, (10, 20, 300, 130), .45)
    assert result is original and not audit["applied"]


@pytest.mark.parametrize("box,beta", [((-1, 0, 100, 100), .45), ((0., 0, 100, 100), .45),
                                     ((0, 0, 100, 100), .1), ((0, 0, 100, 100), float("nan"))])
def test_invalid_source_geometry_and_unbounded_alpha_are_rejected(box, beta):
    with pytest.raises(ValueError):
        roi_body_conditioning(Image.new("RGB", (400, 300)), box, beta)
