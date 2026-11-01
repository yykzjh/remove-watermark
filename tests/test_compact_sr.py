import numpy as np
import pytest

from watermark_studio.compact_sr import checked_weights, tile_plan


def test_compact_sr_rejects_unpinned_content_before_importing_torch(tmp_path):
    path = tmp_path/"untrusted.pth"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="pinned"):
        checked_weights(path)
    path.write_bytes(b"0"*1_000_000)
    with pytest.raises(ValueError, match="SHA256"):
        checked_weights(path)


def test_tiled_receptive_context_covers_each_pixel_exactly_once():
    width, height = 731, 429
    covered = np.zeros((height, width), dtype=np.uint8)
    for (x0, y0, x1, y1), (ex0, ey0, ex1, ey1) in tile_plan((width, height)):
        covered[y0:y1, x0:x1] += 1
        assert (ex0, ey0, ex1, ey1) == (max(0, x0-34), max(0, y0-34),
                                       min(width, x1+34), min(height, y1+34))
    assert np.all(covered == 1)
    with pytest.raises(ValueError):
        list(tile_plan((20, 20), halo=10))
