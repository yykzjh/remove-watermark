import numpy as np

from watermark_studio.exact_metrics import detail_reconstruction_metrics


def test_sharp_wrong_contrast_is_not_faithful_detail():
    grid = np.indices((32, 32)).sum(axis=0) % 4 < 2
    clean = np.repeat(np.where(grid, 200, 55).astype(np.uint8)[..., None], 3, axis=2)
    mask = np.ones((32, 32), dtype=bool)
    correct = detail_reconstruction_metrics(clean, clean, mask)
    wrong = detail_reconstruction_metrics(255 - clean, clean, mask)
    assert correct["highpass_correlation"] > .999999
    assert correct["highpass_rmse"] == 0
    assert .95 < wrong["detail_energy_ratio"] < 1.05
    assert wrong["highpass_correlation"] < -.99
    assert wrong["highpass_rmse"] > 100


def test_empty_target_support_is_not_quality_evidence():
    clean = np.zeros((8, 8, 3), dtype=np.uint8)
    result = detail_reconstruction_metrics(clean, clean, np.zeros((8, 8), dtype=bool))
    assert result["pixels"] == 0
    assert result["highpass_correlation"] is None
