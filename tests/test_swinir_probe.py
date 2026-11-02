import hashlib

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from watermark_studio.swinir_probe import load_swinir, make_swinir, upscale_swinir


class Nearest4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.seen = []

    def forward(self, values):
        self.seen.append(tuple(values.shape[-2:]))
        return torch.nn.functional.interpolate(values, scale_factor=4, mode="nearest")


def test_padding_crop_and_overlapping_assembly_preserve_rgb():
    pixels = np.random.default_rng(1).integers(0, 256, (39, 57, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    net = Nearest4()
    out = upscale_swinir(image, net, tile=32, overlap=8)
    expected = pixels.repeat(4, axis=0).repeat(4, axis=1)
    assert np.array_equal(np.asarray(out), expected)
    assert len(net.seen) > 1
    full = Nearest4()
    assert np.array_equal(np.asarray(upscale_swinir(image, full, tile=0)), expected)
    assert full.seen == [(40, 64)]


def test_loading_rejects_non_model_state(tmp_path):
    path = tmp_path / "fake.pth"
    torch.save({"params": {"bad": torch.zeros(1)}}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="key, shape and dtype"):
        load_swinir(path, digest, "cpu", "lightweight")
    with pytest.raises(ValueError, match="SHA mismatch"):
        load_swinir(path, "0" * 64, "cpu", "lightweight")


def test_profile_validation_and_tile_bounds():
    with pytest.raises(ValueError, match="research profiles"):
        make_swinir("gan")
    with pytest.raises(ValueError, match="Window-aligned"):
        upscale_swinir(Image.new("RGB", (32, 32)), Nearest4(), tile=33)


def test_half_precision_rejected_on_cpu_before_checkpoint_load(tmp_path):
    with pytest.raises(ValueError, match="CUDA float16"):
        load_swinir(tmp_path / "unused", "0" * 64, "cpu", "lightweight", "float16")
