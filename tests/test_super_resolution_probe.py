from types import SimpleNamespace

import pytest
from PIL import Image


def test_rrdb_architecture_shapes_match_fixed_rgb_x4_checkpoint():
    torch=pytest.importorskip("torch")
    from watermark_studio.super_resolution_probe import make_rrdbnet
    model=make_rrdbnet(features=4,growth=2,blocks=1)
    assert set(model.state_dict())>={"body.0.rdb3.conv5.weight","conv_up2.bias","conv_last.weight"}
    with torch.inference_mode():
        result=model(torch.rand(1,3,8,9))
    assert result.shape==(1,3,32,36) and torch.isfinite(result).all()


def test_tiled_sr_keeps_rgb_alignment_without_mutating_the_full_input():
    torch=pytest.importorskip("torch")
    import numpy as np

    from watermark_studio.super_resolution_probe import upscale_tiled
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.ones(()))
        def forward(self,x):
            return torch.nn.functional.interpolate(x,scale_factor=4,mode="nearest")
    values=np.random.default_rng(8).integers(0,256,(35,57,3),dtype=np.uint8)
    image=Image.fromarray(values);before=image.tobytes()
    result=upscale_tiled(image,Net(),tile=32,padding=8)
    assert np.array_equal(np.asarray(result),np.repeat(np.repeat(values,4,0),4,1))
    assert result.size==(228,140) and image.tobytes()==before
    with pytest.raises(ValueError):
        upscale_tiled(image,SimpleNamespace(),tile=1)
