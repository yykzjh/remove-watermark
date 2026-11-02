import numpy as np
import pytest
from PIL import Image

torch=pytest.importorskip("torch")

from watermark_studio.detail_foreground_model import (
    build_foreground_model,
    foreground_scores,
    paired_affine_features,
)


def test_native_classifier_tiles_keep_coordinate_map_and_independent_batch_entries():
    class Pixel(torch.nn.Module):
        def forward(self,values):return values[:,0:1]*40
    source=np.zeros((350,470,3),np.uint8);source[40:90,120:165,0]=255
    before=Image.fromarray(source);reference=Image.new("RGB",before.size,0)
    scores=foreground_scores(Pixel(),before,reference,(10,20,450,330),"cpu",batch_size=3)
    assert scores.shape==(350,470) and np.isfinite(scores).all()
    assert scores[40:90,120:165].min()>.99
    assert scores[100:200,200:300].max()<.01
    assert np.all(scores[:20]==1)  # outside classified scope remains unknown


def test_model_outputs_native_pixel_logits_for_an_odd_sized_tile():
    torch.set_num_threads(1)
    model=build_foreground_model().eval()
    with torch.inference_mode():
        result=model(torch.randn(2,6,65,73))
    assert result.shape==(2,1,65,73) and torch.isfinite(result).all()


def test_affine_features_cancel_neutral_field_and_preserve_new_local_ink_residual():
    torch.set_num_threads(1)
    y,x=torch.meshgrid(torch.linspace(-.6,.6,128),torch.linspace(-.5,.5,128),indexing="ij")
    reference=torch.stack([x,y,(x+y)*.5])[None]
    for transmission in (.15,.55):
        source=transmission*reference+.35
        features=paired_affine_features(torch.cat([source,reference],dim=1))
        assert features.shape==(1,10,128,128)
        assert features[:,6:9].abs().max()<1e-5
        assert torch.max(torch.abs(features[:,9]-(2*transmission-1)))<1e-5
        changed=source.clone();changed[:,:,32:40,48:60]-=.12
        altered=paired_affine_features(torch.cat([changed,reference],dim=1))
        assert altered[:,6:9,32:40,48:60].mean()<-.3
        assert altered[:,6:9,:16,:16].abs().max()<.03


def test_affine_pixel_network_backward_is_finite_without_ground_truth_opacity():
    torch.set_num_threads(1)
    model=build_foreground_model("affine").train()
    values=torch.randn(2,6,64,72)
    logits=model(values)
    target=torch.zeros_like(logits);target[:,:,12:30,20:32]=1
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters())
