import numpy as np
import pytest
from PIL import Image

torch=pytest.importorskip("torch")

from watermark_studio.detail_foreground_model import build_foreground_model, foreground_scores


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
