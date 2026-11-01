import numpy as np
import pytest
from PIL import Image, ImageDraw

from watermark_studio.lighting_diagnostics import lighting_view


def test_constant_light_has_no_artificial_panel_and_source_is_unchanged():
    source=Image.new("RGB",(800,600),(90,100,110))
    before=source.tobytes()
    result=lighting_view(source)
    assert result.size==(512,384) and np.all(np.asarray(result)==128)
    assert source.tobytes()==before


def test_faint_real_rectangle_is_amplified_without_drawing_a_box():
    clean=Image.new("RGB",(512,512),(140,140,140))
    shadow=clean.copy()
    ImageDraw.Draw(shadow).rectangle((120,120,390,390),fill=(130,130,130))
    result=np.asarray(lighting_view(shadow))[...,0]
    assert result[126,200]<110 and result[117,200]>145
    assert np.all(np.asarray(lighting_view(clean))==128)
    assert shadow.getpixel((200,200))==(130,130,130)


@pytest.mark.parametrize("size",[0,2048])
def test_diagnostic_size_is_bounded(size):
    with pytest.raises(ValueError):
        lighting_view(Image.new("RGB",(100,100)),size)
