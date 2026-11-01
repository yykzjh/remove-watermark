import numpy as np
import pytest
from PIL import Image

from watermark_studio.text_presence import TargetTextPresence


def test_no_text_veto_retains_photo_and_uses_bounded_cpu_colour_input():
    photo=Image.new("RGB",(1000,800),(10,20,30));before=photo.tobytes();calls=[]
    def engine(array,**kwargs):
        calls.append(array.shape)
        assert array[0,0].tolist()==[30,20,10]
        assert kwargs=={"use_det":True,"use_cls":False,"use_rec":False}
        return None,None
    report=TargetTextPresence(engine,max_edge=400).inspect(photo,[[.1,.2,.9,1]])
    assert report["status"]=="clear" and calls==[(320,400,3)]
    assert photo.tobytes()==before


def test_any_textlike_region_is_a_veto_even_without_transcription():
    quad=[[10,10],[50,10],[50,30],[10,30]]
    engine=lambda *args,**kwargs:([quad],[.1])
    report=TargetTextPresence(engine).inspect(Image.new("RGB",(200,100)),[[.5,0,1,1]])
    assert report["status"]=="text_present"
    assert np.allclose(report["regions"][0]["textlike_boxes"][0],[.55,.1,.75,.3])
    assert "text" not in report["regions"][0]


@pytest.mark.parametrize("response",[None,([],),(["bad"],None),([[[0,0]]],None)])
def test_ocr_errors_are_never_empty_absence_proofs(response):
    engine=lambda *args,**kwargs:response
    report=TargetTextPresence(engine).inspect(Image.new("RGB",(100,100)),[[0,0,1,1]])
    assert report["status"]=="error"
