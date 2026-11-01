import pytest
from PIL import Image

from watermark_studio.config import Region
from watermark_studio.source_vetting import vet_small_panels


def verdict(**changes):
    return {"physical_scene_text":True,"digital_overlay":False,"object_attachment":True,
            "surface_evidence":True,"uncertain":False,"evidence":"文字随金属铭牌表面透视，载体边缘可见。"}|changes

@pytest.mark.parametrize("second,protected",[(verdict(),True),(verdict(digital_overlay=True),False),
    (verdict(uncertain=True),False),(verdict(surface_evidence=False),False)])
def test_source_vet_requires_both_distinct_models_with_complete_carrier_proof(monkeypatch,second,protected):
    small=Region(kind="panel",box=(.4,.4,.5,.5))
    logo=Region(kind="logo",box=(.9,.9,1,1))
    big=Region(kind="panel",box=(0,.7,.5,1))
    answers=iter([[verdict()],[second]])
    monkeypatch.setattr("watermark_studio.source_vetting._classify",lambda *a:next(answers))
    detected,audit=vet_small_panels(Image.new("RGB",(800,600)),([small,logo,big],"raw"),
                                   None,None,None,None,"cuda","different_weights")
    assert small not in detected[0]
    assert bool(audit[0].get("requires_review")) is (not protected)
    assert logo in detected[0] and big in detected[0] and detected[1]=="raw"
    assert audit[0]["status"]==("protected_physical_text" if protected else "uncertain_physical_carrier")

def test_same_weights_cannot_vet_source_overlay(monkeypatch):
    region=Region(kind="panel",box=(.4,.4,.5,.5))
    monkeypatch.setattr("watermark_studio.source_vetting._classify",lambda *a:pytest.fail("same weights"))
    detected,audit=vet_small_panels(Image.new("RGB",(800,600)),([region],"raw"),
                                   None,None,None,None,"cuda","same_base_weights")
    assert detected[0]==[region] and audit[0]["status"]=="disabled"


def test_unambiguous_digital_footer_is_not_discharged(monkeypatch):
    region=Region(kind="panel",box=(.8,.95,1,1))
    calls=[]
    def classify(*args):
        calls.append(1)
        return [verdict(physical_scene_text=False,digital_overlay=True,object_attachment=False,surface_evidence=False)]
    monkeypatch.setattr("watermark_studio.source_vetting._classify",classify)
    detected,audit=vet_small_panels(Image.new("RGB",(800,600)),([region],"raw"),
                                   None,None,None,None,"cuda","different_weights")
    assert detected[0]==[region] and len(calls)==1 and audit[0]["status"]=="retained_for_edit"
