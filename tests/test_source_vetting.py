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

def test_json_recovery_preserves_trusted_source_carrier_clarification(monkeypatch):
    from watermark_studio.cross_review import CLASSIFY_PROMPT, _recover_classification
    observed=[]
    def structured(chats,*args):
        observed.append(chats[0][0]["content"][-1]["text"])
        return ([verdict(physical_scene_text=False,digital_overlay=True,object_attachment=False,
                          surface_evidence=False,uncertain=True)],{})
    monkeypatch.setattr("watermark_studio.quality._structured_vqa",structured)
    chat=[{"role":"user","content":[{"type":"text","text":"source"},
            {"type":"text","text":CLASSIFY_PROMPT+"真实卷尺的印刷刻度属于实物文字。"}]}]
    report=_recover_classification(chat,"{}","missing flags",None,None,"cpu")
    assert "真实卷尺的印刷刻度属于实物文字" in observed[0]
    assert report["uncertain"] and not report["physical_scene_text"]

def test_source_carrier_classifier_receives_global_context_and_native_detail():
    from watermark_studio.source_vetting import source_carrier_chats
    image=Image.new("RGB",(2000,1500))
    chat=source_carrier_chats(image,(.4,.4,.5,.5))[0][0]["content"]
    views=[c["image"] for c in chat if c["type"]=="image"]
    assert len(views)==2 and views[0].size==(640,480)
    assert max(views[1].size)<=640 and views[1].size!=views[0].size
    assert "不要把别处水印" in chat[0]["text"]

def test_batched_source_claims_never_cross_photos_with_identical_boxes(monkeypatch):
    from watermark_studio.source_vetting import vet_sources
    images=[Image.new("RGB",(800,600),"red"),Image.new("RGB",(800,600),"blue")]
    region=Region(kind="panel",box=(.4,.4,.5,.5))
    def classify(chats,encoder,*args):
        colors=[next(c["image"] for c in chat[0]["content"] if c["type"]=="image").getpixel((0,0))
                for chat in chats]
        if encoder=="editor":
            assert colors==[(255,0,0),(0,0,255)]
            return [verdict(),verdict(physical_scene_text=False,digital_overlay=True,
                                    object_attachment=False,surface_evidence=False)]
        assert colors==[(255,0,0)]
        return [verdict()]
    monkeypatch.setattr("watermark_studio.source_vetting._classify",classify)
    output,audits=vet_sources(images,[([region],"red"),([region],"blue")],
        "editor",None,"verifier",None,"cuda","different_weights",2)
    assert output[0][0]==[] and output[1][0]==[region]
    assert audits[0][0]["status"]=="protected_physical_text"
    assert audits[1][0]["status"]=="retained_for_edit"
    assert all(a[0]["primary_batch_size"]==2 for a in audits.values())

def test_source_batch_protocol_failure_splits_without_guessing_classifications(monkeypatch):
    from watermark_studio.source_vetting import vet_sources
    region=Region(kind="panel",box=(.4,.4,.5,.5))
    sizes=[]
    def classify(chats,*args):
        sizes.append(len(chats))
        if len(chats)>1:raise ValueError("protocol count")
        return [verdict(physical_scene_text=False,digital_overlay=True,object_attachment=False,
                        surface_evidence=False)]
    monkeypatch.setattr("watermark_studio.source_vetting._classify",classify)
    output,audits=vet_sources([Image.new("RGB",(800,600))]*2,[([region],"raw")]*2,
        None,None,None,None,"cuda","different_weights",4)
    assert sizes==[2,1,1]
    assert all(d[0]==[region] for d in output)
    assert all(a[0]["primary_batch_size"]==1 for a in audits.values())


@pytest.mark.parametrize("last",["digital","physical","uncertain","missing","error"])
def test_wide_recheck_needs_two_explicit_digital_proofs(monkeypatch,last):
    digital=verdict(physical_scene_text=False,digital_overlay=True,object_attachment=False,surface_evidence=False)
    final={"digital":digital,"physical":verdict(),"uncertain":digital|{"uncertain":True},
           "missing":{"digital_overlay":True},"error":RuntimeError("unavailable")}[last]
    answers=iter([verdict(),digital,digital,final])
    def classify(*args):
        report=next(answers)
        if isinstance(report,Exception):raise report
        return [report]
    monkeypatch.setattr("watermark_studio.source_vetting._classify",classify)
    region=Region(kind="panel",box=(0,.95,.4,1))
    result,audit=vet_small_panels(Image.new("RGB",(2000,1500)),([region],"raw"),
        None,None,None,None,"cuda","different_weights",recheck=True)
    assert (region in result[0]) is (last=="digital")
    assert bool(audit[0].get("requires_review")) is (last!="digital")


def test_failure_after_physical_claim_does_not_authorize_destructive_edit(monkeypatch):
    def classify(chats,encoder,*args):
        if encoder=="editor":return [verdict()]
        raise RuntimeError("verifier unavailable")
    monkeypatch.setattr("watermark_studio.source_vetting._classify",classify)
    region=Region(kind="panel",box=(.4,.4,.5,.5))
    result,audit=vet_small_panels(Image.new("RGB",(800,600)),([region],"raw"),
        "editor",None,"verifier",None,"cuda","different_weights")
    assert result[0]==[] and audit[0]["requires_review"]


def test_wide_carrier_view_contains_surroundings_of_thin_strip():
    from watermark_studio.source_vetting import source_carrier_chats
    image=Image.new("RGB",(2000,1500))
    views=[]
    for wide in (False,True):
        chat=source_carrier_chats(image,(0,.96,.35,1),wide=wide)[0][0]["content"]
        views.append([c["image"] for c in chat if c["type"]=="image"][1])
    assert views[1].height>views[0].height*3


def test_wide_source_uncertainty_describes_identity_not_absence_of_physical_carrier():
    from watermark_studio.source_vetting import source_carrier_chats
    image=Image.new("RGB",(800,600))
    normal=source_carrier_chats(image,(0,.9,.4,1))[0][0]["content"][-1]["text"]
    wide=source_carrier_chats(image,(0,.9,.4,1),wide=True)[0][0]["content"][-1]["text"]
    assert "载体或表面证据不足必须uncertain。" in normal
    assert "不能仅因此标uncertain" in wide
    assert "不能因缺少实物证据就自动断言是水印" in wide
    assert "载体或表面证据不足必须uncertain。" not in wide
