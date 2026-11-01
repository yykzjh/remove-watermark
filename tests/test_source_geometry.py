import json

import pytest
from PIL import Image

from watermark_studio.detection import parse_detection
from watermark_studio.source_geometry import reconcile_broad, refine_broad_source


def detection(boxes):
    text=json.dumps({"scene":"outdoor","watermarks":[{"kind":kind,"box":box} for kind,box in boxes]})
    return parse_detection(text),text


def fixtures():
    footer=("logo",[850,965,995,995])
    original=detection([("panel",[0,80,720,985]),footer])
    first=detection([("panel",[0,650,405,985]),footer])
    second=detection([("panel",[0,651,407,985]),footer])
    return original,first,second


def test_consensus_replaces_the_overbroad_footprint_and_retains_other_source_overlays():
    original,first,second=fixtures()
    corrected=reconcile_broad(original,first,second)
    assert len(corrected[0])==2 and parse_detection(corrected[1])==corrected[0]
    assert any(r.kind=="logo" for r in corrected[0])
    assert all(not (r.box[0]<.4<r.box[2] and r.box[1]<.3<r.box[3]) for r in corrected[0])
    assert any(r.box[1]<.65<r.box[3] for r in corrected[0])


def test_disagreeing_or_still_broad_detectors_cannot_silently_shrink_a_source_mask():
    original,first,_=fixtures()
    with pytest.raises(ValueError):
        reconcile_broad(original,first,original)
    with pytest.raises(ValueError):
        reconcile_broad(original,first,detection([]))


def test_refinement_requires_digital_carrier_proofs_and_never_changes_source_pixels(monkeypatch):
    from watermark_studio import source_geometry
    original,first,second=fixtures();answers=iter([first,second])
    monkeypatch.setattr(source_geometry,"locate_watermarks_batch",lambda *a:[next(answers)])
    claims={"digital_overlay":True,"physical_scene_text":False,"object_attachment":False,"uncertain":False}
    monkeypatch.setattr(source_geometry,"_digital_claim",lambda *a:claims)
    image=Image.new("RGB",(600,400),"gray");before=image.tobytes()
    updated,audit=refine_broad_source(image,original,object(),None,object(),None,"cpu","different_weights")
    assert audit[0]["status"]=="refined_digital_geometry" and not audit[0]["requires_review"]
    assert len(updated[0])==2 and len(audit[0]["digital_proofs"])==1
    assert image.tobytes()==before


@pytest.mark.parametrize("failure",["geometry","classification","exception"])
def test_any_source_ambiguity_protects_the_whole_unconfirmed_area_and_requires_review(monkeypatch,failure):
    from watermark_studio import source_geometry
    original,first,second=fixtures()
    answers=iter([first,original if failure=="geometry" else second])
    monkeypatch.setattr(source_geometry,"locate_watermarks_batch",lambda *a:[next(answers)])
    def classify(*args):
        if failure=="exception":raise RuntimeError("inspection failed")
        return {"digital_overlay":True,"physical_scene_text":True,"object_attachment":True,"uncertain":False}
    monkeypatch.setattr(source_geometry,"_digital_claim",classify)
    updated,audit=refine_broad_source(Image.new("RGB",(600,400)),original,object(),None,object(),None,"cpu","different_weights")
    assert len(updated[0])==1 and updated[0][0].kind=="logo"
    assert audit[0]["requires_review"] and audit[0]["status"]=="uncertain_physical_carrier"
    assert audit[0]["error"]
    assert "source_detectors" in audit[0]
    if failure=="classification":assert audit[0]["digital_proofs"][0]["editor"]["physical_scene_text"] is True


def test_unrelated_small_masks_and_unknown_weight_independence_keep_the_original_contract():
    original,first,_=fixtures()
    image=Image.new("RGB",(600,400))
    assert refine_broad_source(image,first,None,None,None,None,"cpu","different_weights")== (first,[])
    assert refine_broad_source(image,original,None,None,None,None,"cpu","unknown")== (original,[])

def test_digital_uncertainty_is_not_defined_as_missing_physical_carrier(monkeypatch):
    from watermark_studio.source_geometry import _digital_claim
    flags={"physical_scene_text":False,"digital_overlay":True,"object_attachment":False,
           "surface_evidence":False,"uncertain":False,"area":"none"}
    seen=[]
    def structured(chats,*args):
        seen.append(chats[0][0]["content"][-1]["text"])
        assert "缺少实物载体不等于uncertain" in seen[-1]
        return [flags],.1
    def observe(chats,*args):
        text=chats[0][0]["content"][-1]["text"]
        assert "digital_overlay" not in text and "不预设类别" in text
        return [{"status":"summary_only","summary":"半透明蓝白卡悬浮于画面，不随场景表面透视"}]
    monkeypatch.setattr("watermark_studio.quality._structured_vqa",structured)
    monkeypatch.setattr("watermark_studio.evidence._generate_explanations",observe)
    chats=[[{"role":"user","content":[{"type":"image","image":Image.new("RGB",(20,20))},
                                     {"type":"text","text":"original source task"}]}]]
    result=_digital_claim(chats,None,None,"cpu")
    assert result["digital_overlay"] and not result["uncertain"]
    assert result["evidence_origin"]=="independent_visible_observation"
