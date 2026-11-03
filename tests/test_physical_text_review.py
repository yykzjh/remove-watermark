from copy import deepcopy
from types import SimpleNamespace

import pytest
from PIL import Image

from watermark_studio.config import Region
from watermark_studio.physical_text_review import reconsider


def verdict():
    return {"status":"review","watermark_removed":False,"natural":True,"content_preserved":True,
        "uncertain":False,"guard_codes":[],"outside_repair_change":{"tiles":0},"residual_regions":[],
        "evidence_conflict":True,"source_regions":[Region(box=(0,.6,.4,1)).model_dump()],
        "model_explanation":{"status":"ok","assessment":"no_clear_defect","findings":[]},"seconds":1}


def run(monkeypatch,mode="success"):
    result=verdict();initial=deepcopy(result);photo=Image.new("RGB",(400,300),"gray")
    context={"background_preserved":True,"source_scene":"outdoor"};editor=object();validator=object()
    check={"watermark_removed":True,"natural":True,"uncertain":False,"area":"none"}
    witness={"status":"unchanged","text_boxes":[{"box":[.5,.3,.7,.4]}]}
    if mode=="changed_text":witness["status"]="changed_text"
    if mode=="no_text":witness["status"]="no_text_witness"
    if mode=="judge_uncertain":check["uncertain"]=True
    if mode=="physical_location":check["area"]="center"
    if mode=="source_ambiguity":result["guard_codes"]=["source_physical_ambiguity"]
    if mode=="identity":editor=validator
    monkeypatch.setattr("watermark_studio.grounded_review.inspect_grounded",lambda *args:([check],0))
    monkeypatch.setattr("watermark_studio.text_preservation.enhance_text_contrast",lambda image:image.copy())
    def scan(images,encoder,*args):
        if mode=="carrier_uncertain" and encoder is editor:
            return [ValueError("Indeterminate native scan") for _ in images]
        present=(mode=="watermark_remains" and encoder is editor) or (mode=="fresh_residual" and encoder is validator)
        return [([Region(box=(.8,.8,1,1))] if present else [],"{}") for _ in images]
    monkeypatch.setattr("watermark_studio.detection.locate_watermarks_batch",scan)
    def refresh(value):
        if mode=="refresh_error":raise RuntimeError("evidence refresh failed")
    monkeypatch.setattr("watermark_studio.evidence.refresh_evidence",refresh)
    reconsider(photo,photo.copy(),result,editor,None,validator,None,"cpu",context,"different_weights",
        SimpleNamespace(inspect=lambda *args:witness))
    return result,initial


def test_supported_dispute_keeps_original_verdict_and_full_evidence(monkeypatch):
    result,initial=run(monkeypatch)
    assert result["status"]=="passed" and result["watermark_removed"] is True
    assert result["pre_physical_text_review"]["watermark_removed"] is False
    assert result["physical_text_review"]["status"]=="corrected"
    assert result["physical_text_review"]["fresh_unwhitelisted_residuals"]==[]
    for key in ("natural","content_preserved","uncertain","guard_codes","outside_repair_change"):
        assert result[key]==initial[key]


@pytest.mark.parametrize("mode",["changed_text","no_text","carrier_uncertain","watermark_remains",
    "judge_uncertain","source_ambiguity","identity","fresh_residual","refresh_error"])
def test_incomplete_or_conflicting_witnesses_cannot_approve(monkeypatch,mode):
    result,_=run(monkeypatch,mode)
    assert result["status"]=="review" and result["watermark_removed"] is False
    assert result["physical_text_review"]["status"]!="corrected"

def test_deployment_inquiry_is_opt_in_and_requires_source_and_independent_checks():
    from watermark_studio.config import DeploymentConfig
    assert DeploymentConfig().quality_physical_text_review is False
    with pytest.raises(ValueError,match="Physical text review"):
        DeploymentConfig(quality_physical_text_review=True)
    cfg=DeploymentConfig(quality_physical_text_review=True,quality_verifier=True,
        quality_backend="independent",protection_physical_vet=True)
    assert cfg.quality_physical_text_review is True


def test_quality_inquiry_hook_is_disabled_by_default(monkeypatch):
    from watermark_studio.quality import verify_batch
    calls=[]
    def structured(chats,fields,*args):
        return [{**{k:k!="uncertain" for k in fields},"area":"none"} for _ in chats],0
    monkeypatch.setattr("watermark_studio.quality._structured_vqa",structured)
    monkeypatch.setattr("watermark_studio.quality.deterministic_warnings",lambda *a:[])
    monkeypatch.setattr("watermark_studio.evidence.attach_evidence",lambda *a,**k:None)
    monkeypatch.setattr("watermark_studio.detection.locate_watermarks_batch",lambda *a:[([],"{}")])
    monkeypatch.setattr("watermark_studio.physical_text_review.reconsider",lambda *args:calls.append(args))
    image=Image.new("RGB",(400,300),"gray");context={"background_preserved":True,"source_scene":"outdoor"}
    for enabled in (False,True):
        verify_batch([(image,image.copy())],None,None,"cpu",detail_patches=0,
            source_regions=[[]],editor_context=[context],physical_text_review=enabled)
        assert len(calls)==int(enabled)
    assert calls[0][8]==context

def test_carrier_context_keeps_line_and_surrounding_object_geometry():
    from watermark_studio.physical_text_review import carrier_context_box
    for box in [(20,20,220,40),(780,570,995,595),(0,0,15,8)]:
        result=carrier_context_box(box,(1000,600))
        l,t,r,b=result
        assert 0<=l<=box[0]<box[2]<=r<=1000 and 0<=t<=box[1]<box[3]<=b<=600
        assert r-l==b-t

def test_nondefect_object_location_does_not_change_positive_quality_flags(monkeypatch):
    result,_=run(monkeypatch,"physical_location")
    assert result["status"]=="passed"
    assert result["physical_text_review"]["independent_validator_output"]["area"]=="center"
    assert result["area"]=="none"
