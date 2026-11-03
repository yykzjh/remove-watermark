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
    proof={"physical_text_only":True,"watermark_removed":True,"natural":True,"uncertain":False,"area":"none"}
    check={"watermark_removed":True,"natural":True,"uncertain":False,"area":"none"}
    witness={"status":"unchanged","text_boxes":[{"box":[.5,.3,.7,.4]}]}
    if mode=="changed_text":witness["status"]="changed_text"
    if mode=="no_text":witness["status"]="no_text_witness"
    if mode=="carrier_uncertain":proof["uncertain"]=True
    if mode=="watermark_remains":proof["watermark_removed"]=False
    if mode=="judge_uncertain":check["uncertain"]=True
    if mode=="source_ambiguity":result["guard_codes"]=["source_physical_ambiguity"]
    if mode=="identity":editor=validator
    monkeypatch.setattr("watermark_studio.quality._structured_vqa",lambda *args:([proof],0))
    monkeypatch.setattr("watermark_studio.grounded_review.inspect_grounded",lambda *args:([check],0))
    monkeypatch.setattr("watermark_studio.detection.locate_watermarks_batch",lambda *args:[
        ([Region(box=(.8,.8,1,1))] if mode=="fresh_residual" else [],"{}")])
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
