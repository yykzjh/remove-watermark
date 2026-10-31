import json
from copy import deepcopy

import pytest
from PIL import Image, ImageDraw

from watermark_studio.config import ProcessingConfig, Region
from watermark_studio.cross_review import parse_classification, preserved_patch, reconsider


def report(**changes):
    return {"physical_scene_text":True,"digital_overlay":False,"object_attachment":True,
            "surface_evidence":True,"uncertain":False,"evidence":"字样位于机械金属表面，随表面倾斜且与物体边缘连接。",**changes}


def fixture():
    before=Image.new("RGB",(400,300),"gray")
    ImageDraw.Draw(before).rectangle((0,220,130,299),fill="blue")
    after=before.copy();ImageDraw.Draw(after).rectangle((0,220,130,299),fill="gray")
    q={"status":"review","watermark_removed":False,"natural":True,"content_preserved":True,
       "uncertain":False,"reason":"疑似文字水印","seconds":1,
       "source_regions":[{"kind":"panel","box":[0,.72,.34,1]}],
       "residual_regions":[{"kind":"logo","box":[.55,.3,.7,.45]}],
       "model_explanation":{"status":"ok","summary":"疑似残留","findings":[]}}
    return before,after,q


def calls(monkeypatch, reports=None, residuals=(), recheck=None):
    history=[]
    reports=reports or [report(),report()]
    def classify(chats,encoder,*args):
        history.append(encoder)
        return [deepcopy(reports[len(history)-1])]
    monkeypatch.setattr("watermark_studio.cross_review._classify",classify)
    monkeypatch.setattr("watermark_studio.cross_review._reinspect",lambda *a:
        recheck or {"watermark_removed":True,"natural":True,"uncertain":False,"area":"none"})
    monkeypatch.setattr("watermark_studio.detection.locate_watermarks_batch",lambda *a:[(list(residuals),"scan")])
    return history


def run(before,after,q,editor=None,validator=None):
    editor=editor or object();validator=validator or object()
    reconsider(before,after,q,editor,None,validator,None,"cpu")


def test_corrected_false_alarm_keeps_original_verdict_and_observable_reports(monkeypatch):
    before,after,q=fixture(); history=calls(monkeypatch)
    run(before,after,q)
    assert len(history)==2 and history[0] is not history[1]
    assert q["status"]=="passed" and q["watermark_removed"] is True
    assert q["cross_review"]["status"]=="corrected"
    assert q["cross_review"]["pixel_evidence"][0]["unchanged"]
    assert q["pre_cross_review"]["watermark_removed"] is False
    assert q["model_explanation"]["status"]=="adjudicated"
    assert q["evidence"][0]["issue"]=="physical_text"
    assert not q["retry_feedback"] and "retry_recheck_only" not in q


@pytest.mark.parametrize("code",["repair_boundary_step","repair_texture_collapse",
                                 "outside_repair_change","new_flat_fill","identical"])
def test_numeric_guards_are_never_overridden(monkeypatch,code):
    before,after,q=fixture(); history=calls(monkeypatch)
    q["guard_codes"]=[code]
    run(before,after,q)
    assert q["status"]=="review" and not history
    assert q["cross_review"]["attempted"] is False


@pytest.mark.parametrize("field",["natural","content_preserved","uncertain"])
def test_ambiguous_or_changed_content_blocks_reconsideration(monkeypatch,field):
    before,after,q=fixture();history=calls(monkeypatch)
    q[field]=field=="uncertain"
    run(before,after,q)
    assert q["status"]=="review" and not history


def test_original_unchanged_digital_stamp_inside_known_overlay_cannot_be_exempted(monkeypatch):
    before,after,q=fixture();history=calls(monkeypatch)
    q["residual_regions"]=[{"kind":"logo","box":[.1,.8,.2,.9]}]
    run(before,after,q)
    assert q["status"]=="review" and not history
    assert q["cross_review"]["reason"]=="overlaps_original_overlay_or_feather"


def test_changed_text_and_different_output_size_are_not_proof(monkeypatch):
    before,after,q=fixture();history=calls(monkeypatch)
    ImageDraw.Draw(after).rectangle((220,90,280,135),fill="white")
    run(before,after,q)
    assert q["status"]=="review" and not history
    assert not preserved_patch(before,after.resize((200,150)),[.5,.2,.7,.4])["unchanged"]


def test_editor_claim_alone_cannot_overrule_independent_digital_or_uncertain_result(monkeypatch):
    for objection in (report(physical_scene_text=False,digital_overlay=True),report(uncertain=True)):
        before,after,q=fixture();history=calls(monkeypatch,[report(),objection])
        run(before,after,q)
        assert len(history)==2 and q["status"]=="review"
        assert q["retry_recheck_only"] is True


def test_fresh_full_image_scan_never_whitelists_even_confirmed_dispute_regions(monkeypatch):
    before,after,q=fixture()
    calls(monkeypatch,residuals=[Region(kind="logo",box=(.55,.3,.7,.45))])
    run(before,after,q)
    assert q["status"]=="review" and q["cross_review"]["reason"]=="other_residual_overlays"
    assert q["retry_recheck_only"] is True


def test_reinspection_rejection_is_retained(monkeypatch):
    before,after,q=fixture()
    calls(monkeypatch,recheck={"watermark_removed":False,"natural":True,"uncertain":False})
    run(before,after,q)
    assert q["status"]=="review" and q["cross_review"]["reason"]=="global_reinspection_rejected"
    assert q["retry_recheck_only"] is True


def test_same_model_cannot_supply_independent_confirmation(monkeypatch):
    before,after,q=fixture();history=calls(monkeypatch)
    model=object();run(before,after,q,model,model)
    assert q["status"]=="review" and not history


@pytest.mark.parametrize("changes",[{"uncertain":"false"},{"evidence":""},{"surface_evidence":None}])
def test_invalid_classifier_json_never_qualifies_as_evidence(changes):
    with pytest.raises((ValueError,TypeError)):
        parse_classification(json.dumps(report(**changes)))


def test_cross_review_defaults_false_and_requires_quality_check():
    assert ProcessingConfig().quality_cross_review is False
    with pytest.raises(ValueError):
        ProcessingConfig(quality_cross_review=True)


def test_recovering_a_missing_flag_uses_new_model_judgment_not_a_default(monkeypatch):
    from watermark_studio.cross_review import _recover_classification

    def structured(*args):
        return [report(physical_scene_text=False,digital_overlay=True,uncertain=True)],.1
    monkeypatch.setattr("watermark_studio.quality._structured_vqa",structured)
    result=_recover_classification([{ "role":"user","content":[{"type":"text","text":"classify"}]}],
        '{"physical_scene_text":true,"evidence":"文字位于载体表面"}',"missing fields",None,None,"cpu")
    assert result["format_recovered"]
    assert result["physical_scene_text"] is False and result["digital_overlay"] is True


def test_corrected_claim_requires_actual_observable_explanation(monkeypatch):
    from watermark_studio.cross_review import _recover_classification

    monkeypatch.setattr("watermark_studio.quality._structured_vqa",lambda *a:([report()],.1))
    monkeypatch.setattr("watermark_studio.evidence._generate_explanations",lambda *a:[
        {"status":"error","error":"no visible evidence"}])
    with pytest.raises(ValueError,match="observable explanation"):
        _recover_classification([{ "role":"user","content":[{"type":"text","text":"classify"}]}],
                                '{}',"missing fields",None,None,"cpu")


def test_production_cross_review_requires_an_independent_validator(tmp_path):
    from watermark_studio.config import DeploymentConfig, JobRequest
    from watermark_studio.runtime import Runtime

    runtime=Runtime(DeploymentConfig(state_dir=str(tmp_path/"state"),quality_verifier=True,
                                     quality_backend="resident"))
    runtime.stage="ready"
    request=JobRequest(input_dir=str(tmp_path),output_dir=str(tmp_path/"out"),
        options=ProcessingConfig(quality_check=True,quality_cross_review=True,concurrency=1))
    with pytest.raises(ValueError,match="independent"):
        runtime.submit(request)


def test_a_vague_dispute_requires_precise_new_evidence_before_full_revalidation(monkeypatch):
    before,after,q=fixture()
    q.update(residual_regions=[],evidence=[],evidence_conflict=True,area="top_right")
    history=calls(monkeypatch)
    monkeypatch.setattr("watermark_studio.cross_review.discover_dispute",lambda *a:{
        "status":"ok","findings":[{"issue":"physical_text","box":[.55,.3,.7,.45]}]})
    run(before,after,q)
    assert q["status"]=="passed" and len(history)==2
    assert q["cross_review"]["localization"]["findings"]


def test_vague_no_defect_explanation_never_passes_without_a_precise_safe_location(monkeypatch):
    for findings in ([],[{"issue":"physical_text","box":[.1,.8,.2,.9]}]):
        before,after,q=fixture()
        q.update(residual_regions=[],evidence=[],evidence_conflict=True,area="top_right")
        history=calls(monkeypatch)
        monkeypatch.setattr("watermark_studio.cross_review.discover_dispute",
                            lambda *a, findings=findings:{"status":"ok","findings":findings})
        run(before,after,q)
        assert q["status"]=="review" and not history
        assert q["cross_review"]["attempted"] is True
