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
    monkeypatch.setattr("watermark_studio.cross_review._reinspect",lambda *a,**kw:
        recheck or {"watermark_removed":True,"natural":True,"uncertain":False,"area":"none"})
    monkeypatch.setattr("watermark_studio.detection.locate_watermarks_batch",lambda *a:[(list(residuals),"scan")])
    return history


def run(before,after,q,editor=None,validator=None,weights_relation="different_weights",full_context=False):
    editor=editor or object();validator=validator or object()
    reconsider(before,after,q,editor,None,validator,None,"cpu",weights_relation=weights_relation,full_context=full_context)


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


def identity_fixture():
    before = Image.new("RGB", (400,300), "gray")
    q = {"status":"review","watermark_removed":False,"natural":True,"content_preserved":True,
         "uncertain":False,"reason":"possible overlay","seconds":1,"source_regions":[],
         "residual_regions":[],"guard_codes":["identical"],
         "evidence":[{"issue":"physical_text","box":[.55,.3,.7,.45]}],
         "model_explanation":{"status":"ok","summary":"physical branding","findings":[]}}
    return before, before.copy(), q


def test_clean_identity_requires_other_source_scan_and_every_existing_physical_proof(monkeypatch):
    before, after, q = identity_fixture()
    history = calls(monkeypatch)
    run(before, after, q)
    assert q["status"] == "passed" and len(history) == 2
    assert q["cross_review"]["editor_source_regions"] == []
    assert q["cross_review"]["clean_identity_confirmed"]
    assert q["guard_codes"] == []
    assert q["pre_cross_review"]["guard_codes"] == ["identical"]
    assert q["judgment_basis"]["local_guard_codes"] == []


@pytest.mark.parametrize("change", [
    {"source_regions":[{"kind":"logo","box":[.55,.3,.7,.45]}]},
    {"residual_regions":[{"kind":"logo","box":[.55,.3,.7,.45]}]},
    {"guard_codes":["identical","outside_repair_change"]},
    {"guard_codes":["nearly_unchanged"]},
    {"natural":False}, {"content_preserved":False}, {"uncertain":True},
])
def test_identity_never_exempts_detected_stamps_or_real_defect_guards(monkeypatch, change):
    before, after, q = identity_fixture()
    history = calls(monkeypatch)
    q.update(change)
    run(before,after,q)
    assert q["status"] == "review" and not history


def test_identity_editor_source_detection_or_error_keeps_refusal(monkeypatch):
    for result in ([Region(kind="panel",box=(.1,.1,.3,.3))], RuntimeError("scan failure")):
        before, after, q = identity_fixture()
        history = calls(monkeypatch)
        monkeypatch.setattr("watermark_studio.detection.locate_watermarks_batch",
                            lambda *a, result=result: [result if isinstance(result,Exception) else (result,"scan")])
        run(before,after,q)
        assert q["status"] == "review" and not history and q["guard_codes"] == ["identical"]


def test_identity_without_carrier_agreement_or_final_empty_scan_never_passes(monkeypatch):
    before, after, q = identity_fixture()
    calls(monkeypatch, [report(), report(digital_overlay=True, physical_scene_text=False)])
    run(before,after,q)
    assert q["status"] == "review" and q["guard_codes"] == ["identical"]


def test_clean_identity_still_requires_exact_whole_photo_equality(monkeypatch):
    before, after, q = identity_fixture()
    after.putpixel((0,0), (127,128,128))
    history = calls(monkeypatch)
    run(before,after,q)
    assert q["status"] == "review" and not history


def test_clean_identity_final_scan_cannot_hide_any_remaining_overlay(monkeypatch):
    before, after, q = identity_fixture()
    calls(monkeypatch)
    scans = iter([([], "source"), ([Region(kind="logo",box=(.55,.3,.7,.45))], "output")])
    monkeypatch.setattr("watermark_studio.detection.locate_watermarks_batch", lambda *a: [next(scans)])
    run(before,after,q)
    assert q["status"] == "review" and q["guard_codes"] == ["identical"]
    assert q["cross_review"]["reason"] == "other_residual_overlays"


@pytest.mark.parametrize("full_context",[False,True])
def test_dispute_locator_maps_crop_coordinates_in_code_using_actual_pixel_bounds(monkeypatch,full_context):
    from watermark_studio.cross_review import discover_dispute
    captured = []
    def explain(chats, *args):
        captured.extend(chats)
        return [{"status":"ok","assessment":"uncertain","summary":"metal label",
                 "findings":[{"issue":"physical_text","box":[.2,.1,.5,.3],
                              "observation":"printing follows metal surface"}]}]
    monkeypatch.setattr("watermark_studio.evidence._generate_explanations", explain)
    source=Image.new("RGB",(401,307),"gray")
    result_image=Image.new("RGB",source.size,"green")
    result=discover_dispute(source,result_image,"center",None,None,"cpu",full_context=full_context)
    l,t,r,b=result["crop_in_photo"]
    assert result["findings"][0]["box"] == pytest.approx(
        [l+.2*(r-l),t+.1*(b-t),l+.5*(r-l),t+.3*(b-t)],abs=1e-6)
    assert result["local_findings"][0]["box"] == [.2,.1,.5,.3]
    assert result["coordinate_source"] == "crop_local_mapped_by_program"
    text=" ".join(item.get("text","") for item in captured[0][0]["content"])
    assert "当前局部图像0..1000" in text and "不要使用完整照片坐标" in text
    images=[item["image"] for item in captured[0][0]["content"] if item["type"]=="image"]
    assert len(images)==(2 if full_context else 1)
    assert all(image.getpixel((0,0))==(0,128,0) for image in images)
    assert "只定位当前结果中实际可见" in text


@pytest.mark.parametrize("relation", ["unknown", "same_base_weights", "same_instance"])
def test_distinct_python_instances_do_not_prove_different_checkpoint_weights(monkeypatch, relation):
    before, after, q = fixture()
    history = calls(monkeypatch)
    run(before, after, q, weights_relation=relation)
    assert q["status"] == "review" and not history
    assert q["cross_review"]["reason"] == "different_checkpoint_weights_required"


@pytest.mark.parametrize("relation", ["unknown", "same_base_weights"])
def test_production_rejects_cross_review_for_same_or_unaudited_weights(tmp_path, relation):
    from watermark_studio.config import DeploymentConfig, JobRequest
    from watermark_studio.runtime import Runtime
    runtime=Runtime(DeploymentConfig(state_dir=str(tmp_path/"state"),quality_verifier=True,
                                     quality_backend="independent"))
    runtime.stage="ready"
    runtime.quality_identity={"relation":relation}
    request=JobRequest(input_dir=str(tmp_path),output_dir=str(tmp_path/"out"),
        options=ProcessingConfig(quality_check=True,quality_cross_review=True,concurrency=1))
    with pytest.raises(ValueError,match="different|不同"):
        runtime.submit(request)


@pytest.mark.parametrize("code",["repair_boundary_step","repair_texture_collapse","repair_rectangle_phase",
                                 "outside_repair_change","new_flat_fill","source_physical_ambiguity"])
def test_extra_global_context_cannot_override_any_artifact_or_source_veto(monkeypatch,code):
    before,after,q=fixture();history=calls(monkeypatch)
    q["guard_codes"]=[code]
    run(before,after,q,full_context=True)
    assert q["status"]=="review" and not history

def test_full_context_keeps_both_carrier_proofs_and_unwhitelisted_residual_scan(monkeypatch):
    from watermark_studio.cross_review import dispute_chats
    before,after,q=fixture();history=calls(monkeypatch)
    chats=dispute_chats(before,after,[[.55,.3,.7,.45]],full_context=True)
    content=chats[0][0]["content"]
    assert sum(x["type"]=="image" for x in content)==3
    assert "卷尺刻度" in content[-1]["text"] and "uncertain" in content[-1]["text"]
    run(before,after,q,full_context=True)
    assert len(history)==2 and history[0] is not history[1] and q["status"]=="passed"
    before,after,q=fixture()
    calls(monkeypatch,residuals=[Region(kind="logo",box=(.55,.3,.7,.45))])
    run(before,after,q,full_context=True)
    assert q["status"]=="review" and q["cross_review"]["reason"]=="other_residual_overlays"


def test_output_localizer_cannot_use_entire_context_as_precise_text_evidence(monkeypatch):
    from watermark_studio.cross_review import discover_dispute
    monkeypatch.setattr("watermark_studio.evidence._generate_explanations",lambda *a:[{
        "assessment":"uncertain","findings":[{"issue":"physical_text","box":[0,0,1,1]}]}])
    photo=Image.new("RGB",(1000,800),"gray")
    result=discover_dispute(photo,photo,"top_left",None,None,"cpu")
    assert result["findings"]==[]
    assert result["rejected_locations"][0]["reason"]=="imprecise_crop_coverage"


def test_untrusted_protocol_preserves_meter_carrier_rules_in_global_reinspection(monkeypatch):
    from watermark_studio.cross_review import _reinspect

    captured = []
    def structured(chats, fields, *args):
        captured.extend(item["text"] for item in chats[0][0]["content"] if item["type"] == "text")
        return [{f: f != "uncertain" for f in fields} | {"area": "none"}], .1
    monkeypatch.setattr("watermark_studio.quality._structured_vqa", structured)
    source = Image.new("RGB", (400,300), "gray")
    _reinspect(source, source.copy(), [[.3,.3,.5,.5]], None, None, "cpu", 768, 128,
               inspection_profile="untrusted")
    assert any("机械翻字计数窗" in text for text in captured)
    assert any("不是豁免清单" in text for text in captured)


@pytest.mark.parametrize("full_context,expected",[(False,1),(True,2)])
def test_full_context_can_locate_two_distinct_carrier_texts_without_changing_default(monkeypatch,full_context,expected):
    from watermark_studio.cross_review import discover_dispute
    prompt = []
    def generate(chats,*args):
        prompt.extend(item["text"] for item in chats[0][0]["content"] if item["type"]=="text")
        return [{"assessment":"uncertain","findings":[
            {"issue":"physical_text","box":[.2,.1,.5,.25]},
            {"issue":"physical_text","box":[.2,.3,.5,.45]}]}]
    monkeypatch.setattr("watermark_studio.evidence._generate_explanations",generate)
    image=Image.new("RGB",(1000,800),"gray")
    result=discover_dispute(image,image,"center",None,None,"cpu",full_context=full_context)
    assert len(result["findings"])==expected
    assert any(("findings最多2项" if full_context else "findings最多1项") in text for text in prompt)


def test_full_context_does_not_silently_select_two_from_excessive_locations(monkeypatch):
    from watermark_studio.cross_review import discover_dispute
    monkeypatch.setattr("watermark_studio.evidence._generate_explanations",lambda *a:[{
        "assessment":"uncertain","findings":[
            {"issue":"physical_text","box":[.2,t,.5,t+.1]} for t in (.1,.3,.5)]}])
    image=Image.new("RGB",(1000,800),"gray")
    result=discover_dispute(image,image,"center",None,None,"cpu",full_context=True)
    assert result["findings"]==[] and result["rejected_locations"][-1]["reason"]=="excessive_precise_locations"


@pytest.mark.parametrize("disagrees", ["editor","validator"])
def test_two_locations_require_every_carrier_proof_from_both_models(monkeypatch,disagrees):
    before,after,q=fixture()
    q["residual_regions"].append({"kind":"logo","box":[.75,.3,.9,.45]})
    history=[]
    def classify(chats,*args):
        history.append(len(chats))
        stage="editor" if len(history)==1 else "validator"
        return [report(),report(uncertain=stage==disagrees)]
    monkeypatch.setattr("watermark_studio.cross_review._classify",classify)
    monkeypatch.setattr("watermark_studio.cross_review._reinspect",
                        lambda *a,**kw:pytest.fail("An ambiguous second carrier must prevent global appeal"))
    run(before,after,q,full_context=True)
    assert q["status"]=="review"
    assert history==([2] if disagrees=="editor" else [2,2])



def test_output_only_global_residual_scan_never_sees_original_watermark_pixels(monkeypatch):
    from watermark_studio.cross_review import _reinspect
    captured=[]
    def structured(chats,fields,*args):
        captured.extend(item for item in chats[0][0]["content"] if item["type"]=="image")
        return [{f:f!="uncertain" for f in fields}|{"area":"none"}],.1
    monkeypatch.setattr("watermark_studio.quality._structured_vqa",structured)
    before=Image.new("RGB",(1000,800),"blue")
    after=Image.new("RGB",(1000,800),"green")
    result=_reinspect(before,after,[[.55,.3,.7,.45]],None,None,"cpu",768,128,output_only=True)
    assert result["watermark_removed"] and len(captured)==2
    assert all(item["image"].getpixel((0,0))==(0,128,0) for item in captured)
    assert before.getpixel((0,0))==(0,0,255)
