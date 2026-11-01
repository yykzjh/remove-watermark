from copy import deepcopy

import pytest
from PIL import Image

from watermark_studio.config import Region
from watermark_studio.text_free_review import reconsider_text_free


def fixture(monkeypatch):
    from watermark_studio import detection, quality
    source = Image.new("RGB", (512, 512), "red")
    output = Image.new("RGB", source.size, "green")
    verdict = {"status": "review", "watermark_removed": False, "natural": True,
               "content_preserved": True, "uncertain": False, "guard_codes": [],
               "outside_repair_change": {"tiles": 0}, "residual_regions": [],
               "evidence_conflict": True, "source_regions": [{"kind": "panel", "box": [0, .7, .4, 1]}],
               "model_explanation": {"status": "ok", "assessment": "no_clear_defect", "findings": []}}
    context = {"background_preserved": True, "source_scene": "outdoor"}
    editor, validator = object(), object()
    calls = []
    answers = [
        {"watermark_removed": True, "natural": True, "text_free": True, "uncertain": False, "area": "none"},
        {"watermark_removed": True, "natural": True, "text_free": True, "uncertain": False, "area": "none"},
        {"content_preserved": True, "natural": True, "uncertain": False, "area": "none"},
    ]
    def vqa(chats, fields, model, *args):
        calls.append((chats, fields, model))
        return [answers[len(calls)-1]], .1
    monkeypatch.setattr(quality, "_structured_vqa", vqa)
    monkeypatch.setattr(quality, "output_detail_views", lambda a, b, *args: ([b.copy()], [[0, 0, 1, 1]]))
    monkeypatch.setattr(detection, "locate_watermarks_batch", lambda *a: [([], {})])
    return source, output, verdict, context, editor, validator, calls, answers


def run(f, relation="different_weights"):
    source, output, verdict, context, editor, validator, *_ = f
    reconsider_text_free(source, output, verdict, editor, None, validator, None, "cpu", context, relation)


def test_only_two_independent_output_proofs_reference_and_fresh_scan_may_correct(monkeypatch):
    f = fixture(monkeypatch); source, output, q, _, editor, validator, calls, _ = f
    original = (source.tobytes(), output.tobytes())
    run(f)
    assert q["status"] == "passed"
    assert q["pre_text_free_review"]["status"] == "review"
    assert [c[2] for c in calls] == [editor, validator, editor]
    for chats, _, _ in calls[:2]:
        photos = [item["image"] for item in chats[0][0]["content"] if item["type"] == "image"]
        assert photos and all(photo.getpixel((0, 0)) == (0, 128, 0) for photo in photos)
    assert "watermark_removed" not in calls[2][1]
    assert "content_preserved" in calls[2][1]
    assert original == (source.tobytes(), output.tobytes())


@pytest.mark.parametrize("change", [
    {"natural": False}, {"content_preserved": False}, {"uncertain": True},
    {"guard_codes": ["repair_rectangle_phase"]}, {"guard_codes": ["generation_reference_drift"]},
    {"residual_regions": [{"kind": "logo", "box": [.8, .8, 1, 1]}]},
    {"outside_repair_change": {"tiles": 1}}, {"outside_repair_change": {}},
    {"evidence_conflict": False}, {"model_explanation": {"assessment": "visible_defect"}},
    {"model_explanation": {"status": "ok", "assessment": "no_clear_defect",
                           "findings": [{"issue": "physical_text"}]}},
])
def test_existing_defect_or_missing_evidence_never_opens_an_inquiry(monkeypatch, change):
    f = fixture(monkeypatch); f[2].update(change)
    run(f)
    assert f[2]["status"] == "review" and not f[6]


@pytest.mark.parametrize("change", [{"background_preserved": False}, {"source_scene": "document"}])
def test_unprotected_or_document_output_cannot_use_background_adjudication(monkeypatch, change):
    f = fixture(monkeypatch); f[3].update(change)
    run(f)
    assert f[2]["status"] == "review" and not f[6]


@pytest.mark.parametrize("relation", ["unknown", "same_base_weights", "same_instance"])
def test_different_instances_do_not_prove_independent_weights(monkeypatch, relation):
    f = fixture(monkeypatch)
    run(f, relation)
    assert f[2]["status"] == "review" and not f[6]


@pytest.mark.parametrize("index,change", [
    (0, {"text_free": False}), (1, {"text_free": False}), (0, {"uncertain": True}),
    (1, {"natural": False}), (1, {"watermark_removed": False}),
    (1, {"area": "bottom_left"}), (2, {"natural": False}),
    (2, {"content_preserved": False}), (2, {"uncertain": True}),
])
def test_any_visible_text_uncertainty_or_reference_veto_keeps_review(monkeypatch, index, change):
    f = fixture(monkeypatch); f[7][index].update(change)
    run(f)
    assert f[2]["status"] == "review"


@pytest.mark.parametrize("failed", [False, True])
def test_fresh_unwhitelisted_scan_cannot_be_bypassed(monkeypatch, failed):
    from watermark_studio import detection
    f = fixture(monkeypatch)
    result = RuntimeError("scan failed") if failed else ([Region(kind="logo", box=(.8, .8, 1, 1))], {})
    monkeypatch.setattr(detection, "locate_watermarks_batch", lambda *a: [result])
    run(f)
    assert f[2]["status"] == "review"


def test_explanation_refresh_failure_cannot_leave_a_passed_verdict(monkeypatch):
    from watermark_studio import evidence
    f = fixture(monkeypatch)
    previous = deepcopy(f[2])
    def fail(*args):
        raise RuntimeError("evidence failure")
    monkeypatch.setattr(evidence, "refresh_evidence", fail)
    run(f)
    assert f[2]["status"] == previous["status"] == "review"
    assert f[2]["text_free_review"]["status"] == "error"


def test_target_scope_covers_original_overlays_and_primary_dispute_without_source_pixels(monkeypatch):
    f=fixture(monkeypatch);f[2]["area"]="bottom_left"
    source,output,q,context,editor,validator,calls,_=f
    reconsider_text_free(source,output,q,editor,None,validator,None,"cpu",context,
                         "different_weights",proof_scope="targets")
    assert q["status"]=="passed"
    log=q["text_free_review"]
    assert log["effective_scope"]=="targets"
    assert [0,.7,.4,1] in log["target_boxes"] and len(log["target_boxes"])>=2
    for chats,_,_ in calls[:2]:
        photos=[item["image"] for item in chats[0][0]["content"] if item["type"]=="image"]
        assert all(photo.getpixel((0,0))==(0,128,0) for photo in photos)
    assert "完整输出无任何文字" not in q["model_explanation"]["summary"]


def test_unknown_primary_location_cannot_silently_limit_the_proof_to_known_masks(monkeypatch):
    f=fixture(monkeypatch);f[2]["area"]="none"
    reconsider_text_free(f[0],f[1],f[2],f[4],None,f[5],None,"cpu",f[3],
                         "different_weights",proof_scope="targets")
    assert f[2]["text_free_review"]["effective_scope"]=="global"


@pytest.mark.parametrize("second_area",["none","bottom_left"])
def test_location_contradiction_gets_only_one_same_model_consistency_check(monkeypatch,second_area):
    from watermark_studio import quality
    from watermark_studio.text_free_review import confirmed, inspect_consistently
    flags={"watermark_removed":True,"natural":True,"text_free":True,"uncertain":False}
    responses=iter([flags|{"area":"bottom_left"},flags|{"area":second_area}])
    calls=[]
    def vqa(chats,fields,model,*args):
        calls.append((chats,model));return [next(responses)],.1
    monkeypatch.setattr(quality,"_structured_vqa",vqa)
    model=object();log={}
    chats=[[{"role":"user","content":[{"type":"text","text":"Inspect the actual output"}]}]]
    result=inspect_consistently(chats,tuple(flags),model,None,"cpu",128,log,"own",True)
    assert len(calls)==2 and all(c[1] is model for c in calls)
    assert confirmed(result) is (second_area=="none")
    assert log["own"]["area"]=="bottom_left" and "own_consistency" in log
    assert chats[0][0]["content"]==[{"type":"text","text":"Inspect the actual output"}]
