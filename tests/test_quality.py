import json

import pytest
from PIL import Image

from watermark_studio.config import ProcessingConfig
from watermark_studio.quality import deterministic_warnings, parse_review, pixel_change


def verdict(**changes):
    return json.dumps({"watermark_removed": True, "natural": True,
                       "content_preserved": True, "uncertain": False, **changes})


def test_pass_requires_all_checks_and_no_uncertainty():
    assert parse_review(verdict())["status"] == "passed"
    for flag in ("watermark_removed", "natural", "content_preserved"):
        assert parse_review(verdict(**{flag: False}))["status"] == "review"
    assert parse_review(verdict(uncertain=True))["status"] == "review"


@pytest.mark.parametrize("text", ['{}', 'not json', '{"watermark_removed":"true"}'])
def test_incomplete_or_invalid_verdict_is_never_a_pass(text):
    with pytest.raises((ValueError, TypeError)):
        parse_review(text)


def test_size_change_alone_has_no_change_in_constant_color():
    assert pixel_change(Image.new("RGB", (400, 300), "blue"),
                        Image.new("RGB", (800, 600), "blue")) == 0


def test_quality_is_opt_in_and_parameters_are_bounded():
    assert not ProcessingConfig().quality_check
    with pytest.raises(ValueError):
        ProcessingConfig(quality_batch_size=16)


def test_unchanged_and_introduced_flat_fill_require_review():
    before = Image.new("RGB", (400, 300), "blue")
    assert deterministic_warnings(before, before.copy())
    after = before.copy()
    from PIL import ImageDraw
    ImageDraw.Draw(after).rectangle((0, 0, 200, 150), fill="lime")
    assert deterministic_warnings(before, after)

def test_detail_crops_include_a_small_local_change():
    from PIL import ImageDraw

    from watermark_studio.quality import detail_sheets
    before = Image.new("RGB", (1024, 768), "gray")
    after = before.copy()
    ImageDraw.Draw(after).rectangle((700, 350, 730, 380), fill="red")
    sheets, boxes = detail_sheets(before, after, 4)
    assert len(sheets) == 2 and len(boxes) == 4
    x0, y0, x1, y1 = boxes[0]
    assert x0 <= 715/1024 <= x1 and y0 <= 365/768 <= y1


def test_review_grammar_only_allows_complete_typed_json():
    from watermark_studio.quality import allowed_review_tokens

    class CharTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

    tree = allowed_review_tokens(CharTokenizer(), [0])
    outputs = []
    def walk(node, prefix):
        for token, child in node.items():
            if token == 0:
                outputs.append(json.loads("".join(map(chr, prefix))))
            else:
                walk(child, prefix+[token])
    walk(tree, [])
    assert len(outputs) == 160
    for row in outputs:
        assert all(type(row[k]) is bool for k in
                   ("watermark_removed", "natural", "content_preserved", "uncertain"))
        assert parse_review(json.dumps(row))["status"] in {"passed", "review"}

def test_protected_composition_does_not_change_pixels_outside_feather():
    from watermark_studio.config import Region
    from watermark_studio.images import composite_protected
    before = Image.new("RGB", (100, 80), "blue")
    generated = Image.new("RGB", before.size, "red")
    result = composite_protected(before, generated, [Region(box=(.2, .2, .4, .4))], feather=3)
    assert result.getpixel((30, 24)) == (255, 0, 0)
    assert result.getpixel((10, 10)) == before.getpixel((10, 10))
    import numpy as np
    changed = np.any(np.array(result) != np.array(before), axis=2)
    changed[13:35, 17:43] = False
    assert not changed.any()

def test_local_guard_detects_subtle_change_outside_removed_overlay():
    from PIL import ImageDraw

    from watermark_studio.config import Region
    from watermark_studio.quality import outside_repair_changes
    before = Image.new("RGB", (1024, 768), "gray")
    after = before.copy()
    ImageDraw.Draw(before).rectangle((20, 550, 300, 750), fill="blue")
    regions = [Region(box=(.01, .70, .32, 1.))]
    assert outside_repair_changes(before, after, regions)["tiles"] == 0
    ImageDraw.Draw(after).text((700, 300), "CAMERA", fill=(160, 160, 160))
    change=outside_repair_changes(before, after, regions)
    assert change["tiles"] > 0
    from watermark_studio.evidence import local_findings
    assert local_findings({"outside_repair_change":change})[0]["box"]


def test_local_guard_tolerates_requested_output_resize_and_jpeg():
    import io

    import numpy as np

    from watermark_studio.quality import outside_repair_changes
    rng = np.random.default_rng(42)
    before = Image.fromarray(rng.integers(0, 256, (768, 1024, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    before.save(buffer, format="JPEG", quality=97, subsampling=0)
    after = Image.open(io.BytesIO(buffer.getvalue()))
    assert outside_repair_changes(before, after, [])["tiles"] == 0
    after = before.resize((512, 384), Image.Resampling.LANCZOS)
    assert outside_repair_changes(before, after, [])["tiles"] == 0

def test_local_guard_accounts_for_actual_composition_feather():
    from watermark_studio.config import Region
    from watermark_studio.images import composite_protected
    from watermark_studio.quality import outside_repair_changes
    before = Image.new("RGB", (512, 384), "blue")
    regions = [Region(box=(.4, .4, .6, .6))]
    after = composite_protected(before, Image.new("RGB", before.size, "red"), regions, feather=64)
    assert outside_repair_changes(before, after, regions, feather=64)["tiles"] == 0
    assert outside_repair_changes(before, after, regions, feather=0)["tiles"] > 0

@pytest.mark.parametrize("orientation", range(1, 9))
def test_quality_size_check_uses_display_orientation(tmp_path, monkeypatch, orientation):
    import shutil
    import sys
    from types import SimpleNamespace

    from watermark_studio.config import DeploymentConfig
    from watermark_studio.worker import process_batch

    source, target = tmp_path/"source.jpg", tmp_path/"output.jpg"
    exif = Image.Exif()
    exif[274] = orientation
    Image.new("RGB", (60, 40), "blue").save(source, exif=exif)
    shutil.copyfile(source, target)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        reset_peak_memory_stats=lambda: None, synchronize=lambda: None,
        max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0,
    )))
    monkeypatch.setattr("watermark_studio.quality.verify_batch",
                        lambda *a, **kw: [{"status": "passed", "reason": ""}])
    pipe = SimpleNamespace(_quality_encoder=None, _quality_processor=None, _execution_device="cpu")
    result = process_batch([{"source": str(source), "destination": str(target),
                             "relative": "source.jpg", "reuse_existing": True}],
                           ProcessingConfig(quality_check=True, scope="full_image"), DeploymentConfig(), pipe)[0]
    assert result["quality"]["status"] == "passed"
    assert result["quality"]["dimensions_match"]
    assert result["quality"]["display_output_size"] == ([40, 60] if orientation >= 5 else [60, 40])
    assert result["output_size"] == [60, 40]

@pytest.mark.parametrize("batch_error",["oom","malformed"])
def test_quality_oom_releases_traceback_before_splitting(tmp_path, monkeypatch,batch_error):
    import sys
    import weakref
    from types import SimpleNamespace

    from watermark_studio.config import DeploymentConfig
    from watermark_studio.worker import process_batch

    class OOM(Exception):
        pass
    class HeldTensor:
        pass
    refs=[]
    def verify(pairs,*a,**kw):
        if len(pairs)>1:
            held=HeldTensor()
            refs.append(weakref.ref(held))
            raise (OOM if batch_error=="oom" else ValueError)("simulated batch failure")
        assert refs[0]() is None
        return [{"status":"passed","reason":""}]
    monkeypatch.setattr("watermark_studio.quality.verify_batch",verify)
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=SimpleNamespace(
        reset_peak_memory_stats=lambda:None,synchronize=lambda:None,
        max_memory_allocated=lambda:0,max_memory_reserved=lambda:0,
        empty_cache=lambda:None,OutOfMemoryError=OOM,
    )))
    tasks=[]
    for name in ("a.png","b.png"):
        path=tmp_path/name
        Image.new("RGB",(64,48),"blue").save(path)
        tasks.append({"source":str(path),"destination":str(path),"relative":name,"reuse_existing":True})
    pipe=SimpleNamespace(_quality_encoder=None,_quality_processor=None,_execution_device="cpu")
    rows=process_batch(tasks,ProcessingConfig(scope="full_image",quality_check=True,quality_batch_size=2),
                       DeploymentConfig(),pipe)
    assert all(r["quality"]["status"]=="passed" and r["quality_oom_splits"]==int(batch_error=="oom") for r in rows)
    assert all(r["quality_validation_splits"]==int(batch_error=="malformed") for r in rows)


def test_material_context_is_per_image_and_does_not_leak_between_documents_and_outdoors(monkeypatch):
    from PIL import Image

    from watermark_studio.quality import verify_batch

    observed=[]
    def structured(chats,fields,*args):
        observed.append(chats)
        return [{field:field!="uncertain" for field in fields}|{"area":"none"} for _ in chats],.01
    monkeypatch.setattr("watermark_studio.quality._structured_vqa",structured)
    monkeypatch.setattr("watermark_studio.detection.locate_watermarks_batch",
                        lambda *a:[([],""),([],"")])
    photo=Image.new("RGB",(200,180),"gray")
    results=verify_batch([(photo,photo.copy()),(photo,photo.copy())],None,None,"cpu",
        detail_patches=0,source_regions=[[],[]],
        editor_context=[{"source_scene":"document"},{"source_scene":"outdoor"}])
    for batch in observed:
        texts=[" ".join(item.get("text","") for item in chat[0]["content"]) for chat in batch]
        assert "完整原图是票据" in texts[0] and "完整原图是票据" not in texts[1]
    assert [r["scene_context"] for r in results]==["document","outdoor"]


def test_low_amplitude_inset_rectangles_are_only_soft_candidates_and_real_existing_edges_are_retained():
    from PIL import ImageDraw

    from watermark_studio.config import Region
    from watermark_studio.quality import repair_boundary_warnings
    source=Image.new("RGB",(600,450),(100,100,100))
    after=source.copy()
    box=(120,180,420,400)
    ImageDraw.Draw(source).rectangle(box,fill="blue")
    ImageDraw.Draw(after).rectangle(box,fill=(94,94,94))
    regions=[Region(box=(.2,.4,.7,400/450))]
    assert repair_boundary_warnings(after,regions,before=source)["count"]==0
    assert repair_boundary_warnings(after,regions,before=source,inset_search=True,minimum_step=4)["count"]>0
    assert repair_boundary_warnings(after,regions,before=after,inset_search=True,minimum_step=4)["count"]==0


@pytest.mark.parametrize("profile",["legacy","untrusted"])
@pytest.mark.parametrize("output_natural,reference_natural",[(True,False),(False,True)])
def test_neither_naturalness_view_can_override_a_rejection_from_the_other(monkeypatch,output_natural,reference_natural,profile):
    from watermark_studio.quality import verify_batch
    def structured(chats,fields,*args):
        natural=reference_natural if "content_preserved" in fields else output_natural
        return [{f:(natural if f=="natural" else f!="uncertain") for f in fields}|{"area":"none"}
                for _ in chats],.01
    monkeypatch.setattr("watermark_studio.quality._structured_vqa",structured)
    monkeypatch.setattr("watermark_studio.quality.deterministic_warnings",lambda *a:[])
    monkeypatch.setattr("watermark_studio.evidence._generate_explanations",lambda chats,*a:[
        {"status":"ok","assessment":"no_clear_defect","summary":"No clear defect","findings":[]} for _ in chats])
    photo=Image.new("RGB",(400,300),"gray")
    q=verify_batch([(photo,photo.copy())],None,None,"cpu",detail_patches=0,source_regions=[[]],
                   inspection_profile=profile)[0]
    assert q["status"]=="review" and not q["natural"]
    assert q["output_checks"]["natural"]==output_natural
    assert q["reference_checks"]["natural"]==reference_natural



def test_tiny_real_edit_is_not_confused_with_identity_but_other_vetoes_still_apply(monkeypatch):
    from PIL import ImageDraw

    from watermark_studio.config import Region
    from watermark_studio.quality import verify_batch
    source=Image.new("RGB",(1800,1400),(70,90,60))
    ImageDraw.Draw(source).rectangle((1700,1320,1715,1330),fill="white")
    clean=Image.new("RGB",source.size,(70,90,60))
    region=Region(kind="logo",box=(.94,.94,.96,.96))
    def inspect(chats,fields,*args):
        return [{**{k:k!="uncertain" for k in fields},"area":"none"} for _ in chats],0
    monkeypatch.setattr("watermark_studio.quality._structured_vqa",inspect)
    monkeypatch.setattr("watermark_studio.evidence.attach_evidence",lambda *args,**kwargs:None)
    monkeypatch.setattr("watermark_studio.detection.locate_watermarks_batch",
                        lambda images,*args:[([],"{}") for _ in images])
    monkeypatch.setattr("watermark_studio.quality.repair_boundary_warnings",lambda *a,**k:{"count":0})
    monkeypatch.setattr("watermark_studio.quality.repair_texture_warnings",lambda *a,**k:{"count":0})
    # Only the former watermark footprint changed; the global cue is diluted.
    assert any(w["code"]=="nearly_unchanged" for w in deterministic_warnings(source,clean))
    q=verify_batch([(source,clean)],None,None,"cpu",detail_patches=0,source_regions=[[region]])[0]
    assert q["localized_edit_support"]["supported"] and q["status"]=="passed"
    assert q.get("residual_regions")==[]  # A new scan is still required before passing.
    monkeypatch.setattr("watermark_studio.quality.repair_texture_warnings",
                        lambda *a,**k:{"count":1,"suspect_segments":[]})
    q=verify_batch([(source,clean)],None,None,"cpu",detail_patches=0,source_regions=[[region]])[0]
    assert q["status"]=="review" and "repair_texture_collapse" in q["guard_codes"]


def test_localized_change_requires_every_small_overlay_not_outside_noise_or_large_masks():
    from PIL import ImageDraw

    from watermark_studio.config import Region
    from watermark_studio.quality import localized_small_edit_support
    before=Image.new("RGB",(1000,800),(80,90,60))
    ImageDraw.Draw(before).rectangle((900,700,919,719),fill="white")
    regions=[Region(kind="logo",box=(.89,.86,.93,.92))]
    outside={"tiles":0,"max_tile_mae":.1}
    assert not localized_small_edit_support(before,before.copy(),regions,outside)["supported"]
    after=Image.new("RGB",before.size,(80,90,60))
    assert localized_small_edit_support(before,after,regions,outside)["supported"]
    unchanged=Region(kind="logo",box=(.01,.01,.03,.03))
    assert not localized_small_edit_support(before,after,regions+[unchanged],outside)["supported"]
    assert not localized_small_edit_support(before,after,regions,{"tiles":1,"max_tile_mae":2})["supported"]
    assert not localized_small_edit_support(before,after,[Region(box=(0,0,.5,.5))],outside)["supported"]
    import numpy as np
    noise=Image.fromarray(np.minimum(np.asarray(before,dtype=np.int16)+1,255).astype(np.uint8))
    assert not localized_small_edit_support(before,noise,regions,
                                            {"tiles":0,"max_tile_mae":.5})["supported"]


@pytest.mark.parametrize("second_error",[False,True])
def test_oversized_residual_uses_one_fresh_full_image_without_clamping_or_implicit_pass(monkeypatch,second_error):
    from watermark_studio.config import Region
    from watermark_studio.detection import DetectionAreaError
    from watermark_studio.quality import retry_oversized_residual
    photo=Image.new("RGB",(1600,1200),"gray")
    original=photo.tobytes()
    region=Region(kind="logo",box=(.8,.8,1,1))
    expected=DetectionAreaError("still oversized") if second_error else ([region],'{}')
    calls=[]
    def locate(images,*args):
        calls.append((images,args))
        return [expected]
    monkeypatch.setattr("watermark_studio.detection.locate_watermarks_batch",locate)
    initial=DetectionAreaError("merged area too large")
    result,audit=retry_oversized_residual(photo,initial,None,None,"cpu",768)
    assert result is expected and len(calls)==1
    assert calls[0][0][0] is photo and calls[0][1][3:6]==(1024,1,False)
    assert audit["area_limits_unchanged"] and audit["attempts"]==1
    assert audit["status"]==("error" if second_error else "valid")
    assert photo.tobytes()==original
    assert retry_oversized_residual(photo,RuntimeError("GPU failure"),None,None,"cpu",768)[1] is None
    assert len(calls)==1


def test_verdict_json_enforces_strict_boolean_and_location_contract():
    from watermark_studio.quality import _verdict_json
    fields=("watermark_removed","natural","uncertain")
    valid='{"watermark_removed":true,"natural":true,"uncertain":false,"area":"none"}'
    assert _verdict_json(valid,fields)["natural"] is True
    assert _verdict_json(chr(96)*3+"json\n"+valid+"\n"+chr(96)*3,fields)==json.loads(valid)
    for text in ('{}','null',valid.replace('true','"true"',1),
                 valid.replace('"none"','"upper-left"'),valid.replace(',"uncertain":false','')):
        with pytest.raises((ValueError,TypeError)):
            _verdict_json(text,fields)


def test_rectangle_phase_warning_remains_a_veto_and_produces_shadow_retry_evidence(monkeypatch):
    from PIL import ImageDraw

    from watermark_studio.config import Region
    from watermark_studio.quality import verify_batch
    def inspect(chats,fields,*args,**kwargs):
        return [{f:(f not in ("watermark_removed","uncertain"))
                 for f in fields}|{"area":"bottom_left"} for _ in chats],.01
    monkeypatch.setattr("watermark_studio.quality._structured_vqa",inspect)
    monkeypatch.setattr("watermark_studio.quality.deterministic_warnings",lambda *a:[])
    monkeypatch.setattr("watermark_studio.quality.repair_boundary_warnings",lambda *a,**k:{"count":0})
    monkeypatch.setattr("watermark_studio.quality.repair_texture_warnings",lambda *a,**k:{"count":0})
    monkeypatch.setattr("watermark_studio.detection.locate_watermarks_batch",lambda images,*a:[([],"{}") for _ in images])
    monkeypatch.setattr("watermark_studio.evidence._generate_explanations",lambda chats,*a:[
        {"status":"ok","assessment":"no_clear_defect","summary":"No clear defect","findings":[]} for _ in chats])
    original=Image.new("RGB",(800,600),(140,140,140))
    result=original.copy()
    ImageDraw.Draw(original).rectangle((160,120,560,480),fill=(30,100,220))
    ImageDraw.Draw(result).rectangle((160,120,560,480),fill=(120,120,120))
    regions=[Region(kind="panel",box=(.162,.156,.738,.844))]
    from watermark_studio import rectangle_phase as phase_module
    original_phase=phase_module.rectangle_phase_tight
    cores=[{"mask":list(regions[0].box),"core":[.2,.2,.7,.8]}]
    seen=[]
    def strict_phase(before,after,masks,source_cores,*args,**kwargs):
        seen.append(source_cores)
        return original_phase(before,after,masks,source_cores,*args,**kwargs)
    monkeypatch.setattr(phase_module,"rectangle_phase_tight",strict_phase)
    q=verify_batch([(original,result)],None,None,"cpu",source_regions=[regions],
                   rectangle_guard=True,cross_review_allowed=[True],weights_relation="different_weights",
                   editor_context=[{"protected_core_boxes":cores}])[0]
    assert seen==[cores] and q["review_version"]=="2026-11-02.source-surface-v13-tight-phase"
    assert q["status"]=="review" and not q["natural"] and q["uncertain"]
    assert "repair_rectangle_phase" in q["guard_codes"]
    assert q["cross_review"]["reason"]=="strict_guard_or_content_veto"
    assert any(item["issue"]=="rectangular_shadow" for item in q["retry_feedback"])


def test_large_repair_keeps_native_texture_views_without_more_images():
    from watermark_studio.config import Region
    from watermark_studio.quality import output_detail_views
    image=Image.new("RGB",(2560,1920),(80,90,70))
    region=Region(kind="panel",box=(0,.6,.42,1))
    views,boxes=output_detail_views(image,image,4,[region])
    assert len(views)==len(boxes)==4
    assert views[1].size==views[2].size==(512,512)
    for view,box in zip(views[1:3],boxes[1:3]):
        assert round((box[2]-box[0])*image.width)==view.width
        assert round((box[3]-box[1])*image.height)==view.height
    assert boxes[1][1]<region.box[1]<boxes[1][3]
    assert boxes[2][1]>=region.box[1]


def test_native_texture_views_respect_zero_and_small_detail_budgets():
    from watermark_studio.config import Region
    from watermark_studio.quality import output_detail_views
    image=Image.new("RGB",(2560,1920),"gray")
    regions=[Region(kind="panel",box=(0,.6,.42,1))]
    for count in (0,1,2,3):
        views,boxes=output_detail_views(image,image,count,regions)
        assert len(views)==len(boxes)==count


def test_raw_generation_drift_cannot_pass_after_plausible_colour_composition(monkeypatch):
    from watermark_studio.config import Region
    from watermark_studio.quality import verify_batch
    photo=Image.new("RGB",(400,300),"gray")
    def inspect(chats,fields,*args):
        return [{**{k:k!="uncertain" for k in fields},"area":"none"} for _ in chats],0
    monkeypatch.setattr("watermark_studio.quality._structured_vqa",inspect)
    monkeypatch.setattr("watermark_studio.quality.deterministic_warnings",lambda *a:[])
    monkeypatch.setattr("watermark_studio.evidence.attach_evidence",lambda *a,**k:None)
    monkeypatch.setattr("watermark_studio.detection.locate_watermarks_batch",
        lambda images,*a:[([],"{}") for _ in images])
    context={"harmonization":{"correction_withheld":True,"reference_over_64_fraction":.95}}
    q=verify_batch([(photo,photo.copy())],None,None,"cpu",detail_patches=0,
        source_regions=[[Region(box=(0,.6,.4,1))]],editor_context=[context])[0]
    assert q["status"]=="review" and q["natural"] is False
    assert "generation_reference_drift" in q["guard_codes"]
    assert q["generation_reference_drift"]["correction_withheld"] is True


def test_concise_whole_prompt_is_neutral_and_still_constrains_visible_content():
    from watermark_studio.config import ProcessingConfig
    from watermark_studio.prompts import build_full_prompt
    assert ProcessingConfig(whole_prompt_profile="concise").whole_prompt_profile=="concise"
    prompt=build_full_prompt(profile="concise")
    assert "信息卡底板" in prompt and "真实物体上的文字不变" in prompt
    assert "草地" not in prompt and "树木" not in prompt
    assert len(prompt)<150


def test_reference_native_pair_retains_original_pixel_scale_and_context():
    from watermark_studio.config import Region
    from watermark_studio.quality import reference_inspection
    before=Image.new("RGB",(2560,1920),(130,110,90))
    after=before.copy()
    region=Region(box=(0,.6,.42,1))
    chat=reference_inspection(before,after,[region],768,4,"outdoor")[0]["content"]
    native=[c["image"] for c in chat if c["type"]=="image" and c["image"].size==(512,512)]
    assert len(native)==2
    assert native[0].getpixel((0,0))==(130,110,90)
    assert any("原始像素尺度" in c.get("text","") for c in chat)


def test_generation_drift_produces_specific_trusted_material_feedback():
    from watermark_studio.evidence import feedback_from_quality, retry_guidance
    q={"guard_codes":["generation_reference_drift"],
       "source_regions":[{"kind":"panel","box":[0,.6,.4,1]}]}
    findings=feedback_from_quality(q)
    assert findings==[{"issue":"material_mismatch","box":[0,.6,.4,1]}]
    assert "材质类别" in retry_guidance(findings)


def test_readonly_worker_cannot_forget_raw_drift_when_vlm_accepts(monkeypatch,tmp_path):
    import sys
    from types import SimpleNamespace

    from watermark_studio.config import DeploymentConfig, ProcessingConfig
    from watermark_studio.worker import process_batch
    source=tmp_path/"source.png";output=tmp_path/"output.png"
    Image.new("RGB",(200,160),"gray").save(source)
    Image.new("RGB",(200,160),"gray").save(output)
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=SimpleNamespace(
        reset_peak_memory_stats=lambda:None,synchronize=lambda:None,
        max_memory_allocated=lambda:0,max_memory_reserved=lambda:0)))
    def inspect(chats,fields,*args):
        return [{**{k:k!="uncertain" for k in fields},"area":"none"} for _ in chats],0
    monkeypatch.setattr("watermark_studio.quality._structured_vqa",inspect)
    monkeypatch.setattr("watermark_studio.evidence.attach_evidence",lambda *a,**kw:None)
    monkeypatch.setattr("watermark_studio.detection.locate_watermarks_batch",
        lambda images,*a,**kw:[([],"{}") for _ in images])
    drift={"correction_withheld":True,"reference_over_64_fraction":.95}
    task={"source":str(source),"destination":str(output),"relative":"source.png",
          "reuse_existing":True,"_repair_provenance":{"regions":[{"kind":"panel","box":[0,.6,.4,1]}],
          "scene":"outdoor","harmonization":drift}}
    pipe=SimpleNamespace(_quality_encoder=None,_quality_processor=None,_execution_device="cpu",
                         text_encoder=None,processor=None)
    result=process_batch([task],ProcessingConfig(quality_check=True,scope="full_image",
        preserve_background=True),DeploymentConfig(),pipe)[0]
    assert result["harmonization"]==drift
    assert result["quality"]["status"]=="review"
    assert "generation_reference_drift" in result["quality"]["guard_codes"]


def test_neutral_full_prompt_keeps_precise_targets_without_default_landscape():
    from watermark_studio.config import ProcessingConfig
    from watermark_studio.prompts import build_full_prompt
    assert ProcessingConfig(whole_prompt_profile="neutral").whole_prompt_profile=="neutral"
    prompt=build_full_prompt(profile="neutral")
    for text in ("蓝底白字","灰白半透明底黑字","今日水印","防伪码","衣物褶皱","真实","人物身份"):
        assert text in prompt
    assert "草地" not in prompt and "树木" not in prompt
    assert "同类材质" in prompt and "不替换背景类型" in prompt


def test_untrusted_reference_locations_never_assert_that_the_entire_box_is_watermarked():
    from watermark_studio.config import Region
    from watermark_studio.quality import reference_inspection

    photo = Image.new("RGB", (400, 300), "gray")
    regions = [Region(kind="panel", box=(0, .06, .74, 1))]
    def text(chat):
        return "\n".join(item["text"] for item in chat[0]["content"] if item["type"] == "text")
    legacy = text(reference_inspection(photo, photo.copy(), regions, 768, 4, "outdoor"))
    untrusted = text(reference_inspection(photo, photo.copy(), regions, 768, 4, "outdoor",
                                          untrusted_regions=True))
    assert "水印存在是输入事实" in legacy
    assert "水印存在是输入事实" not in untrusted
    assert "禁止把上游定位当作删改许可" in untrusted
    assert "真实内容位于候选框内" in untrusted
