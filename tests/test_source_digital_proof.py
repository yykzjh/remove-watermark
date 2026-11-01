import hashlib

import pytest
from PIL import Image

from watermark_studio.config import Region
from watermark_studio.gray_conditioning import gray_whole_reference
from watermark_studio.source_digital_proof import certify_gray_regions, digital_overlay


def report(**changes):
    return {"physical_scene_text":False,"digital_overlay":True,"object_attachment":False,
            "surface_evidence":False,"uncertain":False,"evidence":"相机标识悬浮在照片角落，不随下层物体表面透视。"}|changes


@pytest.mark.parametrize("changes",[
    {"physical_scene_text":True},{"object_attachment":True},{"surface_evidence":True},
    {"uncertain":True},{"digital_overlay":False},{"evidence":""},{"digital_overlay":"true"},
])
def test_source_identity_requires_complete_noncontradictory_digital_evidence(changes):
    assert digital_overlay(report()) and not digital_overlay(report(**changes))


def test_every_gray_footer_has_two_source_proofs_and_native_fingerprint(monkeypatch):
    image=Image.new("RGB",(800,600),(80,90,70))
    region=Region(kind="logo",box=(.75,.90,.99,.99))
    cores={region.box:(.77,.92,.98,.98)}
    editor,verifier=object(),object()
    called=[]
    def classify(chats,model,*args):
        called.append(model)
        from watermark_studio.source_digital_proof import SOURCE_DIGITAL_PROMPT
        assert chats[0][0]["content"][-1]["text"]==SOURCE_DIGITAL_PROMPT
        assert "必须uncertain" in SOURCE_DIGITAL_PROMPT and "正面证据" in SOURCE_DIGITAL_PROMPT
        return [report()]
    monkeypatch.setattr("watermark_studio.source_digital_proof._classify",classify)
    proofs=certify_gray_regions(image,[region],cores,[],editor,None,verifier,None,"cpu","different_weights")
    assert called==[editor,verifier] and proofs[0]["approved"] and not proofs[0].get("requires_review")
    original=image.tobytes()
    reference,used=gray_whole_reference(image,[region],cores,[],proofs)
    assert used and reference.getpixel((630,565))==(128,128,128)
    assert reference.getpixel((10,10))==image.getpixel((10,10))
    assert image.tobytes()==original and proofs[0]["source_rgb_sha256"]==hashlib.sha256(original).hexdigest()


def test_a_single_source_disagreement_cannot_erase_or_release_a_footer(monkeypatch):
    image=Image.new("RGB",(800,600),"gray")
    region=Region(kind="logo",box=(.75,.90,.99,.99))
    editor,verifier=object(),object()
    monkeypatch.setattr("watermark_studio.source_digital_proof._classify",
        lambda chats,model,*args:[report(uncertain=model is verifier)])
    proofs=certify_gray_regions(image,[region],{region.box:region.box},[],editor,None,verifier,None,"cpu","different_weights")
    assert not proofs[0]["approved"] and proofs[0]["requires_review"]
    reference,used=gray_whole_reference(image,[region],{region.box:region.box},[],proofs)
    assert reference is image and not used


def test_false_fingerprint_or_forged_digital_permission_is_rejected(monkeypatch):
    image=Image.new("RGB",(800,600),"gray")
    region=Region(kind="logo",box=(.75,.90,.99,.99))
    cores={region.box:region.box}
    monkeypatch.setattr("watermark_studio.source_digital_proof._classify",lambda *a:[report()])
    proof=certify_gray_regions(image,[region],cores,[],object(),None,object(),None,"cpu","different_weights")[0]
    for invalid in (proof|{"source_rgb_sha256":"stale"},proof|{"independent":report(physical_scene_text=True)},
                    proof|{"core":[0,0,1,1]}):
        with pytest.raises(ValueError):
            gray_whole_reference(image,[region],cores,[],[invalid])


def test_missing_raw_core_or_same_weights_blocks_gray_source_edit_without_calling_models(monkeypatch):
    image=Image.new("RGB",(800,600),"gray")
    region=Region(kind="logo",box=(.75,.90,.99,.99))
    monkeypatch.setattr("watermark_studio.source_digital_proof._classify",lambda *a:pytest.fail("No source proof allowed"))
    for cores,relation in (({},"different_weights"),({region.box:region.box},"same_base_weights")):
        proof=certify_gray_regions(image,[region],cores,[],object(),None,object(),None,"cpu",relation)[0]
        assert not proof["approved"] and proof["requires_review"]



@pytest.mark.parametrize("confirmed",[False,True])
def test_gray_worker_covers_footer_or_blocks_unproved_source_even_if_qa_says_passed(tmp_path,monkeypatch,confirmed):
    import json
    import sys
    from types import SimpleNamespace

    from PIL import ImageDraw

    from watermark_studio.config import DeploymentConfig, ProcessingConfig
    from watermark_studio.detection import parse_detection
    from watermark_studio.worker import process_batch
    image=Image.new("RGB",(800,600),(80,90,70))
    ImageDraw.Draw(image).rectangle((640,550,790,590),fill="white")
    raw=json.dumps({"scene":"outdoor","watermarks":[{"kind":"logo","box":[790,900,990,990]}]})
    regions=parse_detection(raw)
    path,target=tmp_path/"source.png",tmp_path/"result.png";image.save(path)
    original=path.read_bytes()
    editor,verifier=object(),object()
    monkeypatch.setattr("watermark_studio.worker.locate_watermarks_batch",lambda *a:[(regions,raw)])
    monkeypatch.setattr("watermark_studio.source_digital_proof._classify",
        lambda *a:[report() if confirmed else report(physical_scene_text=True,digital_overlay=False)])
    monkeypatch.setattr("watermark_studio.batching.dimensions",lambda image,*a:image.size)
    calls=[]
    def generate(pipe,items,*args):
        assert confirmed and len(items)==1 and items[0]["image"].size==image.size
        assert items[0]["image"].getpixel((700,560))==(128,128,128)
        assert items[0]["image"].getpixel((10,10))==image.getpixel((10,10))
        assert items[0]["_semantic_reference"] is None
        calls.append(True)
        return [Image.new("RGB",image.size,(80,90,70))]
    monkeypatch.setattr("watermark_studio.batching.independent_batch",generate)
    monkeypatch.setattr("watermark_studio.quality.verify_batch",lambda *a,**kw:[{"status":"passed"}])
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=SimpleNamespace(
        reset_peak_memory_stats=lambda:None,synchronize=lambda:None,
        max_memory_allocated=lambda:0,max_memory_reserved=lambda:0)))
    pipe=SimpleNamespace(text_encoder=editor,processor=None,_execution_device="cpu",
        _quality_encoder=verifier,_quality_processor=None,_quality_weights_relation="different_weights",
        _source_rim_text_detector=None)
    config=DeploymentConfig(prefill_overlays=True,prefill_style="gray",protect_source_rims=True,
        quality_verifier=True,quality_backend="independent",protection_core_feather=True)
    row=process_batch([{"source":str(path),"destination":str(target),"relative":"source.png"}],
        ProcessingConfig(scope="full_image",preserve_background=True,quality_check=True),config,pipe)[0]
    assert row["ok"] and path.read_bytes()==original
    assert row["gray_source_certificates"][0]["approved"] is confirmed
    if confirmed:
        assert calls==[True] and row["model_inferences"]==1 and row["quality"]["status"]=="passed"
    else:
        assert not calls and row["model_inferences"]==0 and row["quality"]["status"]=="review"
        assert "source_physical_ambiguity" in row["quality"]["guard_codes"]
        assert row["retry_blocked"]=="uncertain_source_carrier" and target.read_bytes()==original


def test_gray_small_raw_core_does_not_change_small_opaque_composition_policy():
    import json

    from watermark_studio.detection import parse_detection, source_core_boxes
    from watermark_studio.source_digital_proof import gray_source_cores
    raw=json.dumps({"watermarks":[{"kind":"logo","box":[790,900,990,990]}]})
    assert not source_core_boxes(raw)
    cores=gray_source_cores(raw)
    assert len(cores)==1
    mask,core=next(iter(cores.items()))
    assert core==parse_detection(raw,relative_padding=0)[0].box and mask!=core


def test_structured_source_identity_never_uses_physical_only_appeal_recovery(monkeypatch):
    from watermark_studio.source_digital_proof import SOURCE_DIGITAL_PROMPT, _classify
    monkeypatch.setattr("watermark_studio.cross_review._recover_classification",lambda *a:pytest.fail("Wrong task"))
    def structured(chats,fields,*a):
        text=chats[0][0]["content"][-1]["text"]
        assert "不是专门表示缺少实物载体" in text and "正面证据" in text
        assert "physical_print_surface" in fields and "real_object_attachment" in fields
        values={"physical_scene_text":False,"digital_compositing_evidence":True,
            "real_object_attachment":False,"physical_print_surface":False,"ambiguous_identity":False}
        return [{f:values[f] for f in fields}|{"area":"none"}],.1
    monkeypatch.setattr("watermark_studio.quality._structured_vqa",structured)
    chats=[[{"role":"user","content":[{"type":"text","text":SOURCE_DIGITAL_PROMPT}]}]]
    monkeypatch.setattr("watermark_studio.evidence._generate_explanations",lambda *a:[{"status":"error"}])
    with pytest.raises(ValueError,match="observable explanation"):
        _classify(chats,None,None,"cpu")
    monkeypatch.setattr("watermark_studio.evidence._generate_explanations",lambda *a:[
        {"status":"summary_only","summary":"半透明卡片跨越不同方向的背景纹理，保持相同屏幕平面。"}])
    result=_classify(chats,None,None,"cpu")[0]
    assert digital_overlay(result) and result["protocol"]=="source-digital-structured-v4"
