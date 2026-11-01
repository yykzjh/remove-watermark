import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from watermark_studio.config import DeploymentConfig, ProcessingConfig
from watermark_studio.detection import parse_detection, source_core_boxes
from watermark_studio.images import composite_protected, pixel_box

RAW=json.dumps({"watermarks":[{"kind":"panel","box":[100,600,450,950]},
                               {"kind":"logo","box":[850,950,990,990]}]})


def test_source_core_fades_only_safety_margin_and_keeps_watermark_opaque():
    source=Image.new("RGB",(1000,800),(50,60,70))
    ImageDraw.Draw(source).rectangle((100,480,449,759),fill="blue")
    generated=Image.new("RGB",source.size,(90,100,110))
    masks=parse_detection(RAW)
    cores=source_core_boxes(RAW)
    assert len(cores)==1 and masks[0].box in cores
    result=composite_protected(source,generated,masks,12,core_boxes=cores)
    pixels=np.asarray(result)
    assert (pixels[480:760,100:450]==(90,100,110)).all()
    l,t,_r,_b=pixel_box(masks[0].box,source.size)
    assert result.getpixel((l,t))==source.getpixel((l,t))
    assert np.array_equal(pixels[:t-13],np.asarray(source)[:t-13])
    # A small logo remains fully opaque throughout its padded box.
    x0,y0,x1,y1=pixel_box(masks[1].box,source.size)
    assert (pixels[y0:y1,x0:x1]==(90,100,110)).all()


@pytest.mark.parametrize("oversized",[False,True])
@pytest.mark.parametrize("profile",["fidelity","located"])
def test_protected_complete_scan_still_sends_one_full_photo_to_diffusion(tmp_path,monkeypatch,oversized,profile):
    from watermark_studio.worker import process_batch
    source=tmp_path/"source.png";target=tmp_path/"out.png"
    image=Image.new("RGB",(800,600),"gray");image.save(source);before=source.read_bytes()
    masks=parse_detection(RAW)
    from watermark_studio.detection import DetectionAreaError
    scans=[]
    def locate(images,*args):
        scans.append(args)
        if oversized and len(scans)==1:
            return [DetectionAreaError("oversized source")]
        if oversized:
            assert "ORIGINAL" in args[-1] and "weaken area limits" in args[-1]
        return [(masks,RAW)]
    monkeypatch.setattr("watermark_studio.worker.locate_watermarks_batch",locate)
    observed=[]
    def refine(images,found,*args,**kwargs):
        assert images[0].size==(800,600) and kwargs["complete"] is True
        return found,{0:1}
    monkeypatch.setattr("watermark_studio.detection.refine_source_details",refine)
    monkeypatch.setattr("watermark_studio.batching.dimensions",lambda image,budget:(800,600))
    def generate(pipe,items,*args):
        observed.extend(item["image"].size for item in items)
        for item in items:
            assert ("自动定位提示" in item["prompt"]) is (profile=="located")
            if profile=="located":
                assert "相机标识及附属细小编码" in item["prompt"] and "真实铭牌" in item["prompt"]
        return [Image.new("RGB",(800,600),"gray") for _ in items]
    monkeypatch.setattr("watermark_studio.batching.independent_batch",generate)
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=SimpleNamespace(
        reset_peak_memory_stats=lambda:None,synchronize=lambda:None,max_memory_allocated=lambda:0,
        max_memory_reserved=lambda:0)))
    pipe=SimpleNamespace(text_encoder=None,processor=None,_execution_device="cpu")
    row=process_batch([{"source":str(source),"destination":str(target),"relative":"source.png"}],
        ProcessingConfig(scope="full_image",preserve_background=True,whole_prompt_profile=profile),
        DeploymentConfig(protection_detail_scan=True,protection_core_feather=True),pipe)[0]
    assert row["ok"] and row["model_inferences"]==1 and observed==[(800,600)]
    assert row["processing_path"]=="whole_composite" and row["source_detail_views"]==1
    assert len(row["protected_core_boxes"])==1
    assert len(scans)==(2 if oversized else 1)
    if oversized:
        assert row["source_geometry_requery"]["same_area_limit"] is True
    assert Image.open(target).size==(800,600) and source.read_bytes()==before


@pytest.mark.parametrize("reuse",[False,True])
@pytest.mark.parametrize("adaptive",[False,True])
def test_quality_weight_swap_finishes_before_verification_and_skips_readonly_edit_phase(tmp_path,monkeypatch,reuse,adaptive):
    from watermark_studio.worker import process_batch
    source=tmp_path/"source.png";target=tmp_path/"out.png"
    Image.new("RGB",(64,64),"gray").save(source)
    if reuse:
        target.write_bytes(source.read_bytes())
    moves=[]
    class Encoder:
        device="cuda:0"
        def to(self,device,**kwargs):
            self.device=str(device);moves.append(self.device)
            return self
    encoder=Encoder()
    pipe=SimpleNamespace(_quality_encoder=encoder,_quality_processor=None,_execution_device="cuda:0",
                         text_encoder=None,processor=None)
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=SimpleNamespace(
        reset_peak_memory_stats=lambda:None,synchronize=lambda:None,empty_cache=lambda:None,
        max_memory_allocated=lambda:0,max_memory_reserved=lambda:0)))
    monkeypatch.setattr("watermark_studio.batching.dimensions",lambda *args:(64,64))
    def generate(pipe,items,*args):
        assert encoder.device=="cpu"
        return [Image.new("RGB",(64,64),"gray") for _ in items]
    monkeypatch.setattr("watermark_studio.batching.independent_batch",generate)
    def verify(pairs,model,*args,**kwargs):
        assert model is encoder and encoder.device=="cuda:0"
        return [{"status":"passed","reason":""} for _ in pairs]
    monkeypatch.setattr("watermark_studio.quality.verify_batch",verify)
    row=process_batch([{"source":str(source),"destination":str(target),"relative":"a.png","reuse_existing":reuse}],
        ProcessingConfig(scope="full_image",quality_check=True),
        DeploymentConfig(quality_verifier=True,quality_backend="independent",
                         quality_phase_offload=not adaptive,quality_phase_offload_above=768 if adaptive else 0),pipe)[0]
    assert row["ok"] and row["quality"]["status"]=="passed" and row["quality_phase_offload"]
    assert moves==(["cuda:0"] if reuse else ["cpu","cuda:0"])

@pytest.mark.parametrize("scene,size,budget,expected",[
    ("document",(2000,1500),1024,1536),("document",(800,600),1024,1024),
    ("document",(2000,1500),768,768),("outdoor",(2000,1500),1024,1024)])
def test_document_detail_budget_routes_only_large_protected_whole_inputs(tmp_path,monkeypatch,scene,size,budget,expected):
    from watermark_studio.worker import process_batch
    source=tmp_path/"s.png";target=tmp_path/"out.png"
    Image.new("RGB",size,"gray").save(source)
    raw=json.dumps({"watermarks":[{"kind":"panel","box":[100,600,450,950]}],"scene":scene})
    monkeypatch.setattr("watermark_studio.worker.locate_watermarks_batch",lambda *a,**k:[(parse_detection(raw),raw)])
    budgets=[]
    def dimensions(image,resolution):
        assert image.size==size
        budgets.append(resolution)
        return (640,480)
    monkeypatch.setattr("watermark_studio.batching.dimensions",dimensions)
    calls=[]
    def generate(pipe,items,*a):
        calls.append(len(items))
        return [Image.new("RGB",(640,480),"gray") for _ in items]
    monkeypatch.setattr("watermark_studio.batching.independent_batch",generate)
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=SimpleNamespace(
        reset_peak_memory_stats=lambda:None,synchronize=lambda:None,max_memory_allocated=lambda:0,
        max_memory_reserved=lambda:0)))
    row=process_batch([{"source":str(source),"destination":str(target),"relative":"s.png"}],
        ProcessingConfig(scope="full_image",preserve_background=True,resolution=budget),
        DeploymentConfig(protected_document_resolution=1536),
        SimpleNamespace(text_encoder=None,processor=None,_execution_device="cpu"))[0]
    assert row["ok"] and budgets==[expected] and row["model_input_budgets"]==[expected]
    assert Image.open(target).size==size and calls==[1]

@pytest.mark.parametrize("broad",[False,True])
def test_ambiguous_source_copy_is_review_and_advice_preserves_physical_text(tmp_path,monkeypatch,broad):
    from watermark_studio.evidence import feedback_from_quality
    from watermark_studio.worker import process_batch
    source=tmp_path/"source.png";target=tmp_path/"out.png"
    Image.new("RGB",(80,60),"gray").save(source)
    raw=json.dumps({"watermarks":[{"kind":"panel","box":[400,400,500,500]}]})
    monkeypatch.setattr("watermark_studio.worker.locate_watermarks_batch",lambda *a:[(parse_detection(raw),raw)])
    evidence={"box":[.4,.4,.5,.5],"status":"uncertain_physical_carrier","requires_review":True}
    if broad:evidence["version"]="source-broad-consensus-v1"
    monkeypatch.setattr("watermark_studio.source_vetting.vet_sources",lambda images,found,*a:(
        [([],raw)],{0:[evidence]}))
    monkeypatch.setattr("watermark_studio.batching.independent_batch",lambda *a:pytest.fail("ambiguous source cannot be repainted"))
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=SimpleNamespace(
        reset_peak_memory_stats=lambda:None,synchronize=lambda:None,max_memory_allocated=lambda:0,
        max_memory_reserved=lambda:0)))
    pipe=SimpleNamespace(text_encoder=None,processor=None,_execution_device="cpu",
                         _quality_encoder=SimpleNamespace(device="cuda:0"),_quality_processor=None)
    row=process_batch([{"source":str(source),"destination":str(target),"relative":"source.png"}],
        ProcessingConfig(scope="full_image",preserve_background=True),
        DeploymentConfig(protection_physical_vet=True,quality_backend="independent"),pipe)[0]
    assert row["ok"] and row["action"]=="source_carrier_review" and target.read_bytes()==source.read_bytes()
    assert row["quality"]["status"]=="review" and row["retry_blocked"]=="uncertain_source_carrier"
    advice=feedback_from_quality(row["quality"])
    assert advice==[{"issue":"uncertain_detail" if broad else "physical_text","box":[.4,.4,.5,.5]}]
    assert row["quality"]["judgment_basis"]["local_guard_codes"]==["source_physical_ambiguity"]
    assert row["quality"]["retry_feedback"]==advice
    assert ("检测范围过大" in row["quality"]["reason"]) is broad


@pytest.mark.parametrize("offload", [False, True])
def test_source_scan_offloads_unused_verifier_and_restores_it_before_source_proofs(
        tmp_path, monkeypatch, offload):
    from watermark_studio.worker import process_batch

    source, target = tmp_path/"source.png", tmp_path/"out.png"
    Image.new("RGB", (80, 60), "gray").save(source)
    raw = json.dumps({"watermarks": [{"kind": "panel", "box": [10, 600, 400, 980]}]})
    encoder = SimpleNamespace(device="cuda:0")
    pipe = SimpleNamespace(text_encoder=None, processor=None, _execution_device="cuda:0",
                           _quality_encoder=encoder, _quality_processor=None)
    events = []
    def park(model):
        assert model is encoder
        events.append("park"); model.device = "cpu"
        return model
    def restore(model, device):
        assert model is encoder
        events.append("restore"); model.device = device
        return model
    def locate(*args):
        events.append("detect")
        assert encoder.device == ("cpu" if offload else "cuda:0")
        return [(parse_detection(raw), raw)]
    def vet(images, found, *args):
        events.append("vet")
        assert encoder.device == "cuda:0"
        return found, {}
    def generate(pipe, items, *args):
        events.append("generate")
        assert encoder.device == ("cpu" if offload else "cuda:0")
        return [Image.new("RGB", (80, 60), "silver") for _ in items]
    def verify(pairs, *args, **kwargs):
        events.append("verify")
        assert encoder.device == "cuda:0"
        return [{"status": "passed", "reason": ""} for _ in pairs]
    monkeypatch.setattr("watermark_studio.quality_loading.park_quality_encoder", park)
    monkeypatch.setattr("watermark_studio.quality_loading.restore_quality_encoder", restore)
    monkeypatch.setattr("watermark_studio.worker.locate_watermarks_batch", locate)
    monkeypatch.setattr("watermark_studio.source_vetting.vet_sources", vet)
    monkeypatch.setattr("watermark_studio.batching.dimensions", lambda image, *a: image.size)
    monkeypatch.setattr("watermark_studio.batching.independent_batch", generate)
    monkeypatch.setattr("watermark_studio.quality.verify_batch", verify)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        reset_peak_memory_stats=lambda: None, empty_cache=lambda: None,
        synchronize=lambda: None, max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0)))
    row = process_batch([{"source": str(source), "destination": str(target), "relative": "source.png"}],
        ProcessingConfig(scope="full_image", preserve_background=True, quality_check=True),
        DeploymentConfig(quality_verifier=True, quality_backend="independent",
            protection_physical_vet=True, quality_phase_offload=offload), pipe)[0]
    assert row["ok"] and row["quality"]["status"] == "passed" and row["model_inferences"] == 1
    assert events == (["park", "detect", "restore", "vet", "park", "generate", "restore", "verify"]
                      if offload else ["detect", "vet", "generate", "verify"])
