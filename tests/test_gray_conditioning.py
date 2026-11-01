import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from watermark_studio.config import DeploymentConfig, ProcessingConfig
from watermark_studio.detection import parse_detection, source_core_boxes
from watermark_studio.gray_conditioning import gray_whole_reference
from watermark_studio.source_panel_rims import propose_blue_card_rims


def fixture():
    image=Image.new("RGB",(800,600),(70,90,55))
    ImageDraw.Draw(image).rectangle((20,400,350,459),fill=(30,110,220))
    ImageDraw.Draw(image).rectangle((20,460,350,589),fill=(220,220,220))
    raw=json.dumps({"scene":"outdoor","watermarks":[{"kind":"panel","box":[25,666,440,983]}]})
    regions=parse_detection(raw)
    cores=source_core_boxes(raw)
    proof=propose_blue_card_rims(image,regions,
        [{"mask":list(k),"core":list(v)} for k,v in cores.items()])[0]
    yes={"digital_card":True,"rim_clear":True,"uncertain":False,"area":"none"}
    proof.update(approved=True,editor=yes,independent=yes,weights_relation="different_weights",
        ocr={"status":"clear"},source_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),source_rim_sha256=hashlib.sha256(
            image.crop(tuple(proof["restore_pixels"])).tobytes()).hexdigest())
    return image,raw,regions,cores,proof


def test_gray_input_is_full_frame_and_changes_only_source_proven_core():
    image,_,regions,cores,proof=fixture()
    original=image.tobytes()
    reference,applied=gray_whole_reference(image,regions,cores,[proof])
    assert applied and reference.size==image.size and image.tobytes()==original
    outside=np.ones((image.height,image.width),bool)
    for l,t,r,b in reference.info["gray_conditioning"]["rectangles"]:
        outside[t:b,l:r]=False
        assert np.all(np.asarray(reference)[t:b,l:r]==128)
        assert t>=proof["header_top_pixel"]
    assert np.array_equal(np.asarray(reference)[outside],np.asarray(image)[outside])
    assert not reference.info["gray_conditioning"]["quality_permission"]


def test_gray_initialization_never_erases_unproven_or_stale_regions():
    image,_,regions,cores,proof=fixture()
    assert gray_whole_reference(image,regions,cores,[proof|{"approved":False}])[0] is image
    with pytest.raises(ValueError):
        gray_whole_reference(image,regions,cores,[proof|{"source_rim_sha256":"stale"}])


def test_gray_worker_still_generates_one_whole_frame_and_checks_original(tmp_path,monkeypatch):
    from watermark_studio.worker import process_batch
    image,raw,regions,_,proof=fixture()
    path,target=tmp_path/"source.png",tmp_path/"result.png";image.save(path)
    original=path.read_bytes()
    monkeypatch.setattr("watermark_studio.worker.locate_watermarks_batch",lambda *a:[(regions,raw)])
    monkeypatch.setattr("watermark_studio.source_panel_rims.inspect_photo_rims",lambda *a:[proof])
    monkeypatch.setattr("watermark_studio.batching.dimensions",lambda image,*a:image.size)
    generated=[]
    def generate(pipe,items,*args):
        assert len(items)==1 and items[0]["image"].size==image.size
        assert items[0]["image"].getpixel((150,430))==(128,128,128)
        assert items[0]["_semantic_reference"] is None
        assert items[0]["prompt"].startswith("Outpaint the image:")
        generated.append(True)
        return [Image.new("RGB",image.size,(70,90,55))]
    monkeypatch.setattr("watermark_studio.batching.independent_batch",generate)
    def verify(pairs,*args,**kwargs):
        assert pairs[0][0].tobytes()==image.tobytes()
        return [{"status":"review","guard_codes":["repair_boundary_step"]}]
    monkeypatch.setattr("watermark_studio.quality.verify_batch",verify)
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=SimpleNamespace(
        reset_peak_memory_stats=lambda:None,synchronize=lambda:None,
        max_memory_allocated=lambda:0,max_memory_reserved=lambda:0)))
    pipe=SimpleNamespace(text_encoder=None,processor=None,_execution_device="cpu",
        _quality_encoder=None,_quality_processor=None,_quality_weights_relation="different_weights",
        _source_rim_text_detector=None)
    config=DeploymentConfig(prefill_overlays=True,prefill_style="gray",protect_source_rims=True,
        quality_verifier=True,quality_backend="independent",protection_core_feather=True)
    row=process_batch([{"source":str(path),"destination":str(target),"relative":"source.png"}],
        ProcessingConfig(scope="full_image",preserve_background=True,quality_check=True),config,pipe)[0]
    assert row["ok"] and generated==[True] and row["model_inferences"]==1
    assert row["whole_conditioning_source"]=="source_proven_gray_initialization"
    assert row["quality"]["status"]=="review" and path.read_bytes()==original



def test_gray_proof_is_bound_to_full_photo_not_just_upper_rim():
    image,_,regions,cores,proof=fixture()
    changed=image.copy()
    changed.putpixel((200,500),(221,221,221))
    # The formerly proven upper strip is identical, but the source card changed.
    assert changed.crop(tuple(proof["restore_pixels"])).tobytes()==image.crop(tuple(proof["restore_pixels"])).tobytes()
    with pytest.raises(ValueError,match="full original"):
        gray_whole_reference(changed,regions,cores,[proof])
