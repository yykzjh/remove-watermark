import hashlib

import numpy as np
import pytest
from PIL import Image, ImageDraw

from watermark_studio.config import Region
from watermark_studio.source_panel_rims import propose_blue_card_rims
from watermark_studio.translucent_hint import translucent_body_hint


def fixture():
    rng = np.random.default_rng(28)
    gray = rng.integers(55, 175, (600,800), dtype=np.uint8)
    clean = np.repeat(gray[...,None], 3, axis=2)
    pixels = clean.copy()
    pixels[400:460,20:351] = (30,110,220)
    pixels[460:590,20:351] = np.rint(.55*clean[460:590,20:351]+.45*255).astype(np.uint8)
    source = Image.fromarray(pixels)
    region = Region(kind="panel", box=(0,.59,.49,1))
    core = {"mask": list(region.box), "core": [0,.62,.46,.99]}
    row = propose_blue_card_rims(source, [region], [core])[0]
    yes = {"digital_card":True,"rim_clear":True,"uncertain":False,"area":"none"}
    row.update(approved=True, editor=yes, independent=yes, weights_relation="different_weights",
               ocr={"status":"clear"}, source_rim_sha256=hashlib.sha256(
                   source.crop(tuple(row["restore_pixels"])).tobytes()).hexdigest())
    return source, clean, row


def test_known_transmission_hint_recovers_real_structure_and_preserves_every_other_pixel():
    source, clean, row = fixture()
    original = source.tobytes()
    hint, audit = translucent_body_hint(source, [row], .55)
    assert audit[0]["applied"] and not audit[0]["quality_permission"]
    l,t,r,b = audit[0]["body_pixels"]
    before = np.asarray(source)
    after = np.asarray(hint)
    outside = np.ones(before.shape[:2], bool); outside[t:b,l:r] = False
    assert np.array_equal(after[outside], before[outside])
    assert np.abs(after[t+16:b-16,l+16:r-16].astype(float)-
                  clean[t+16:b-16,l+16:r-16].astype(float)).mean() < 1
    assert source.tobytes() == original and hint.size == source.size


def test_missing_or_stale_proof_cannot_initialize_a_body():
    source, _, row = fixture()
    assert translucent_body_hint(source, [row|{"approved":False}])[0] is source
    _, audit = translucent_body_hint(source, [row|{"source_rim_sha256":"stale"}])
    assert not audit[0]["applied"]


def test_opaque_uniform_body_does_not_supply_recoverable_photo_structure():
    source, _, row = fixture()
    ImageDraw.Draw(source).rectangle((20,460,350,589), fill=(225,225,225))
    _, audit = translucent_body_hint(source, [row])
    assert not audit[0]["applied"]


@pytest.mark.parametrize("beta", [0., .9, float("nan")])
def test_invalid_transmission_cannot_silently_change_a_reference(beta):
    source, _, row = fixture()
    with pytest.raises(ValueError):
        translucent_body_hint(source, [row], beta)


def test_hint_worker_uses_one_whole_generation_and_original_semantics_and_qa(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace

    from watermark_studio.config import DeploymentConfig, ProcessingConfig
    from watermark_studio.detection import parse_detection
    from watermark_studio.worker import process_batch

    source, _, proof = fixture()
    path, target = tmp_path/"source.png", tmp_path/"result.png"
    source.save(path)
    original = path.read_bytes()
    core = proof["core"]
    raw = json.dumps({"scene":"outdoor","watermarks":[
        {"kind":"panel","box":[round(v*1000) for v in core]}]})
    regions = parse_detection(raw)
    # Bind the proof to the actual detector mask/core returned for this source.
    from watermark_studio.detection import source_core_boxes
    cores = [{"mask":list(mask),"core":list(box)} for mask,box in source_core_boxes(raw).items()]
    proof = propose_blue_card_rims(source, regions, cores)[0]
    yes = {"digital_card":True,"rim_clear":True,"uncertain":False,"area":"none"}
    proof.update(approved=True, editor=yes, independent=yes, weights_relation="different_weights",
                 ocr={"status":"clear"}, source_rim_sha256=hashlib.sha256(
                     source.crop(tuple(proof["restore_pixels"])).tobytes()).hexdigest())
    monkeypatch.setattr("watermark_studio.worker.locate_watermarks_batch", lambda *a:[(regions,raw)])
    monkeypatch.setattr("watermark_studio.source_panel_rims.inspect_photo_rims", lambda *a:[proof])
    monkeypatch.setattr("watermark_studio.batching.dimensions", lambda image,*a:image.size)
    generated = []
    def generate(pipe, items, *args):
        assert len(items) == 1
        edit = items[0]
        assert edit["image"].size == source.size and edit["image"].tobytes() != source.tobytes()
        assert edit["_semantic_reference"].tobytes() == source.tobytes()
        assert edit["prefilled"] is False  # Must NOT invoke conditioning drift exemptions.
        assert any(r["applied"] for r in edit["translucent_hint"])
        generated.append(True)
        return [source.copy()]
    monkeypatch.setattr("watermark_studio.batching.independent_batch", generate)
    def verify(pairs, *args, **kwargs):
        assert pairs[0][0].tobytes() == source.tobytes()
        assert "conditioning_initialization" not in kwargs["editor_context"][0]
        return [{"status":"review","guard_codes":["repair_texture_collapse"]}]
    monkeypatch.setattr("watermark_studio.quality.verify_batch", verify)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        reset_peak_memory_stats=lambda:None, synchronize=lambda:None,
        max_memory_allocated=lambda:0, max_memory_reserved=lambda:0)))
    pipe = SimpleNamespace(text_encoder=None,processor=None,_execution_device="cpu",
        _quality_encoder=None,_quality_processor=None,_quality_weights_relation="different_weights",
        _source_rim_text_detector=None)
    config = DeploymentConfig(translucent_body_hint=True,protect_source_rims=True,
        quality_verifier=True,quality_backend="independent",protection_core_feather=True)
    row = process_batch([{"source":str(path),"destination":str(target),"relative":"source.png"}],
        ProcessingConfig(scope="full_image",preserve_background=True,quality_check=True), config, pipe)[0]
    assert row["ok"] and row["model_inferences"] == 1 and generated == [True]
    assert row["conditioning_prefill"] == [False] and row["original_semantic_reference"]
    assert row["whole_conditioning_source"] == "assumed_translucent_body_hint"
    assert row["quality"]["status"] == "review" and path.read_bytes() == original


@pytest.mark.parametrize("kwargs", [
    {"translucent_body_hint":True},
    {"translucent_body_hint":True, "protect_source_rims":True, "quality_verifier":True,
     "quality_backend":"independent", "prefill_overlays":True},
    {"translucent_transmission":float("nan")},
])
def test_invalid_hint_configuration_is_rejected(kwargs):
    from pydantic import ValidationError

    from watermark_studio.config import DeploymentConfig
    with pytest.raises(ValidationError):
        DeploymentConfig(**kwargs)
