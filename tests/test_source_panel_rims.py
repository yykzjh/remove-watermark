import hashlib

import numpy as np
import pytest
from PIL import Image, ImageDraw

from watermark_studio.config import Region
from watermark_studio.source_panel_rims import (
    propose_blue_card_rims,
    proven_photo_rim,
    restore_proven_rims,
)


def fixture():
    photo = Image.new("RGB", (800, 600), (80, 75, 50))
    draw = ImageDraw.Draw(photo)
    draw.rectangle((20, 400, 350, 459), fill=(30, 110, 220))
    draw.rectangle((20, 460, 350, 589), fill=(220, 220, 220))
    region = Region(kind="panel", box=(0, .59, .49, 1))
    core = {"mask": list(region.box), "core": [0, .62, .46, .99]}
    return photo, region, core


def test_blue_header_proposal_does_not_approve_or_change_a_source():
    source, region, core = fixture()
    before = source.tobytes()
    found = propose_blue_card_rims(source, [region], [core])
    assert len(found) == 1 and found[0]["header_top_pixel"] == 400
    assert found[0]["restore_box"][3]*source.height == 392
    assert found[0]["approved"] is False and source.tobytes() == before
    # An identical physical blue/gray board also gives a CV proposal. That is
    # precisely why independent digital-layer and clear-rim proofs are mandatory.
    yes = {"digital_card": True, "rim_clear": True, "uncertain": False, "area": "none"}
    no = yes | {"digital_card": False}
    assert not proven_photo_rim(yes, no, "different_weights", {"status": "clear"})
    assert not proven_photo_rim(yes, yes, "same_weights", {"status": "clear"})
    assert not proven_photo_rim(yes, yes, "different_weights", {"status": "text_present"})


def test_unproven_or_overlapping_rims_never_restore_source_watermarks():
    source, region, core = fixture()
    candidate = Image.new("RGB", source.size, (20, 30, 40))
    proposals = propose_blue_card_rims(source, [region], [core])
    unchanged, count = restore_proven_rims(source, candidate, proposals)
    assert count == 0 and unchanged.tobytes() == candidate.tobytes()
    overlap = Region(kind="logo", box=(.2, .60, .3, .64))
    assert propose_blue_card_rims(source, [region, overlap], [core]) == []
    assert propose_blue_card_rims(source, [region], []) == []


def test_proven_source_pixels_are_exact_without_changing_a_single_other_pixel():
    source, region, core = fixture()
    candidate = Image.new("RGB", source.size, (20, 30, 40))
    record = propose_blue_card_rims(source, [region], [core])[0]
    yes = {"digital_card": True, "rim_clear": True, "uncertain": False, "area": "none"}
    record.update(approved=True, editor=yes, independent=yes, weights_relation="different_weights",
                  ocr={"status": "clear"})
    record["source_rim_sha256"] = hashlib.sha256(source.crop(tuple(record["restore_pixels"])).tobytes()).hexdigest()
    result, count = restore_proven_rims(source, candidate, [record])
    assert count == 1
    pixels = np.asarray(result)
    l, t, r, b = record["restore_pixels"]
    assert np.array_equal(pixels[t:b, l:r], np.asarray(source)[t:b, l:r])
    outside = np.ones((600, 800), dtype=bool)
    outside[t:b, l:r] = False
    assert np.array_equal(pixels[outside], np.asarray(candidate)[outside])
    changed = source.copy()
    changed.putpixel((l, t), (255, 0, 0))
    rejected, rejected_count = restore_proven_rims(changed, candidate, [record])
    assert rejected_count == 0 and rejected.tobytes() == candidate.tobytes()
    assert source.getpixel((25, 410)) == (30, 110, 220)
    assert candidate.getpixel((25, 410)) == result.getpixel((25, 410)) == (20, 30, 40)
    with pytest.raises(ValueError):
        restore_proven_rims(source, candidate.resize((20, 20)), [record])


def approved_record(source, record):
    yes = {"digital_card": True, "rim_clear": True, "uncertain": False, "area": "none"}
    return record | {"approved": True, "editor": yes, "independent": yes,
                     "weights_relation": "different_weights", "ocr": {"status": "clear"},
                     "source_rim_sha256": hashlib.sha256(
                         source.crop(tuple(record["restore_pixels"])).tobytes()).hexdigest()}


def test_tighter_scope_keeps_detecting_bad_repair_inside_a_preserved_photo_rim():
    from watermark_studio.quality import repair_texture_warnings
    from watermark_studio.source_panel_rims import effective_rim_scope

    source, region, core = fixture()
    rng = np.random.default_rng(15)
    clean = Image.fromarray(rng.integers(55, 140, (600, 800, 3), dtype=np.uint8))
    source = clean.copy()
    draw = ImageDraw.Draw(source)
    draw.rectangle((20, 400, 350, 459), fill=(30, 110, 220))
    draw.rectangle((20, 460, 350, 589), fill=(220, 220, 220))
    candidate = clean.copy()
    ImageDraw.Draw(candidate).rectangle((20, 400, 350, 589), fill=(128, 128, 128))
    record = approved_record(source, propose_blue_card_rims(source, [region], [core])[0])
    final, count = restore_proven_rims(source, candidate, [record])
    assert count == 1 and final.tobytes() == candidate.tobytes()
    scoped, scoped_cores, invalid = effective_rim_scope(source, [region], [core], [record])
    assert not invalid and scoped[0].box[1] > region.box[1]
    # Padding can already be original photo while the real repair remains flat.
    # Tightening the INSPECTION scope must catch it, rather than hiding it.
    old = repair_texture_warnings(final, [region], before=source, core_boxes=[core])
    new = repair_texture_warnings(final, scoped, before=source, core_boxes=scoped_cores)
    assert old["count"] == 0 and new["count"] > 0
    changed = source.copy()
    changed.putpixel((10, 370), (255, 0, 0))
    unchanged, _, invalid = effective_rim_scope(changed, [region], [core], [record])
    assert invalid and unchanged == [region]


@pytest.mark.parametrize("digital", [False, True])
def test_worker_keeps_one_full_generation_and_cannot_override_source_identity_conflict(
        tmp_path, monkeypatch, digital):
    import json
    import sys
    from types import SimpleNamespace

    from watermark_studio.config import DeploymentConfig, ProcessingConfig
    from watermark_studio.detection import parse_detection
    from watermark_studio.worker import process_batch

    source, _, _ = fixture()
    ImageDraw.Draw(source).line((20, 380, 300, 380), fill=(255, 0, 0), width=3)
    source_file, output_file = tmp_path/"source.png", tmp_path/"output.png"
    source.save(source_file)
    source_bytes = source_file.read_bytes()
    raw = json.dumps({"scene": "outdoor", "watermarks": [
        {"kind": "panel", "box": [0, 620, 460, 990]}]})
    regions = parse_detection(raw)
    monkeypatch.setattr("watermark_studio.worker.locate_watermarks_batch",
                        lambda images, *args: [(regions, raw)])
    monkeypatch.setattr("watermark_studio.batching.dimensions", lambda image, *args: image.size)
    calls = []
    def generate(pipe, items, *args):
        calls.append(len(items))
        assert items[0]["image"].size == source.size
        assert items[0]["image"].getpixel((100, 420)) == (30, 110, 220)
        return [Image.new("RGB", source.size, (30, 40, 50))]
    monkeypatch.setattr("watermark_studio.batching.independent_batch", generate)
    def proof(photo, proposals, *args):
        assert proposals
        records = [approved_record(photo, r) for r in proposals]
        if not digital:
            for record in records:
                record.update(approved=False, editor=record["editor"] | {"digital_card": False})
        return records
    monkeypatch.setattr("watermark_studio.source_panel_rims.inspect_photo_rims", proof)
    def verify(pairs, *args, **kwargs):
        if digital:
            assert kwargs["editor_context"][0]["source_rim_protection"][0]["approved"]
            assert pairs[0][1].getpixel((100, 380)) == (255, 0, 0)
        return [{"status": "passed", "watermark_removed": True, "natural": True,
                 "content_preserved": True, "uncertain": False, "reason": ""}]
    monkeypatch.setattr("watermark_studio.quality.verify_batch", verify)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        reset_peak_memory_stats=lambda: None, synchronize=lambda: None,
        max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0)))
    pipe = SimpleNamespace(text_encoder=None, processor=None, _execution_device="cpu",
                           _quality_encoder=SimpleNamespace(device="cuda:0"),
                           _quality_processor=None, _quality_weights_relation="different_weights",
                           _source_rim_text_detector=object())
    row = process_batch([{"source": str(source_file), "destination": str(output_file),
                          "relative": "source.png"}],
                        ProcessingConfig(scope="full_image", preserve_background=True,
                                         quality_check=True),
                        DeploymentConfig(quality_verifier=True, quality_backend="independent",
                                         protect_source_rims=True), pipe)[0]
    assert row["ok"] and source_file.read_bytes() == source_bytes
    assert Image.open(output_file).size == source.size
    if digital:
        assert calls == [1] and row["model_inferences"] == 1
        assert row["protected_source_rim_count"] == 1
        assert Image.open(output_file).getpixel((100, 380)) == (255, 0, 0)
    else:
        assert calls == [] and row["model_inferences"] == 0
        assert output_file.read_bytes() == source_bytes
        assert row["quality"]["status"] == "review"
        assert "source_physical_ambiguity" in row["quality"]["guard_codes"]


def test_quality_checks_the_real_inner_edge_after_source_rim_restoration(monkeypatch):
    from watermark_studio.quality import verify_batch

    source, region, core = fixture()
    candidate = Image.new("RGB", source.size, (100, 90, 60))
    proof = approved_record(source, propose_blue_card_rims(source, [region], [core])[0])
    candidate, _ = restore_proven_rims(source, candidate, [proof])
    def structured(chats, fields, *args):
        return [{f: f != "uncertain" for f in fields} | {"area": "none"}
                for _ in chats], .01
    monkeypatch.setattr("watermark_studio.quality._structured_vqa", structured)
    monkeypatch.setattr("watermark_studio.quality.deterministic_warnings", lambda *a: [])
    monkeypatch.setattr("watermark_studio.quality.repair_boundary_warnings", lambda *a, **k: {"count": 0})
    checked = []
    def texture(image, regions, *args, **kwargs):
        checked.append(regions[0].box)
        return {"count": int(regions[0].box[1] > region.box[1]), "suspect_segments": []}
    monkeypatch.setattr("watermark_studio.quality.repair_texture_warnings", texture)
    monkeypatch.setattr("watermark_studio.evidence._generate_explanations", lambda chats, *a: [
        {"status": "ok", "assessment": "no_clear_defect", "summary": "", "findings": []}
        for _ in chats])
    result = verify_batch([(source, candidate)], None, None, "cpu", detail_patches=0,
        source_regions=[[region]], editor_context=[{"protected_core_boxes": [core],
            "source_rim_protection": [proof]}])[0]
    assert checked == [region.box, tuple(proof["guard_region"]["box"])]
    assert result["model_status"] == "passed" and result["status"] == "review"
    assert not result["natural"] and "repair_texture_collapse" in result["guard_codes"]
    assert result["source_rim_guard"]["texture"]["count"] == 1


def test_source_proven_rim_is_used_by_colour_alignment_before_composition():
    from watermark_studio.images import composite_protected, pixel_box
    from watermark_studio.source_panel_rims import effective_rim_scope

    source, region, core = fixture()
    clean = Image.new("RGB", source.size, (80, 75, 50))
    generated = clean.copy()
    ImageDraw.Draw(generated).rectangle((0, 354, 391, 599), fill=(45, 40, 25))
    proof = approved_record(source, propose_blue_card_rims(source, [region], [core])[0])
    old = composite_protected(source, generated, [region], 12, "harmonic",
                              core_boxes={tuple(core["mask"]): tuple(core["core"])})
    old, _ = restore_proven_rims(source, old, [proof])
    scoped, cores, invalid = effective_rim_scope(source, [region], [core], [proof], header_core=True)
    assert not invalid and cores[0]["core"][1] == 400/600
    _, inspection_cores, _ = effective_rim_scope(source, [region], [core], [proof])
    assert inspection_cores[0]["core"][1] == 392/600  # The old inspection is not weakened.
    refined = composite_protected(source, generated, scoped, 12, "harmonic",
                                  core_boxes={tuple(v["mask"]): tuple(v["core"]) for v in cores})
    refined, count = restore_proven_rims(source, refined, [proof])
    assert count == 1
    target = (64, 400, 300, 430)
    def error(photo):
        return np.abs(np.asarray(photo.crop(target), dtype=float)
                      - np.asarray(clean.crop(target), dtype=float)).mean()
    assert error(refined) < error(old)*.75
    l, t, r, b = proof["restore_pixels"]
    assert refined.crop((l,t,r,b)).tobytes() == source.crop((l,t,r,b)).tobytes()
    # The blue header starts at400; blending must already be opaque there.
    assert refined.getpixel((100,400)) != source.getpixel((100,400))
    mask = pixel_box(scoped[0].box, source.size)
    assert mask[1] == 392


def test_raw_capture_never_overwrites_same_sized_photos_or_changes_the_source(tmp_path, monkeypatch):
    import json

    from watermark_studio.images import harmonize_repair

    monkeypatch.setenv("WATERMARK_REPAIR_CAPTURE", str(tmp_path))
    source = Image.new("RGB", (100, 80), (100, 90, 80))
    original = source.tobytes()
    regions = [Region(box=(.1,.5,.5,.9))]
    for colour in [(80,70,60), (70,60,50)]:
        harmonize_repair(source, Image.new("RGB", source.size, colour), regions)
    captures = list(tmp_path.glob("*/capture.json"))
    assert len(captures) == 2 and source.tobytes() == original
    assert len({json.loads(p.read_text())["generated_rgb_sha256"] for p in captures}) == 2
    assert all(json.loads(p.read_text())["original_rgb_sha256"] ==
               hashlib.sha256(original).hexdigest() for p in captures)


def test_whole_source_binding_is_distinct_from_exact_band_restoration():
    from watermark_studio.source_panel_rims import full_source_binding
    source,region,core=fixture()
    record=approved_record(source,propose_blue_card_rims(source,[region],[core])[0])
    assert not full_source_binding(source,record)
    bound=record|{"source_rgb_sha256":hashlib.sha256(source.convert("RGB").tobytes()).hexdigest()}
    assert full_source_binding(source,bound)
    changed=source.copy();changed.putpixel((700,100),(1,2,3))
    assert not full_source_binding(changed,bound)
    # Legacy proof can still copy exactly the already-certified strip. It
    # cannot authorize changing or initializing any other original pixels.
    _result,count=restore_proven_rims(changed,changed,[record])
    assert count==1


def test_explicit_nonapproval_cannot_restore_pixels_despite_old_positive_model_fields():
    source,region,core=fixture()
    record=approved_record(source,propose_blue_card_rims(source,[region],[core])[0])
    candidate=Image.new("RGB",source.size,"black")
    for approval in (False,None,1,"true"):
        result,count=restore_proven_rims(source,candidate,[record|{"approved":approval}])
        assert count==0 and result.tobytes()==candidate.tobytes()
