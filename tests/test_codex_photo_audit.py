import copy

import pytest

from watermark_studio.codex_photo_audit import freeze_plan, validate_verdict


def verdict():
    return {"acceptable": True, "watermark_removed": True, "natural_fill": True,
            "content_preserved": True, "uncertain": False, "defects": [],
            "evidence": ["Both overlays have disappeared; surrounding structures remain continuous."]}


@pytest.mark.parametrize("key", ["watermark_removed", "natural_fill", "content_preserved", "uncertain"])
def test_inconsistent_pass_cannot_be_accepted(key):
    value = verdict()
    value[key] = not value[key]
    with pytest.raises(ValueError, match="contradicts"):
        validate_verdict(value)


def test_no_evidence_is_not_a_valid_pass():
    value = verdict()
    value["evidence"] = []
    with pytest.raises(ValueError, match="evidence"):
        validate_verdict(value)


def test_sample_is_frozen_and_cannot_be_regraded_after_repairs(tmp_path):
    records = [{"file": str(n) + ".jpg", "sha256": str(n), "quality_status": "passed"}
               for n in range(1200)]
    records.append({"file": "review.jpg", "quality_status": "review"})
    plan = freeze_plan(records, tmp_path)
    assert len(set(plan["sample"])) == 1000
    assert plan["initially_rejected"] == ["review.jpg"]
    assert freeze_plan(records, tmp_path) == plan
    changed = copy.deepcopy(records)
    changed[0]["sha256"] = "repaired bytes"
    with pytest.raises(ValueError, match="changed"):
        freeze_plan(changed, tmp_path)


def test_codex_detail_views_use_the_same_exif_orientation_as_complete_images(tmp_path):
    from PIL import Image

    from watermark_studio.codex_photo_audit import detail_pairs
    source, output = tmp_path / "source.jpg", tmp_path / "output.jpg"
    photo = Image.new("RGB", (80, 40), "green")
    exif = Image.Exif()
    exif[274] = 6
    photo.save(source, exif=exif)
    output.write_bytes(source.read_bytes())
    pairs = detail_pairs(source, output, [{"box": [0, 0, 1, 1]}], tmp_path)
    with Image.open(pairs[0]) as pair:
        assert pair.size == (80, 80)


def test_complete_model_input_is_oriented_in_the_same_coordinate_system(tmp_path):
    from PIL import Image

    from watermark_studio.codex_photo_audit import display_input
    source = tmp_path / "original.jpg"
    photo = Image.new("RGB", (80, 40), "green")
    exif = Image.Exif()
    exif[274] = 6
    photo.save(source, exif=exif)
    staged = display_input(source, tmp_path, "source")
    with Image.open(staged) as image:
        assert image.size == (40, 80)
        assert image.getexif().get(274, 1) == 1
    with Image.open(source) as image:
        assert image.size == (80, 40) and image.getexif()[274] == 6


def test_known_legacy_review_is_preserved_but_unknown_identity_cannot_be_reused(tmp_path):
    import hashlib
    import json

    from watermark_studio.codex_photo_audit import LEGACY_PROMPT_SHA256, review_photo
    from watermark_studio.dataset_transfer import digest
    from watermark_studio.production_archives import file_sha256
    original, target = tmp_path / "original", tmp_path / "target"
    original.write_bytes(b"source")
    target.write_bytes(b"candidate")
    folder = tmp_path / "case"
    folder.mkdir()
    saved = {"identity": {"original_md5": digest(original), "output_sha256": file_sha256(target),
                          "model": "gpt-6.1-sol", "prompt_sha256": next(iter(LEGACY_PROMPT_SHA256))},
             "verdict": verdict()}
    path = folder / "review.json"
    path.write_text(json.dumps(saved))
    before = path.read_bytes()
    assert review_photo("must-not-run", original, target, folder) == saved
    assert path.read_bytes() == before
    saved["identity"]["prompt_sha256"] = hashlib.sha256(b"unknown policy").hexdigest()
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="immutable"):
        review_photo("must-not-run", original, target, folder)


@pytest.mark.parametrize("kind,visible", [("content_change", "yes"), ("unnatural_fill", True)])
def test_visible_restoration_requires_an_explicit_valid_content_claim(kind, visible):
    value = verdict()
    value.update(acceptable=False, content_preserved=False)
    value["defects"] = [{"kind": kind, "source_visible": visible, "box": [.1, .1, .2, .2],
                         "evidence": "Visible physical sign changed.", "repair_instruction": "Restore original."}]
    with pytest.raises(ValueError, match="visible-original"):
        validate_verdict(value)


def test_strict_scope_excludes_early_stops_without_changing_passed_sample(tmp_path):
    rows = [{"file": str(n), "quality_status": "passed", "server_attempts": 1} for n in range(1200)]
    rows += [{"file": "early-stop", "quality_status": "review", "server_attempts": 1},
             {"file": "exhausted", "quality_status": "review", "server_attempts": 3}]
    original, strict = tmp_path / "original", tmp_path / "strict"
    original.mkdir(); strict.mkdir()
    normal = freeze_plan(rows, original)
    plan = freeze_plan(rows, strict, minimum_server_attempts=3)
    assert plan["initially_rejected"] == ["exhausted"]
    assert plan["deferred_server_review"] == ["early-stop"]
    assert plan["sample"] == normal["sample"]
    with pytest.raises(ValueError, match="frozen"):
        freeze_plan(rows, strict, minimum_server_attempts=0)
