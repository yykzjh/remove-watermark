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
