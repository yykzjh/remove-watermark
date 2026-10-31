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
    assert outside_repair_changes(before, after, regions)["tiles"] > 0


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
