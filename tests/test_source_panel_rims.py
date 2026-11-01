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
    record.update(editor=yes, independent=yes, weights_relation="different_weights",
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
