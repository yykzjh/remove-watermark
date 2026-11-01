import numpy as np
from PIL import Image, ImageDraw

from watermark_studio.config import Region
from watermark_studio.source_complexity import source_resolution_features


def fixture(textured):
    yy, xx = np.indices((1200, 1600))
    values = (60+((xx//8+yy//8) % 2)*90).astype(np.uint8) if textured else np.full((1200, 1600), 100, np.uint8)
    source = Image.fromarray(np.repeat(values[..., None], 3, axis=2))
    region = Region(kind="panel", box=(.10, .50, .60, .90))
    core = {"mask": list(region.box), "core": [.12, .52, .58, .88]}
    ImageDraw.Draw(source).rectangle((192, 624, 928, 1056), fill=(220, 220, 220))
    return source, region, core


def test_score_uses_source_photo_detail_and_never_changes_source_pixels():
    textured, region, core = fixture(True)
    smooth, _, _ = fixture(False)
    original = textured.tobytes()
    a = source_resolution_features(textured, [region], [core], scene="outdoor")
    b = source_resolution_features(smooth, [region], [core], scene="outdoor")
    assert a["eligible"] and a["tiles"] > 0 and a["score"] > b["score"]
    assert textured.tobytes() == original


def test_overlay_interior_text_and_colour_cannot_drive_visible_rim_score():
    source, region, core = fixture(True)
    a = source_resolution_features(source, [region], [core], scene="outdoor")
    ImageDraw.Draw(source).rectangle((350, 770, 500, 850), fill=(0, 0, 0))
    b = source_resolution_features(source, [region], [core], scene="outdoor")
    assert a == b


def test_missing_source_provenance_and_non_outdoor_panels_do_not_route():
    source, region, core = fixture(True)
    assert not source_resolution_features(source, [region], [], scene="outdoor")["eligible"]
    assert not source_resolution_features(source, [region], [core], scene="document")["eligible"]
