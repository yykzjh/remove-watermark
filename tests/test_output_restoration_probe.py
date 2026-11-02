import runpy
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

probe = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/probe-output-restoration.py"))


def test_selection_uses_only_preexisting_heldout_high_resolution_crops():
    def pair(original, variant, split="validation", lower=True):
        return {"id": f"{original}-{variant}", "source_original_sha256": original, "split": split,
                "training_crop_kind": "lower" if lower else "parent", "model_input_resolution": 1024}
    data = {"pairs": [pair(o, v) for o in ("aaa", "bbb") for v in range(4)] +
            [pair("train", 0, split="train"), pair("parent", 0, lower=False)]}
    selected = probe["select_pairs"](data, 2, 4, 123)
    assert len(selected) == 8
    assert {r["source_original_sha256"] for r in selected} == {"aaa", "bbb"}
    assert selected == probe["select_pairs"](data, 2, 4, 123)
    with pytest.raises(ValueError, match="Insufficient"):
        probe["select_pairs"](data, 3, 4, 123)


def test_known_source_geometry_separates_panel_and_app_mark():
    mask = Image.new("L", (64, 64))
    draw = ImageDraw.Draw(mask)
    draw.rectangle((2, 4, 30, 40), fill=255)
    draw.rectangle((40, 50, 60, 60), fill=255)
    regions = probe["known_regions"]({"render": {"panel": [2, 4, 31, 41]}}, mask)
    assert [r["kind"] for r in regions] == ["panel", "logo"]
    assert regions[1]["box"] == [40/64, 50/64, 61/64, 61/64]
    assert probe["known_regions"]({"render": {}}, Image.new("L", (64, 64))) == []
