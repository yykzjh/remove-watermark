"""Validation must replay native ROI pixels instead of rounding the model grid twice."""
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from watermark_studio import batching, training_pairs


def fixture(tmp_path, monkeypatch):
    yy, xx = np.mgrid[:32, :48]
    rgb = np.stack([xx*4, yy*6, xx+yy], axis=2).astype(np.uint8)
    original = Image.fromarray(rgb)
    source = tmp_path/"original.png"
    original.save(source)
    edited = original.copy()
    edited.paste("white", (8, 12, 28, 26))
    crop = (2, 3, 46, 31)
    frozen = edited.crop(crop).resize((128, 64), Image.Resampling.LANCZOS)
    pair = tmp_path/"frozen.png"
    frozen.save(pair)
    font = tmp_path/"font"
    font.write_bytes(b"fixed validation font")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    render = {"panel": (8, 12, 28, 26), "profile": "camera-v2", "kind": "panel"}
    row = {"id": "source-roi-0", "input": str(pair), "source_original": str(source),
           "source_original_sha256": sha(source), "crop": (0, 0, 48, 32),
           "roi_context_crop_native_parent": crop, "seed": 123,
           "render": json.loads(json.dumps(render | {"kind": "roi_panel"})),
           "model_input_resolution": 512}
    data = {"prompt_profile": "roi_focused_v1", "font_sha256": sha(font)}
    monkeypatch.setattr(batching, "dimensions", lambda image, budget: (128, 64))
    monkeypatch.setattr(training_pairs, "render_profile_pair",
                        lambda *args, **kwargs: (edited.copy(), None, render.copy()))
    helper = runpy.run_path(str(Path(__file__).parents[1]/"scripts/probe-lora-pilot.py"))
    return row, data, font, helper["validation_reference"], edited.crop(crop)


def test_json_frozen_metadata_replays_exact_native_reference(tmp_path, monkeypatch):
    row, data, font, replay, native = fixture(tmp_path, monkeypatch)
    actual, frozen = replay(row, data, font)
    assert actual.size == (44, 28) and frozen.size == (128, 64)
    assert actual.tobytes() == native.tobytes()


@pytest.mark.parametrize("alter", ["source", "font", "frozen"])
def test_validation_rejects_mutated_original_font_or_frozen_pixels(tmp_path, monkeypatch, alter):
    row, data, font, replay, _ = fixture(tmp_path, monkeypatch)
    if alter == "font":
        font.write_bytes(b"changed font")
    else:
        path = Path(row["source_original"] if alter == "source" else row["input"])
        photo = Image.open(path).convert("RGB")
        photo.putpixel((0, 0), (255, 0, 255))
        photo.save(path)
    with pytest.raises(ValueError):
        replay(row, data, font)
