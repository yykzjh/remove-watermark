import json
from types import SimpleNamespace

import pytest
from PIL import Image

from watermark_studio.config import DeploymentConfig, ProcessingConfig
from watermark_studio.detection import parse_detection
from watermark_studio.worker import process_image


def test_detection_schema_padding_and_invalid_results():
    boxes = parse_detection('{"watermarks":[{"kind":"logo","box":[800,0,1000,100]}]}')
    assert boxes[0].box == pytest.approx((0.792, 0, 1, 0.108))
    assert parse_detection('{"watermarks":[]}') == []
    for box in [[900, 100, 100, 200], [0, 0, 1001, 10], [0, 0, 1000, 1000]]:
        with pytest.raises(ValueError):
            parse_detection(json.dumps({"watermarks": [{"kind": "panel", "box": box}]}))
    with pytest.raises((TypeError, ValueError)):
        parse_detection('{"not_boxes":[]}')


def test_no_detection_copies_original_bytes_without_claiming_restoration(tmp_path, monkeypatch):
    source, target = tmp_path/"original.jpg", tmp_path/"result.jpg"
    Image.new("RGB", (50, 40), (110, 90, 60)).save(source)
    monkeypatch.setattr("watermark_studio.worker.locate_watermarks", lambda *args: ([], "{}"))
    pipe = SimpleNamespace(text_encoder=None, processor=None, _execution_device="cpu")
    result = process_image(
        {"source": str(source), "destination": str(target), "relative": "original.jpg"},
        ProcessingConfig(), DeploymentConfig(), pipe,
    )
    assert result["action"] == "no_watermark_detected"
    assert source.read_bytes() == target.read_bytes()

def test_overlapping_detection_groups_are_merged():
    regions = parse_detection(json.dumps({"watermarks": [
        {"kind": "panel", "box": [100, 100, 300, 300]},
        {"kind": "logo", "box": [200, 200, 400, 400]},
        {"kind": "logo", "box": [350, 350, 500, 500]},
    ]}), padding=0)
    assert len(regions) == 1
    assert regions[0].kind == "panel"
    assert regions[0].box == (0.1, 0.1, 0.5, 0.5)
