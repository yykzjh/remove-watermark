"""Ensure diagnostic metadata cannot disclose target position or expected answers."""
import runpy
from pathlib import Path

from PIL import Image


def test_presence_case_chat_does_not_leak_injected_position_or_label(tmp_path):
    module = runpy.run_path(str(Path(__file__).parents[1]/"scripts/probe-presence-on-cases.py"))
    source, output = tmp_path/"source.png", tmp_path/"result.png"
    Image.new("RGB", (1024, 768), (90, 110, 85)).save(source)
    Image.new("RGB", (1024, 768), (91, 110, 85)).save(output)
    case = {"source": str(source), "output": str(output), "regions": []}
    hidden = case | {"file": "WATERMARK_IN_TOP_LEFT.png", "expected_overlay_present": True,
                     "injected_remaining_overlay": {"box": [.01, .01, .15, .2]},
                     "expected": "review", "quality": {"area": "top_left"}}
    plain = module["case_chat"](case, 768)
    changed = module["case_chat"](hidden, 768)
    left, right = plain[0]["content"], changed[0]["content"]
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert a["type"] == b["type"]
        if a["type"] == "image":
            assert a["image"].size == b["image"].size
            assert a["image"].tobytes() == b["image"].tobytes()
        else:
            assert a == b
