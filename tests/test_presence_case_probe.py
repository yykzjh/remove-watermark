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


def test_shadow_failure_is_not_assumed_to_be_overlay():
    module = runpy.run_path(str(Path(__file__).parents[1]/"scripts/probe-presence-on-cases.py"))
    assert "expected_overlay_present" not in module["score_case"]({"expected": "review", "kind": "shadow"})
    positive = module["score_case"]({"injected_remaining_overlay": {"box": [.1, .1, .2, .2]}})
    assert positive["expected_overlay_present"] is True
    clean = {"expected_overlay_present": False, "injected_remaining_overlay": {}}
    assert module["score_case"](clean)["expected_overlay_present"] is False
