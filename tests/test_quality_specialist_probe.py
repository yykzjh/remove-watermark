import runpy
from pathlib import Path


def helper():
    return runpy.run_path(str(Path(__file__).parents[1]/"scripts/probe-quality-specialists.py"))


def test_only_detector_role_is_routed_and_all_evidence_arguments_are_preserved():
    inspector, locator, other, processor = object(), object(), object(), object()
    calls = []
    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return "detected"
    counters = {"calls": 0, "images": 0}
    route = helper()["detection_role"](original, inspector, locator, processor, counters)
    images = [object(), object()]
    assert route(images, inspector, object(), "cuda:0", 1024, 2, True) == "detected"
    assert calls[0][0] == (images, locator, processor, "cuda:0", 1024, 2, True)
    route(images, other, processor, "cuda:0", resolution=768)
    assert calls[1][0][1] is other and calls[1][1] == {"resolution": 768}
    assert counters == {"calls": 1, "images": 2}


def test_frozen_source_veto_cannot_be_cleared_by_either_model():
    apply = helper()["enforce_frozen_constraints"]
    value = {"status": "passed", "guard_codes": ["existing"], "reason": ""}
    actual = apply({"frozen_vetoes": ["source_physical_ambiguity"]}, value)
    assert actual["status"] == "review"
    assert actual["guard_codes"] == ["existing", "source_physical_ambiguity"]
    error = {"status": "error", "reason": "failed"}
    assert apply({"frozen_vetoes": ["source_guard"]}, error)["status"] == "error"
