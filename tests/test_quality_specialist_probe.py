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


def test_higher_native_scan_resolution_preserves_the_inspector_and_other_detectors():
    calls=[]
    inspector,locator,processor,other=object(),object(),object(),object()
    def original(*args,**kwargs):
        calls.append((args,kwargs))
    route=helper()["detection_role"](original,inspector,locator,processor,
                                    {"calls":0,"images":0},2048)
    images=[object()]
    route(images,inspector,None,"cuda:0",768,2)
    assert calls[-1][0][4:]==(2048,2)
    route(images,inspector,None,"cuda:0",resolution=1024)
    assert calls[-1][1]["resolution"]==2048
    route(images,other,processor,"cuda:0",resolution=768)
    assert calls[-1][0][1] is other and calls[-1][1]["resolution"]==768


def test_output_source_reference_keeps_output_pixels_and_original_in_separate_labeled_roles():
    from PIL import Image

    from watermark_studio.config import Region
    source, output = Image.new("RGB", (640, 480), "red"), Image.new("RGB", (640, 480), "blue")
    chat = [{"role": "user", "content": [{"type": "text", "text": "result introduction"},
                                       {"type": "image", "image": output},
                                       {"type": "text", "text": "original inspection"}]}]
    actual = helper()["source_aware_output_chat"](chat, source, [Region(box=(.1, .6, .4, .9))])
    content = actual[-1]["content"]
    assert content[3]["image"] is output
    assert len(chat[-1]["content"]) == 3
    assert chat[-1]["content"][-1]["text"] == "original inspection"
    assert "只验收RESULT" in content[0]["text"]
    assert content[1]["image"].getpixel((0, 0)) == (255, 0, 0)
    assert "直接复制仍是不合格" in content[-1]["text"]
    assert "候选范围" in content[-1]["text"]


def test_carrier_output_inspection_never_sees_the_watermarked_original_or_a_physical_exemption():
    from PIL import Image
    before, after = Image.new("RGB", (800, 600), "red"), Image.new("RGB", (800, 600), "blue")
    chat = [{"role": "user", "content": [{"type": "image", "image": after},
                                       {"type": "text", "text": "inspect all output pixels"}]}]
    routed = helper()["carrier_context_chat"](chat, before, after, [[.6, .7, .7, .8]])
    assert len(chat[-1]["content"]) == 2
    pictures = [r["image"] for r in routed[-1]["content"] if r["type"] == "image"]
    assert len(pictures) == 2 and all(p.getpixel((0, 0)) == (0, 0, 255) for p in pictures)
    assert "不证明它是实物或水印" in routed[-1]["content"][1]["text"]
    assert routed[-1]["content"][-1]["text"] == "inspect all output pixels"


def test_carrier_reference_compares_exact_same_native_scope_in_both_photos():
    from PIL import Image
    before, after = Image.new("RGB", (800, 600), "red"), Image.new("RGB", (800, 600), "blue")
    chat = [{"role": "user", "content": [{"type": "text", "text": "compare visible content"}]}]
    routed = helper()["carrier_context_chat"](chat, before, after, [[.6, .7, .7, .8]], True)
    pictures = [r["image"] for r in routed[-1]["content"] if r["type"] == "image"]
    assert len(pictures) == 2 and pictures[0].size == pictures[1].size
    assert pictures[0].getpixel((0, 0)) == (255, 0, 0)
    assert pictures[1].getpixel((0, 0)) == (0, 0, 255)
    assert "乱码" in routed[-1]["content"][-1]["text"]
    assert chat[-1]["content"][-1]["text"] == "compare visible content"


def test_carrier_context_keeps_object_context_and_bounds_images_and_count():
    from PIL import Image
    view, scopes = helper()["carrier_context_views"](
        Image.new("RGB", (1000, 800)), [[.98, .98, 1., 1.], [0, 0, .03, .03], [.4, .4, .45, .45]])
    assert len(view) == 2
    for x, y, r, b in scopes:
        assert 0 <= x < r <= 1000 and 0 <= y < b <= 800
        assert r-x >= 288 and b-y >= 288
