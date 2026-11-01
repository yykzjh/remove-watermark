import pytest
from PIL import Image

from watermark_studio.defect_protocol import inspect_defects
from watermark_studio.quality import CONTENT_PROMPT, OUTPUT_PROMPT, scene_review_context


def chat(kind):
    image = Image.new("RGB", (20, 20), "gray")
    prompt = CONTENT_PROMPT if kind == "reference" else OUTPUT_PROMPT.replace(
        "这是一张去水印结果的检验画布：上方为整图，下方（如有）为同一结果图的放大细节。",
        "图1是完整结果照片，后续图是标明原照片坐标的放大局部。结合整图透视和物体关系区分真实印刷文字。")
    return [[{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt + scene_review_context("document")}]}]]


@pytest.mark.parametrize("kind,key", [("output", "watermark_removed"), ("reference", "content_preserved")])
@pytest.mark.parametrize("problem", [True, False])
def test_defect_semantics_preserve_images_scene_and_all_veto_flags(monkeypatch, kind, key, problem):
    chats = chat(kind)
    image = chats[0][0]["content"][0]["image"]
    original_text = chats[0][0]["content"][-1]["text"]
    def inspect(prepared, fields, *args):
        assert prepared[0][0]["content"][0]["image"] is image
        assert scene_review_context("document") in prepared[0][0]["content"][-1]["text"]
        assert key not in prepared[0][0]["content"][-1]["text"]
        return [{fields[0]: problem, fields[1]: problem, fields[2]: problem, "area": "bottom"}], 2.5
    monkeypatch.setattr("watermark_studio.quality._structured_vqa", inspect)
    values, seconds = inspect_defects(chats, kind, None, None, "cpu", 128)
    assert values[0][key] is not problem
    assert values[0]["natural"] is not problem
    assert values[0]["uncertain"] is problem
    assert values[0]["area"] == "bottom" and seconds == 2.5
    assert values[0]["raw_defect_checks"]["repair_defect"] is problem
    assert chats[0][0]["content"][-1]["text"] == original_text


def test_defect_protocol_does_not_fill_missing_fields_with_approval(monkeypatch):
    monkeypatch.setattr("watermark_studio.quality._structured_vqa",
                        lambda *args: ([{"overlay_present": False, "area": "none"}], 0))
    with pytest.raises(ValueError, match="Incomplete"):
        inspect_defects(chat("output"), "output", None, None, "cpu", 128)
    with pytest.raises(ValueError, match="Unknown"):
        inspect_defects(chat("output"), "anything", None, None, "cpu", 128)
