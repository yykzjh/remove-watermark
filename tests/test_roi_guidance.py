import pytest
from PIL import Image

from watermark_studio.roi_guidance import describe_visible_context


def edit():
    return {"image":Image.new("RGB",(200,100),"gray"),
            "_semantic_reference":Image.new("RGB",(200,100),"blue"),
            "state":{"whole_generation":False,"task":{"relative":"a/photo.png"}},
            "box":(20,30,100,100),"crop":(0,0,200,100),"index":0,
            "resolution":1024,"seed":42,"prompt":"Remove only digital overlay."}


def test_source_observation_is_quoted_data_and_cannot_alter_edit_authority(monkeypatch):
    item=edit();pixels=item["image"].tobytes();original=item["_semantic_reference"].tobytes()
    def observe(chats,*args):
        assert chats[0][0]["content"][0]["image"].getpixel((10,10))==(0,0,255)
        assert "[0.1, 0.3, 0.5, 1.0]" in chats[0][0]["content"][1]["text"]
        return [{"status":"summary_only","summary":"A dark trouser leg crosses the upper right edge.",
                 "generation_tokens":20,"max_tokens":144}]
    monkeypatch.setattr("watermark_studio.evidence._generate_explanations",observe)
    audit=describe_visible_context([item],None,None,"cpu")
    assert audit[0]["applied"] and audit[0]["file"]=="a/photo.png"
    assert "Unverified source-only observation" in item["prompt"]
    assert '"A dark trouser leg' in item["prompt"]
    assert item["image"].tobytes()==pixels and item["_semantic_reference"].tobytes()==original
    assert item["box"]==(20,30,100,100) and item["crop"]==(0,0,200,100) and item["seed"]==42
    assert "quality" not in item and "approved" not in audit[0]


@pytest.mark.parametrize("report",[
    {"status":"error"},
    {"status":"summary_only","summary":"too long","generation_tokens":144},
    {"status":"summary_only","summary":"","generation_tokens":0},
])
def test_incomplete_observation_keeps_the_exact_baseline_prompt(monkeypatch,report):
    item=edit();prompt=item["prompt"]
    monkeypatch.setattr("watermark_studio.evidence._generate_explanations",lambda *a:[report])
    audit=describe_visible_context([item],None,None,"cpu")
    assert not audit[0]["applied"] and item["prompt"]==prompt


def test_small_codes_and_whole_image_path_do_not_pay_for_roi_observations(monkeypatch):
    small=edit();small["resolution"]=512
    whole=edit();whole["state"]["whole_generation"]=True
    monkeypatch.setattr("watermark_studio.evidence._generate_explanations",lambda *a:pytest.fail("Unexpected cost"))
    assert describe_visible_context([small,whole],None,None,"cpu")==[]
