import pytest
from PIL import Image

from watermark_studio.config import Region
from watermark_studio.shadow_assessment import parse_shadow, shadow_chats


def test_shadow_parser_rejects_duplicates_reordered_and_incomplete_decisions():
    value='{"observation":"正常结果中可见浅淡矩形暗边，与周围光照不连贯。","repair_state":"确认有缺陷"}'
    assert parse_shadow(value)["repair_state"]=="确认有缺陷"
    for raw in (
        '{"repair_state":"确认有缺陷","observation":"正常结果中可见浅淡矩形暗边。"}',
        '{"observation":"矩形暗边","repair_state":"确认有缺陷"}',
        '{"observation":"正常结果中可见浅淡矩形暗边。","repair_state":"通过"}',
        value[:-1],value.replace('"repair_state":','"observation":"重复依据","repair_state":'),
    ):
        with pytest.raises(ValueError):
            parse_shadow(raw)


def test_shadow_views_keep_normal_source_result_and_label_diagnostic_only():
    before=Image.new("RGB",(1000,800),(80,110,60))
    after=Image.new("RGB",before.size,(90,120,70))
    chats=shadow_chats(before,after,[Region(kind="panel",box=(.1,.6,.5,.95))])
    images=[p["image"] for p in chats[0][0]["content"] if p["type"]=="image"]
    assert len(images)==5
    assert images[0].getpixel((0,0))==(80,110,60)
    assert images[1].getpixel((0,0))==(90,120,70)
    assert before.getpixel((0,0))==(80,110,60)
    with pytest.raises(ValueError):
        shadow_chats(before,after.resize((500,400)),[])
