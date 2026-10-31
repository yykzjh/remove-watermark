import json
from types import SimpleNamespace

import pytest
from PIL import Image

from watermark_studio.config import DeploymentConfig, ProcessingConfig
from watermark_studio.detection import parse_detection
from watermark_studio.worker import process_image


def test_detection_schema_padding_and_invalid_results():
    boxes = parse_detection('{"watermarks":[{"kind":"logo","box":[800,0,1000,100]}]}')
    assert boxes[0].box == pytest.approx((0.78, 0, 1, 0.114))
    assert parse_detection('{"watermarks":[]}') == []
    for box in [[900, 100, 100, 200], [0, 0, 1001, 10], [0, 0, 1000, 1000]]:
        with pytest.raises(ValueError):
            parse_detection(json.dumps({"watermarks": [{"kind": "panel", "box": box}]}))
    with pytest.raises((TypeError, ValueError)):
        parse_detection('{"not_boxes":[]}')


def test_no_detection_copies_original_bytes_without_claiming_restoration(tmp_path, monkeypatch):
    source, target = tmp_path/"original.jpg", tmp_path/"result.jpg"
    Image.new("RGB", (50, 40), (110, 90, 60)).save(source)
    monkeypatch.setattr("watermark_studio.worker.locate_watermarks_batch", lambda images, *args: [([], "{}") for _ in images])
    import sys
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        reset_peak_memory_stats=lambda: None, synchronize=lambda: None,
        max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0,
    )))
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
    ]}), padding=0, relative_padding=0)
    assert len(regions) == 1
    assert regions[0].kind == "panel"
    assert regions[0].box == (0.1, 0.1, 0.5, 0.5)

def test_margin_covers_underestimated_panel_bottom():
    region = parse_detection(
        '{"watermarks":[{"kind":"panel","box":[556,712,985,950]}]}'
    )[0]
    # The shifted fixture panel ends at y=561 in a 576 px image.
    assert region.box[3] * 576 > 560

def test_retry_hint_must_be_confirmed_on_original_crop(monkeypatch):
    from watermark_studio.config import Region
    from watermark_studio.detection import confirm_source_hints
    source = Image.new("RGB", (1000, 1000), "gray")
    hint = {"kind": "logo", "box": (.8, .8, 1., 1.)}
    monkeypatch.setattr("watermark_studio.detection.locate_watermarks_batch",
                        lambda images, *a: [([], "{}") for _ in images])
    assert confirm_source_hints(source, [], [hint], None, None, None) == []
    monkeypatch.setattr("watermark_studio.detection.locate_watermarks_batch",
                        lambda images, *a: [([Region(kind="logo", box=(.4,.4,.9,.9))], "{}")
                                            for _ in images])
    regions = confirm_source_hints(source, [], [hint], None, None, None)
    assert regions[0].box == pytest.approx((.79, .79, .965, .965))


def test_confirmed_source_region_does_not_need_another_gpu_scan(monkeypatch):
    from watermark_studio.config import Region
    from watermark_studio.detection import confirm_source_hints
    called = []
    monkeypatch.setattr("watermark_studio.detection.locate_watermarks_batch",
                        lambda images, *a: called.extend(images) or [])
    known = [Region(kind="logo", box=(.7, .7, 1, 1))]
    found = confirm_source_hints(Image.new("RGB", (100,100)), known,
                                 [{"kind":"logo", "box":(.8,.8,.9,.9)}], None,None,None)
    assert not called and found == known


@pytest.mark.parametrize("replies,expected_calls,valid",[
    (['{"watermarks":[]}'],1,True),
    (["not JSON",'{"watermarks":[]}'],2,True),
    (["not JSON","still not JSON"],2,False),
    (['{"watermarks":[{"kind":"text","box":[800,800,990,990]}]}',
      '{"watermarks":[{"kind":"logo","box":[800,800,990,990]}]}'],2,True),
    (['{"watermarks":[{"kind":"text","box":[800,800,990,990]}]}',
      '{"watermarks":[{"kind":"text","box":[800,800,990,990]}]}'],2,False),
    (['{"watermarks":[{"kind":"text","box":[800,800,990,990]}]}',
      '{"watermarks":[{"kind":"logo","box":[0,0,1000,1000]}]}'],2,False),
    (['{"watermarks":[{"kind":"panel","box":[0,0,1000,1000]}]}'],1,False),
    (['{"watermarks":[{"kind":"logo","box":[800,800,990,1022]}]}',
      '{"watermarks":[{"kind":"logo","box":[800,800,990,1000]}]}'],2,True),
    (['{"watermarks":[{"kind":"logo","box":[800,800,990,1022]}]}',
      '{"watermarks":[{"kind":"logo","box":[800,800,990,1022]}]}'],2,False),
    (['{"watermarks":[{"kind":"logo","box":[800,800,990,1022]}]}',
      '{"watermarks":[{"kind":"panel","box":[0,0,1000,1000]}]}'],2,False),
])
def test_locator_repairs_only_malformed_format_once_and_never_defaults_to_empty(monkeypatch,replies,expected_calls,valid):
    import sys
    from contextlib import nullcontext

    from watermark_studio.detection import locate_watermarks_batch

    class Inputs(dict):
        input_ids=SimpleNamespace(shape=(1,1))
        def to(self,device): return self
    class Generated:
        def __getitem__(self,key): return self
    prompts=[]
    outputs=iter(replies)
    processor=SimpleNamespace(tokenizer=SimpleNamespace(padding_side="right"))
    def apply(chats,**kwargs):
        prompts.append(chats)
        return Inputs()
    processor.apply_chat_template=apply
    processor.batch_decode=lambda *a,**k:[next(outputs)]
    encoder=SimpleNamespace(generate=lambda **k:Generated())
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(
        inference_mode=nullcontext,cuda=SimpleNamespace(empty_cache=lambda:None)))
    result=locate_watermarks_batch([Image.new("RGB",(200,150))],encoder,processor,"cpu")[0]
    assert len(prompts)==expected_calls
    assert (not isinstance(result,Exception))==valid
    assert processor.tokenizer.padding_side=="right"
    if valid and expected_calls==2:
        assert json.loads(result[1])["format_recovery"]["used"] is True
