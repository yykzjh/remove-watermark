import numpy as np
import pytest
from PIL import Image, ImageDraw

from watermark_studio.config import Region
from watermark_studio.images import prefill_overlay_reference
from watermark_studio.prompts import build_prefilled_prompt


def test_source_text_colors_do_not_survive_initialization_and_unmasked_pixels_are_exact():
    source=Image.new("RGB",(400,300),(80,100,60))
    draw=ImageDraw.Draw(source)
    draw.rectangle((0,180,159,299),fill="blue")
    draw.text((20,210),"CAMERA 2026",fill="white")
    original=source.tobytes()
    regions=[Region(kind="panel",box=(0,.6,.4,1))]
    conditioned,used=prefill_overlay_reference(source,(0,180,160,300),(0,100,300,300),regions)
    assert used and conditioned.size==(300,200) and source.tobytes()==original
    assert np.array_equal(np.asarray(conditioned)[:80],np.asarray(source.crop((0,100,300,300)))[:80])
    inside=np.asarray(conditioned)[100:180,20:140]
    assert abs(inside.mean(axis=(0,1))-np.array([80,100,60])).max()<3


def test_other_visible_watermarks_are_excluded_from_color_reference():
    source=Image.new("RGB",(400,300),(80,100,60))
    draw=ImageDraw.Draw(source)
    draw.rectangle((0,180,159,299),fill="blue")
    draw.rectangle((300,240,399,299),fill="magenta")
    regions=[Region(box=(0,.6,.4,1)),Region(box=(.75,.8,1,1))]
    conditioned,used=prefill_overlay_reference(source,(0,180,160,300),(0,100,400,300),regions)
    assert used
    assert np.abs(np.asarray(conditioned)[100:180,20:140].mean(axis=(0,1))-[80,100,60]).max()<3
    assert conditioned.getpixel((350,160))==source.getpixel((350,260))


def test_without_real_context_prefill_is_skipped_and_original_input_is_retained():
    source=Image.new("RGB",(120,100),"blue")
    conditioned,used=prefill_overlay_reference(source,(0,0,120,100),(0,0,120,100),
                                              [Region(box=(0,0,1,1))])
    assert not used and conditioned.tobytes()==source.tobytes()


def test_prefill_prompt_requires_actual_texture_not_a_blank_or_invented_landscape():
    prompt=build_prefilled_prompt((0,60,40,100),(0,20,80,100),scene="document")
    assert "pre-erased" in prompt and "flat or blurred fill" in prompt
    assert "document/receipt" in prompt and "do not assume outdoor terrain" in prompt


@pytest.mark.parametrize("original_semantics",[False,True])
def test_prefill_does_not_rewrite_source_or_add_a_second_diffusion_attempt(tmp_path,monkeypatch,original_semantics):
    import json
    import sys
    from types import SimpleNamespace

    from watermark_studio.config import DeploymentConfig, ProcessingConfig
    from watermark_studio.worker import process_batch
    source,target=tmp_path/"source.png",tmp_path/"result.png"
    photo=Image.new("RGB",(800,600),(80,100,60))
    ImageDraw.Draw(photo).rectangle((0,360,329,599),fill="blue")
    photo.save(source);original=source.read_bytes()
    region=Region(kind="panel",box=(0,.58,.432,1))
    def detect(images,*args):
        assert images[0].getpixel((100,500))==(0,0,255)
        return [([region],json.dumps({"scene":"outdoor","watermarks":[
            {"kind":"panel","box":[0,600,400,1000]}]}))]
    monkeypatch.setattr("watermark_studio.worker.locate_watermarks_batch",detect)
    monkeypatch.setattr("watermark_studio.batching.dimensions",lambda *a:(64,64))
    calls=[]
    def generate(pipe,items,*args):
        calls.append(len(items))
        assert ("pre-erased" in items[0]["prompt"]) is not original_semantics
        semantic=items[0].get("_semantic_reference")
        if original_semantics:
            assert semantic.size==items[0]["image"].size
            assert semantic.getpixel((100,280))==(0,0,255)
        else:
            assert semantic is None
        assert items[0]["image"].getpixel((100,280))!=(0,0,255)
        return [Image.new("RGB",(64,64),(80,100,60))]
    monkeypatch.setattr("watermark_studio.batching.independent_batch",generate)
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=SimpleNamespace(
        reset_peak_memory_stats=lambda:None,synchronize=lambda:None,
        max_memory_allocated=lambda:0,max_memory_reserved=lambda:0)))
    pipe=SimpleNamespace(text_encoder=None,processor=None,_execution_device="cpu")
    row=process_batch([{"source":str(source),"destination":str(target),"relative":"source.png"}],
        ProcessingConfig(scope="full_image",preserve_background=True,first_pass_mode="focused"),
        DeploymentConfig(prefill_overlays=True,prefill_original_semantics=original_semantics),pipe)[0]
    assert row["ok"] and row["conditioning_prefill"]==[True] and row["model_inferences"]==1
    assert row["opaque_core_boxes"]==[None]
    assert calls==[1] and source.read_bytes()==original
    with Image.open(target) as output:
        assert output.size==(800,600) and output.getpixel((700,50))==(80,100,60)
        assert output.getpixel((329,500))==(80,100,60)

def test_whole_prefill_initializes_all_trusted_panels_once_and_retains_other_pixels():
    from watermark_studio.images import prefill_whole_reference
    source=Image.new("RGB",(600,400),(90,100,70))
    draw=ImageDraw.Draw(source)
    draw.rectangle((60,240,239,359),fill="blue")
    draw.rectangle((360,40,539,159),fill="white")
    draw.rectangle((540,360,599,399),fill="magenta")
    regions=[Region(kind="panel",box=(.08,.58,.42,.92)),
             Region(kind="panel",box=(.58,.08,.92,.42)),
             Region(kind="logo",box=(.9,.9,1,1))]
    cores={regions[0].box:(.1,.6,.4,.9),regions[1].box:(.6,.1,.9,.4)}
    before=source.tobytes()
    result,used=prefill_whole_reference(source,regions,cores)
    assert used and result.size==source.size and source.tobytes()==before
    expected=np.ones((400,600),dtype=bool)
    expected[240:360,60:240]=False;expected[40:160,360:540]=False
    assert np.array_equal(np.asarray(source)[expected],np.asarray(result)[expected])
    assert np.max(np.abs(np.asarray(result)[260:340,80:220].mean(axis=(0,1))-[90,100,70]))<3
    assert result.getpixel((570,380))==(255,0,255)


def test_whole_prefill_missing_or_mismatched_provenance_cannot_erase_a_photo():
    from watermark_studio.images import prefill_whole_reference
    source=Image.new("RGB",(300,200),"blue")
    region=Region(kind="panel",box=(.1,.5,.5,.9))
    for mapping in [{},{region.box:(0,0,1,1)}]:
        result,used=prefill_whole_reference(source,[region],mapping)
        assert not used and result.tobytes()==source.tobytes()


@pytest.mark.parametrize("original_semantics",[False,True])
def test_whole_prefill_keeps_a_single_full_frame_generation_and_original_for_qa(tmp_path,monkeypatch,original_semantics):
    import json
    import sys
    from types import SimpleNamespace

    from watermark_studio.config import DeploymentConfig, ProcessingConfig
    from watermark_studio.detection import parse_detection
    from watermark_studio.worker import process_batch
    source,target=tmp_path/"source.png",tmp_path/"result.png"
    photo=Image.new("RGB",(800,600),(80,100,60))
    ImageDraw.Draw(photo).rectangle((80,360,319,539),fill="blue")
    photo.save(source);original=source.read_bytes()
    raw=json.dumps({"scene":"outdoor","watermarks":[{"kind":"panel","box":[100,600,400,900]}]})
    regions=parse_detection(raw)
    monkeypatch.setattr("watermark_studio.worker.locate_watermarks_batch",lambda *args:[(regions,raw)])
    monkeypatch.setattr("watermark_studio.batching.dimensions",lambda image,*a:image.size)
    calls=[]
    def generate(pipe,items,*args):
        calls.append(len(items))
        assert len(items)==1 and items[0]["image"].size==photo.size
        assert items[0]["image"].getpixel((150,420))!=(0,0,255)
        assert items[0]["image"].getpixel((10,10))==photo.getpixel((10,10))
        if original_semantics:
            assert items[0]["_semantic_reference"].getpixel((150,420))==(0,0,255)
            assert "后期数字水印" in items[0]["prompt"]
        else:
            assert items[0]["_semantic_reference"] is None
            assert "完整照片" in items[0]["prompt"]
        assert "crop" not in items[0]["prompt"]
        return [Image.new("RGB",photo.size,(80,100,60))]
    monkeypatch.setattr("watermark_studio.batching.independent_batch",generate)
    def verify(pairs,*args,**kwargs):
        assert pairs[0][0].getpixel((150,420))==(0,0,255)
        return [{"status":"passed"}]
    monkeypatch.setattr("watermark_studio.quality.verify_batch",verify)
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=SimpleNamespace(
        reset_peak_memory_stats=lambda:None,synchronize=lambda:None,
        max_memory_allocated=lambda:0,max_memory_reserved=lambda:0)))
    pipe=SimpleNamespace(text_encoder=None,processor=None,_execution_device="cpu",
                         _quality_encoder=None,_quality_processor=None)
    row=process_batch([{"source":str(source),"destination":str(target),"relative":"source.png"}],
        ProcessingConfig(scope="full_image",preserve_background=True,quality_check=True),
        DeploymentConfig(prefill_overlays=True,protection_core_feather=True,
                         prefill_original_semantics=original_semantics),pipe)[0]
    assert row["ok"] and row["model_inferences"]==1 and calls==[1]
    assert row["original_semantic_reference"] is original_semantics
    assert row["conditioning_prefill"]==[True] and row["whole_conditioning_source"]=="source_harmonic_initialization"
    assert row["quality"]["status"]=="passed" and source.read_bytes()==original
    assert Image.open(target).size==photo.size

def test_conditioning_colour_field_is_distinct_from_final_raw_drift_veto():
    from watermark_studio.images import harmonize_repair, prefill_whole_reference
    values=np.linspace(200,255,400).astype(np.uint8)
    clean=Image.fromarray(np.broadcast_to(values[None,:,None],(300,400,3)).copy())
    original=clean.copy()
    ImageDraw.Draw(original).rectangle((40,180,159,269),fill="blue")
    region=Region(kind="panel",box=(.08,.58,.42,.92))
    conditioned,used=prefill_whole_reference(original,[region],{region.box:(.1,.6,.4,.9)})
    assert used and conditioned.getpixel((80,220))[0]>190
    # An actual generated candidate is never afforded that initialization mode.
    bad=Image.new("RGB",clean.size,(30,30,30))
    result=harmonize_repair(clean,bad,[region])
    assert result.info["harmonization"]["correction_withheld"] is True
    assert result.info["harmonization"]["conditioning_only"] is False


def test_small_footer_without_a_core_does_not_cancel_the_separate_large_panel():
    from watermark_studio.images import prefill_whole_reference

    original=Image.new("RGB",(800,600),(90,80,70))
    panel=Region(kind="panel",box=(.02,.58,.44,.98))
    footer=Region(kind="panel",box=(.84,.93,.99,.99))
    draw=ImageDraw.Draw(original)
    draw.rectangle((20,360,339,579),fill="blue")
    draw.rectangle((680,564,779,589),fill="white")
    photo_bytes=original.tobytes()
    initialized,used=prefill_whole_reference(original,[panel,footer],
                                             {panel.box:(.025,.6,.425,.965)})
    assert used and initialized.size==original.size
    assert initialized.getpixel((100,450))!=(0,0,255)
    assert initialized.getpixel((720,575))==original.getpixel((720,575))
    assert initialized.getpixel((500,450))==original.getpixel((500,450))
    assert original.tobytes()==photo_bytes
    # Missing metadata for a large target still blocks initialization entirely.
    untouched,used=prefill_whole_reference(original,[panel,footer],{})
    assert not used and untouched.tobytes()==photo_bytes


def test_compact_candidate_keeps_scope_physical_text_and_document_semantics():
    from watermark_studio.prompts import build_compact_focused_prompt

    prompt=build_compact_focused_prompt("panel",(10,30,50,80),(0,0,100,100),scene="document")
    assert "x=10.0%至50.0%" in prompt and "y=30.0%至80.0%" in prompt
    assert "今日水印" in prompt and "防伪码" in prompt and "方形阴影" in prompt
    assert "真实标牌" in prompt and "测量刻度全部保持原样" in prompt
    assert "document/receipt" in prompt
