import numpy as np
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


def test_prefill_does_not_rewrite_source_or_add_a_second_diffusion_attempt(tmp_path,monkeypatch):
    import json
    import sys
    from types import SimpleNamespace

    from watermark_studio.config import DeploymentConfig, ProcessingConfig
    from watermark_studio.worker import process_batch
    source,target=tmp_path/"source.png",tmp_path/"result.png"
    photo=Image.new("RGB",(800,600),(80,100,60))
    ImageDraw.Draw(photo).rectangle((0,360,319,599),fill="blue")
    photo.save(source);original=source.read_bytes()
    region=Region(kind="panel",box=(0,.6,.4,1))
    def detect(images,*args):
        assert images[0].getpixel((100,500))==(0,0,255)
        return [([region],json.dumps({"scene":"outdoor","watermarks":[
            {"kind":"panel","box":[0,600,400,1000]}]}))]
    monkeypatch.setattr("watermark_studio.worker.locate_watermarks_batch",detect)
    monkeypatch.setattr("watermark_studio.batching.dimensions",lambda *a:(64,64))
    calls=[]
    def generate(pipe,items,*args):
        calls.append(len(items))
        assert "pre-erased" in items[0]["prompt"]
        assert items[0]["image"].getpixel((100,280))!=(0,0,255)
        return [Image.new("RGB",(64,64),(80,100,60))]
    monkeypatch.setattr("watermark_studio.batching.independent_batch",generate)
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=SimpleNamespace(
        reset_peak_memory_stats=lambda:None,synchronize=lambda:None,
        max_memory_allocated=lambda:0,max_memory_reserved=lambda:0)))
    pipe=SimpleNamespace(text_encoder=None,processor=None,_execution_device="cpu")
    row=process_batch([{"source":str(source),"destination":str(target),"relative":"source.png"}],
        ProcessingConfig(scope="full_image",preserve_background=True,first_pass_mode="focused"),
        DeploymentConfig(prefill_overlays=True),pipe)[0]
    assert row["ok"] and row["conditioning_prefill"]==[True] and row["model_inferences"]==1
    assert calls==[1] and source.read_bytes()==original
    with Image.open(target) as output:
        assert output.size==(800,600) and output.getpixel((700,50))==(80,100,60)
