import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from watermark_studio.config import DeploymentConfig, ProcessingConfig
from watermark_studio.detection import parse_detection, source_core_boxes
from watermark_studio.images import composite_protected, pixel_box

RAW=json.dumps({"watermarks":[{"kind":"panel","box":[100,600,450,950]},
                               {"kind":"logo","box":[850,950,990,990]}]})


def test_source_core_fades_only_safety_margin_and_keeps_watermark_opaque():
    source=Image.new("RGB",(1000,800),(50,60,70))
    ImageDraw.Draw(source).rectangle((100,480,449,759),fill="blue")
    generated=Image.new("RGB",source.size,(90,100,110))
    masks=parse_detection(RAW)
    cores=source_core_boxes(RAW)
    assert len(cores)==1 and masks[0].box in cores
    result=composite_protected(source,generated,masks,12,core_boxes=cores)
    pixels=np.asarray(result)
    assert (pixels[480:760,100:450]==(90,100,110)).all()
    l,t,_r,_b=pixel_box(masks[0].box,source.size)
    assert result.getpixel((l,t))==source.getpixel((l,t))
    assert np.array_equal(pixels[:t-13],np.asarray(source)[:t-13])
    # A small logo remains fully opaque throughout its padded box.
    x0,y0,x1,y1=pixel_box(masks[1].box,source.size)
    assert (pixels[y0:y1,x0:x1]==(90,100,110)).all()


@pytest.mark.parametrize("oversized",[False,True])
def test_protected_complete_scan_still_sends_one_full_photo_to_diffusion(tmp_path,monkeypatch,oversized):
    from watermark_studio.worker import process_batch
    source=tmp_path/"source.png";target=tmp_path/"out.png"
    image=Image.new("RGB",(800,600),"gray");image.save(source);before=source.read_bytes()
    masks=parse_detection(RAW)
    from watermark_studio.detection import DetectionAreaError
    scans=[]
    def locate(images,*args):
        scans.append(args)
        if oversized and len(scans)==1:
            return [DetectionAreaError("oversized source")]
        if oversized:
            assert "ORIGINAL" in args[-1] and "weaken area limits" in args[-1]
        return [(masks,RAW)]
    monkeypatch.setattr("watermark_studio.worker.locate_watermarks_batch",locate)
    observed=[]
    def refine(images,found,*args,**kwargs):
        assert images[0].size==(800,600) and kwargs["complete"] is True
        return found,{0:1}
    monkeypatch.setattr("watermark_studio.detection.refine_source_details",refine)
    monkeypatch.setattr("watermark_studio.batching.dimensions",lambda image,budget:(800,600))
    def generate(pipe,items,*args):
        observed.extend(item["image"].size for item in items)
        return [Image.new("RGB",(800,600),"gray") for _ in items]
    monkeypatch.setattr("watermark_studio.batching.independent_batch",generate)
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=SimpleNamespace(
        reset_peak_memory_stats=lambda:None,synchronize=lambda:None,max_memory_allocated=lambda:0,
        max_memory_reserved=lambda:0)))
    pipe=SimpleNamespace(text_encoder=None,processor=None,_execution_device="cpu")
    row=process_batch([{"source":str(source),"destination":str(target),"relative":"source.png"}],
        ProcessingConfig(scope="full_image",preserve_background=True),
        DeploymentConfig(protection_detail_scan=True,protection_core_feather=True),pipe)[0]
    assert row["ok"] and row["model_inferences"]==1 and observed==[(800,600)]
    assert row["processing_path"]=="whole_composite" and row["source_detail_views"]==1
    assert len(row["protected_core_boxes"])==1
    assert len(scans)==(2 if oversized else 1)
    if oversized:
        assert row["source_geometry_requery"]["same_area_limit"] is True
    assert Image.open(target).size==(800,600) and source.read_bytes()==before
