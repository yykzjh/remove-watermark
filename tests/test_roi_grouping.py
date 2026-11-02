import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from watermark_studio.config import DeploymentConfig, ProcessingConfig, Region
from watermark_studio.roi_grouping import coalesce_edits


def fixtures():
    source=Image.new("RGB",(1200,900),"gray")
    state={"source":source,"whole_generation":False,"source_scene":"outdoor"}
    from watermark_studio.images import context_box
    def edit(box, resolution):
        crop=context_box(box,source.size,.45)
        return {"state":state,"box":box,"crop":crop,"image":source.crop(crop),
                "resolution":resolution,"prefilled":False,"core_box":None,
                "_context":.45,"_extra_prompt":""}
    return [edit((0,630,660,900),1024),edit((960,840,1200,900),512)]


def test_joint_input_keeps_original_masks_and_conserves_every_member():
    edits=fixtures();result=coalesce_edits(edits,"adaptive")
    assert len(result)==1 and result[0]["_roi_members"]==edits
    assert result[0]["resolution"]==1024
    assert "BETWEEN" in result[0]["prompt"] and "verification-code" in result[0]["prompt"]
    assert edits[0]["box"]==(0,630,660,900)  # no union becomes edit authority


def test_unreadable_code_and_separate_mode_do_not_group():
    edits=fixtures()
    assert coalesce_edits(edits) is edits
    edits[1]["box"]=(960,894,1200,900)
    assert len(coalesce_edits(edits,"adaptive"))==2


@pytest.mark.parametrize("scene",["document","indoor","unknown"])
def test_complex_contexts_and_whole_generation_keep_original_route(scene):
    edits=fixtures();edits[0]["state"]["source_scene"]=scene
    assert len(coalesce_edits(edits,"adaptive"))==2


def test_group_never_mixes_different_photographs_or_initialized_inputs():
    edits=fixtures();edits[1]["state"]=dict(edits[1]["state"])
    assert len(coalesce_edits(edits,"adaptive"))==2
    edits=fixtures();edits[1]["prefilled"]=True
    assert len(coalesce_edits(edits,"adaptive"))==2


def test_worker_shared_reference_does_not_paste_between_masks(tmp_path,monkeypatch):
    from watermark_studio.worker import process_batch
    image=Image.new("RGB",(1200,900),"gray")
    ImageDraw.Draw(image).rectangle((700,700,900,880),fill="red")
    source=tmp_path/"original.png";target=tmp_path/"output.png";image.save(source)
    regions=[Region(box=(0,.7,.55,1)),Region(box=(.8,.933333,1,1))]
    raw='{"scene":"outdoor","watermarks":[]}'
    monkeypatch.setattr("watermark_studio.worker.locate_watermarks_batch",lambda *a:[(regions,raw)])
    monkeypatch.setattr("watermark_studio.batching.dimensions",lambda image,resolution:(resolution,resolution))
    calls=[]
    def generate(pipe,items,*args):
        calls.extend(items)
        return [Image.new("RGB",(128,128),"green") for _ in items]
    monkeypatch.setattr("watermark_studio.batching.independent_batch",generate)
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=SimpleNamespace(
        reset_peak_memory_stats=lambda:None,synchronize=lambda:None,
        max_memory_allocated=lambda:0,max_memory_reserved=lambda:0)))
    pipe=SimpleNamespace(text_encoder=None,processor=None,_execution_device="cpu")
    opts=ProcessingConfig(scope="full_image",preserve_background=True,first_pass_mode="focused",
        roi_grouping="adaptive",context=.45,feather=0,repair_blending="feather")
    before=source.read_bytes()
    row=process_batch([{"source":str(source),"destination":str(target),"relative":"original.png"}],
                      opts,DeploymentConfig(),pipe)[0]
    assert row["ok"] and len(calls)==row["model_inferences"]==1
    assert len(row["regions"])==2 and len(row["roi_groups"][0]["members"])==2
    with Image.open(target) as output:
        assert output.size==image.size
        assert np.array_equal(np.asarray(output)[700:881,700:901],np.asarray(image)[700:881,700:901])
        assert output.getpixel((200,800))==(0,128,0)
        assert output.getpixel((1100,880))==(0,128,0)
    assert source.read_bytes()==before


def test_shared_harmonic_alignment_excludes_all_watermark_pixels():
    from watermark_studio.roi_grouping import blend_group
    source=Image.new("RGB",(1200,900),(150,145,130))
    regions=[Region(box=(0,.7,.55,1)),Region(box=(.8,.933333,1,1))]
    for region in regions:
        from watermark_studio.images import pixel_box
        b=pixel_box(region.box,source.size)
        ImageDraw.Draw(source).rectangle((b[0],b[1],b[2]-1,b[3]-1),fill="blue")
    edits=fixtures()
    for edit in edits:edit["state"]["source"]=source
    joined=coalesce_edits(edits,"adaptive")[0]
    output=blend_group(source,Image.new("RGB",(300,200),(90,85,70)),joined,0,"harmonic",
                       reference_original=source,exclude_regions=regions)
    assert output.getpixel((200,800))==(150,145,130)
    assert output.getpixel((1100,880))==(150,145,130)
    assert output.getpixel((800,800))==source.getpixel((800,800))
