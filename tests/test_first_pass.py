
import pytest
from PIL import Image

from watermark_studio.config import DeploymentConfig, ProcessingConfig, Region
from watermark_studio.detection import refine_source_details
from watermark_studio.runtime import Runtime


def test_first_pass_mode_requires_explicit_preservation():
    assert ProcessingConfig().first_pass_mode=="whole"
    with pytest.raises(ValueError):
        ProcessingConfig(first_pass_mode="thorough")


def test_detail_scan_adds_mapped_small_overlay_without_expanding_large_panel(monkeypatch):
    source=Image.new("RGB",(4000,3000))
    panel=Region(kind="panel",box=(0,.6,.45,1))
    def locate(crops,*args):
        assert len(crops)==4
        return [([],"") for _ in range(3)]+[([Region(kind="logo",box=(.6,.93,.99,1))],"")]
    monkeypatch.setattr("watermark_studio.detection.locate_watermarks_batch",locate)
    results,counts=refine_source_details([source],[([panel],"global")],None,None,"cpu")
    assert counts=={0:4} and results[0][0][0]==panel
    assert len(results[0][0])==2
    assert results[0][0][1].box[0]==pytest.approx(.776)
    assert results[0][0][1].box[1]==pytest.approx(.9608)


def test_detail_scan_skips_native_small_images_and_existing_small_overlay(monkeypatch):
    def locate(crops,*args):
        assert crops==[]
        return []
    monkeypatch.setattr("watermark_studio.detection.locate_watermarks_batch",locate)
    images=[Image.new("RGB",(1000,800)),Image.new("RGB",(4000,3000))]
    found=[([],""),([Region(kind="logo",box=(.8,.95,1,1))],"")]
    assert refine_source_details(images,found,None,None,"cpu")[1]=={0:0,1:0}


def test_first_pass_counts_do_not_relabel_retries_or_existing_outputs(tmp_path):
    runtime=Runtime(DeploymentConfig(backend="mock",state_dir=str(tmp_path)))
    runtime.job={"id":"first-pass","state":"running","total":2,"completed":0,"succeeded":0,
                 "failed":0,"errors":[],"no_watermark":0,"invalid":0,"options":{"quality_check":True}}
    runtime._persist()
    runtime._record({"file":"a.jpg","ok":True,"action":"restored","quality":{"status":"review"},
        "quality_attempts":1,"_retry_task":{"relative":"a.jpg"}})
    runtime._record({"file":"a.jpg","ok":True,"action":"restored","quality":{"status":"passed"},"quality_attempts":2})
    runtime._record({"file":"old.jpg","ok":True,"action":"existing_output_verified","quality":{"status":"passed"},"quality_attempts":1})
    assert runtime.job["first_pass_total"]==1
    assert runtime.job["first_pass_counts"]=={"review":1}
    assert runtime.job["first_pass_existing"]==1


def test_small_overlay_prompt_is_focused_and_preserves_physical_scene():
    from watermark_studio.prompts import build_small_overlay_prompt

    text=build_small_overlay_prompt((80,90,100,100),(60,70,100,100))
    assert "verification code" in text and "real physical printing" in text
    assert "50.0%..100.0%" in text
