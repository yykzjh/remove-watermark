import json
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from watermark_studio.config import DeploymentConfig, ProcessingConfig, Region
from watermark_studio.deployment_cli import parse_deployment
from watermark_studio.roi_sampling import (
    configure_roi_sampling,
    prepare_roi_sampling,
    roi_sampling_scope,
    source_area_fraction,
)
from watermark_studio.sampling import verify_sampling_steps


class Pipe:
    def __init__(self):
        self.config=SimpleNamespace(sample_sigmas=[1.,.98,.95,.93,.9,.85,.7,.4])
        self.scheduler=SimpleNamespace(timesteps=list(range(8)))
        self._watermark_uniform_steps=0
        self._watermark_sampling_info={"sampling":"checkpoint_sample_sigmas","requested_steps":8}
    def register_to_config(self,**values):
        for name,value in values.items():setattr(self.config,name,value)


def edit(box,state=None):
    state=state or {"source":Image.new("RGB",(100,100)),"whole_generation":False}
    return {"box":box,"state":state,"image":Image.new("RGB",(40,40)),"seed":42,"prompt":"original"}


def test_default_stays_native_and_cli_routes_only_when_explicit(tmp_path):
    path=tmp_path/"deployment.toml";path.write_text("")
    cfg,_=parse_deployment(["--config",str(path)])
    assert cfg.uniform_steps==cfg.roi_steps==0
    cfg,_=parse_deployment(["--config",str(path),"--roi-steps","12","--roi-step-min-area",".14"])
    assert cfg.roi_steps==12 and cfg.roi_step_min_area==.14
    with pytest.raises(ValueError,match="not both"):
        DeploymentConfig(uniform_steps=12,roi_steps=12)
    for value in (float("nan"),float("inf"),-1,1.1):
        with pytest.raises(ValueError):DeploymentConfig(roi_step_min_area=value)


def test_source_geometry_routes_without_reading_quality_or_mutating_content():
    big,small=edit((0,0,20,70)),edit((0,0,20,69))
    original=[(e["image"].tobytes(),e["box"],e["seed"],e["prompt"]) for e in (big,small)]
    prepare_roi_sampling([big,small],DeploymentConfig(roi_steps=12))
    assert [e["_roi_uniform_steps"] for e in (big,small)]==[12,0]
    assert [e["_roi_sampling_route"]["source_area_fraction"] for e in (big,small)]==[.14,.138]
    for e,previous in zip((big,small),original,strict=True):
        assert (e["image"].tobytes(),e["box"],e["seed"],e["prompt"])==previous
        assert "quality" not in e
    whole=edit((0,0,100,100));whole["state"]["whole_generation"]=True
    prepare_roi_sampling([whole],DeploymentConfig(roi_steps=12))
    assert whole["_roi_uniform_steps"]==0


def test_union_does_not_charge_overlap_or_empty_space_and_never_mixes_sources():
    assert source_area_fraction([(0,0,20,50),(10,0,30,50)],(100,100))==.15
    assert source_area_fraction([(0,0,10,10),(90,90,100,100)],(100,100))==.02
    parent=edit((0,0,30,50))
    parent["_roi_members"]=[edit((0,0,20,50),parent["state"]),edit((10,0,30,50),parent["state"])]
    prepare_roi_sampling([parent],DeploymentConfig(roi_steps=12,roi_step_min_area=.18))
    assert parent["_roi_uniform_steps"]==0
    parent["_roi_members"][0]["state"]={}
    with pytest.raises(ValueError,match="different source"):
        prepare_roi_sampling([parent],DeploymentConfig(roi_steps=12))


def test_schedule_scope_restores_native_after_failure_and_records_executed_steps():
    pipe=Pipe();original=list(pipe.config.sample_sigmas);info=pipe._watermark_sampling_info
    configure_roi_sampling(pipe,12,.14)
    with pytest.raises(RuntimeError,match="failed decode"), roi_sampling_scope(pipe,[{"_roi_uniform_steps":12}]):
        assert pipe.config.sample_sigmas is None and pipe._watermark_uniform_steps==12
        pipe.scheduler.timesteps=list(range(12))
        audit=verify_sampling_steps(pipe,{"num_inference_steps":12})
        assert audit["actual_steps"]==12 and audit["sampling"]=="source_area_routed_uniform_schedule"
        raise RuntimeError("failed decode")
    assert pipe.config.sample_sigmas==original and pipe._watermark_uniform_steps==0
    assert pipe._watermark_sampling_info is info
    with roi_sampling_scope(pipe,[{"_roi_uniform_steps":0}]):
        assert pipe.config.sample_sigmas==original
        pipe.scheduler.timesteps=list(range(8))
        assert verify_sampling_steps(pipe,{})["actual_steps"]==8
    with pytest.raises(ValueError,match="one compatible"), roi_sampling_scope(
            pipe,[{"_roi_uniform_steps":0},{"_roi_uniform_steps":12}]):pass


@pytest.mark.parametrize("bucketing",[False,True])
def test_equal_shape_rois_with_different_steps_are_never_in_one_batch(tmp_path,monkeypatch,bucketing):
    from watermark_studio.worker import process_batch

    tasks=[]
    for index in range(3):
        src=tmp_path/f"{index}.png";Image.new("RGB",(800,600),(60,60,60)).save(src)
        tasks.append({"relative":src.name,"source":str(src),"destination":str(tmp_path/f"out{index}.png")})
    regions=[Region(box=(0,.7,.4,1)),Region(box=(0,.6,.4,1)),Region(box=(0,.7,.4,1))]
    monkeypatch.setattr("watermark_studio.worker.locate_watermarks_batch",lambda *a:[
        ([r],json.dumps({"scene":"outdoor","watermarks":[{"kind":"panel","box":[v*1000 for v in r.box]}]}))
        for r in regions])
    monkeypatch.setattr("watermark_studio.batching.dimensions",lambda *a:(64,64))
    observed=[]
    def generate(pipe,items,*args):
        selected={e["_roi_uniform_steps"] for e in items}
        assert len(selected)==1
        observed.append((next(iter(selected)),len(items)))
        return [Image.new("RGB",(64,64),(60,60,60)) for e in items]
    monkeypatch.setattr("watermark_studio.batching.independent_batch",generate)
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=SimpleNamespace(
        reset_peak_memory_stats=lambda:None,synchronize=lambda:None,
        max_memory_allocated=lambda:0,max_memory_reserved=lambda:0)))
    rows=process_batch(tasks,ProcessingConfig(scope="full_image",preserve_background=True,first_pass_mode="focused"),
        DeploymentConfig(roi_steps=12,batch_size=2,batch_bucketing=bucketing),SimpleNamespace(text_encoder=None,processor=None,_execution_device="cpu"))
    assert all(r["ok"] for r in rows)
    assert sorted(observed)==([(0,2),(12,1)] if bucketing else [(0,1),(0,1),(12,1)])
