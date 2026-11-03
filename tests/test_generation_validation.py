import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from watermark_studio.config import DeploymentConfig, ProcessingConfig, Region
from watermark_studio.generation_validation import (
    NonFiniteGenerationError,
    finite_generation_scope,
    require_finite,
)


@pytest.mark.parametrize("value",[float("nan"),float("inf"),-float("inf")])
def test_nonfinite_values_are_never_silently_converted_to_pixels(value):
    with pytest.raises(NonFiniteGenerationError,match="no image"):
        require_finite(np.asarray([value],dtype=np.float32),"test")


def test_finite_decode_is_identical_and_all_hooks_restore_after_failure():
    class VAE:
        def decode(self,z,**kwargs):
            return (z*2,)
    class Processor:
        @staticmethod
        def numpy_to_pil(images):
            return images+1
    pipe=SimpleNamespace(vae=VAE(),image_processor=Processor())
    before_decode=pipe.vae.decode;before_convert=pipe.image_processor.numpy_to_pil
    value=np.ones((1,3,4,4),dtype=np.float32)
    with finite_generation_scope(pipe):
        assert np.array_equal(pipe.vae.decode(value)[0],value*2)
        assert np.array_equal(pipe.image_processor.numpy_to_pil(value),value+1)
        with pytest.raises(NonFiniteGenerationError):
            pipe.vae.decode(np.asarray([float("nan")]))
        with pytest.raises(NonFiniteGenerationError):
            pipe.image_processor.numpy_to_pil(np.asarray([float("inf")]))
    assert pipe.vae.decode==before_decode
    assert pipe.image_processor.numpy_to_pil==before_convert


def test_decoder_overflow_is_rejected_before_postprocessing_can_clip_it():
    class VAE:
        def decode(self,z,**kwargs):
            return (np.full_like(z,float("inf")),)
    pipe=SimpleNamespace(vae=VAE())
    with finite_generation_scope(pipe),pytest.raises(NonFiniteGenerationError,match="VAE output"):
        pipe.vae.decode(np.zeros((1,3,4,4),dtype=np.float32))


def test_accurate_accumulation_is_opt_in_and_requires_float8():
    assert DeploymentConfig().float8_accurate_accum is False
    assert DeploymentConfig(quantization="float8",float8_accurate_accum=True).float8_accurate_accum is True
    with pytest.raises(ValueError,match="requires float8"):
        DeploymentConfig(float8_accurate_accum=True)


def test_invalid_numeric_batch_is_not_retried_or_written(tmp_path,monkeypatch):
    from watermark_studio.worker import process_batch

    tasks=[]
    for index in range(2):
        src=tmp_path/f"{index}.png";Image.new("RGB",(800,600),"gray").save(src)
        tasks.append({"source":str(src),"destination":str(tmp_path/f"out{index}.png"),"relative":src.name})
    region=Region(box=(0,.6,.4,1))
    raw=json.dumps({"watermarks":[{"kind":"panel","box":[0,600,400,1000]}]})
    monkeypatch.setattr("watermark_studio.worker.locate_watermarks_batch",lambda *a:[([region],raw)]*2)
    monkeypatch.setattr("watermark_studio.batching.dimensions",lambda *a:(64,64))
    calls=[]
    def invalid(pipe,items,*args):
        calls.append(len(items))
        raise NonFiniteGenerationError("Non-finite VAE output; no image may be written")
    monkeypatch.setattr("watermark_studio.batching.independent_batch",invalid)
    cuda=SimpleNamespace(OutOfMemoryError=type("OOM",(RuntimeError,),{}),
        reset_peak_memory_stats=lambda:None,synchronize=lambda:None,empty_cache=lambda:None,
        max_memory_allocated=lambda:0,max_memory_reserved=lambda:0)
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=cuda))
    pipe=SimpleNamespace(text_encoder=None,processor=None,_execution_device="cpu")
    rows=process_batch(tasks,ProcessingConfig(scope="full_image",preserve_background=True,first_pass_mode="focused"),
        DeploymentConfig(batch_size=2,batch_bucketing=True),pipe)
    assert calls==[2] and all(not r["ok"] for r in rows)
    assert all("Non-finite" in r["error"] for r in rows)
    assert not any((tmp_path/f"out{i}.png").exists() for i in range(2))


def test_float8_accumulation_flag_reaches_torchao_configuration(monkeypatch):
    torch=pytest.importorskip("torch")
    ao=pytest.importorskip("torchao.quantization")
    from watermark_studio.optimizations import quantize_transformer

    captured=[]
    monkeypatch.setattr(ao,"quantize_",lambda module,config,**kwargs:captured.append(config))
    model=torch.nn.Sequential(torch.nn.Linear(8,8))
    quantize_transformer(model,"float8","row",accurate_accum=True)
    quantize_transformer(model,"float8","tensor")
    assert captured[0].mm_config.use_fast_accum is False
    assert captured[1].mm_config.use_fast_accum is True

def test_activation_amax_floor_is_opt_in_and_cannot_apply_to_other_quantizers():
    assert DeploymentConfig().float8_amax_floor==0
    assert DeploymentConfig(quantization="float8",float8_amax_floor=1e-6).float8_amax_floor==1e-6
    with pytest.raises(ValueError,match="requires float8"):
        DeploymentConfig(float8_amax_floor=1e-6)


def test_torchao_zero_row_is_preserved_with_positive_amax_floor(monkeypatch):
    torch=pytest.importorskip("torch")
    ao=pytest.importorskip("torchao.quantization")
    from torchao.quantization.quant_primitives import _choose_scale_float8, _quantize_affine_float8

    from watermark_studio.optimizations import quantize_transformer

    inputs=torch.tensor([[0.,0.,0.,0.],[.2,-.5,1.,.7]],dtype=torch.bfloat16)
    scales=_choose_scale_float8(inputs,[1,4],hp_value_lb=1e-6)
    quantized=_quantize_affine_float8(inputs,scales).float()
    assert torch.isfinite(quantized).all() and torch.equal(quantized[0],torch.zeros(4))
    original_scales=_choose_scale_float8(inputs,[1,4])
    assert torch.equal(scales[1],original_scales[1])
    captured=[]
    monkeypatch.setattr(ao,"quantize_",lambda module,config,**kwargs:captured.append(config))
    quantize_transformer(torch.nn.Linear(8,8),"float8","row",amax_floor=1e-6)
    assert captured[0].activation_value_lb==1e-6

def test_row_fp8_fails_early_without_safe_activation_scaling():
    with pytest.raises(ValueError,match="zero-row NaNs"):
        DeploymentConfig(quantization="float8",float8_granularity="row")
    assert DeploymentConfig(quantization="float8",float8_granularity="row",float8_amax_floor=1e-6)
