import sys
from types import SimpleNamespace

import pytest

from watermark_studio.config import DeploymentConfig
from watermark_studio.quality_loading import load_quality_encoder


@pytest.mark.parametrize("mode",["none","int8","nf4"])
def test_verifier_quantization_is_explicit_and_leaves_visual_modules_unquantized(monkeypatch,mode):
    captured={}
    class Encoder:
        def to(self,device): captured["moved"]=device;return self
        def eval(self): return self
        def named_modules(self): return []
    class Factory:
        @staticmethod
        def from_pretrained(path,**kwargs):
            captured.update(kwargs);return Encoder()
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(sys.modules,"bitsandbytes",SimpleNamespace())
    monkeypatch.setitem(sys.modules,"transformers",SimpleNamespace(
        BitsAndBytesConfig=lambda **kw:kw,Qwen3VLForConditionalGeneration=Factory))
    config=DeploymentConfig(quality_verifier=True,quality_backend="independent",quality_quantization=mode)
    load_quality_encoder("/cached/model",config,"cuda:0")
    if mode=="none":
        assert captured["moved"]=="cuda:0" and "quantization_config" not in captured
    else:
        assert "moved" not in captured
        assert captured["device_map"]=={"":"cuda:0"}
        quant=captured["quantization_config"]
        assert quant["llm_int8_skip_modules"]==["visual","lm_head"]
        assert quant.get("load_in_4bit",False)==(mode=="nf4")


def test_quantized_verification_requires_the_separate_verifier():
    with pytest.raises(ValueError):
        DeploymentConfig(quality_quantization="nf4")
