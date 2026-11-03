import json
import sys
from types import SimpleNamespace

import pytest

from watermark_studio.config import DeploymentConfig
from watermark_studio.quality_loading import load_quality_encoder


@pytest.fixture
def model_path(tmp_path):
    (tmp_path/"config.json").write_text(json.dumps({"model_type":"qwen3_vl"}))
    return str(tmp_path)


@pytest.mark.parametrize("mode",["none","int8","nf4"])
def test_verifier_quantization_is_explicit_and_leaves_visual_modules_unquantized(monkeypatch,mode,model_path):
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
    load_quality_encoder(model_path,config,"cuda:0")
    if mode=="none":
        assert captured["moved"]=="cuda:0" and "quantization_config" not in captured
    else:
        assert "moved" not in captured
        assert captured["device_map"]=={"":"cuda:0"}
        quant=captured["quantization_config"]
        assert quant["llm_int8_skip_modules"]==["model.visual","visual","lm_head"]
        assert quant.get("load_in_4bit",False)==(mode=="nf4")


def test_quantized_verification_requires_the_separate_verifier():
    with pytest.raises(ValueError):
        DeploymentConfig(quality_quantization="nf4")


def test_quantization_exclusion_matches_full_visual_module_paths(monkeypatch,model_path):
    # Transformers matches these patterns from the beginning of the full name.
    import re
    captured = {}
    class Encoder:
        def eval(self): return self
        def named_modules(self): return []
    class Factory:
        @staticmethod
        def from_pretrained(path, **kwargs):
            captured.update(kwargs)
            return Encoder()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(sys.modules, "bitsandbytes", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        BitsAndBytesConfig=lambda **kw: kw, Qwen3VLForConditionalGeneration=Factory))
    config = DeploymentConfig(quality_verifier=True, quality_backend="independent",
                              quality_quantization="nf4")
    load_quality_encoder(model_path, config, "cuda:0")
    patterns = captured["quantization_config"]["llm_int8_skip_modules"]
    def excluded(name):
        return any(re.match(f"{key}\\.", name) or re.match(key, name) or name.endswith(key)
                   for key in patterns)
    assert excluded("model.visual.blocks.0.attn.qkv")
    assert excluded("model.visual.merger.mlp.0")
    assert excluded("lm_head")
    assert not excluded("model.language_model.layers.0.self_attn.q_proj")


def test_quantized_visual_module_fails_closed(monkeypatch,model_path):
    class Encoder:
        def eval(self): return self
        def named_modules(self):
            return [("model.visual.blocks.0.attn.qkv", type("Linear4bit", (), {})())]
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(sys.modules, "bitsandbytes", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        BitsAndBytesConfig=lambda **kw: kw, Qwen3VLForConditionalGeneration=SimpleNamespace(
            from_pretrained=lambda *a, **kw: Encoder())))
    config = DeploymentConfig(quality_verifier=True, quality_backend="independent",
                              quality_quantization="nf4")
    with pytest.raises(RuntimeError, match="vision encoder"):
        load_quality_encoder(model_path, config, "cuda:0")



def test_new_checkpoint_uses_its_native_architecture_without_changing_old_loads(tmp_path,monkeypatch):
    called=[]
    class Encoder:
        def to(self,*args): return self
        def named_modules(self): return []
        def eval(self): return self
    def factory(name):
        return SimpleNamespace(from_pretrained=lambda *a,**kw:called.append(name) or Encoder())
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(sys.modules,"transformers",SimpleNamespace(BitsAndBytesConfig=lambda **k:k,
        Qwen3VLForConditionalGeneration=factory("qwen3_vl"),
        Qwen3_5ForConditionalGeneration=factory("qwen3_5"),
        InternVLForConditionalGeneration=factory("internvl")))
    cfg=DeploymentConfig(quality_verifier=True,quality_backend="independent")
    for name in ("qwen3_5","qwen3_vl","internvl"):
        (tmp_path/"config.json").write_text(json.dumps({"model_type":name}))
        load_quality_encoder(tmp_path,cfg,"cpu")
    assert called==["qwen3_5","qwen3_vl","internvl"]
    (tmp_path/"config.json").write_text(json.dumps({"model_type":"unrecognized"}))
    with pytest.raises(ValueError,match="Unsupported"):
        load_quality_encoder(tmp_path,cfg,"cpu")
    assert len(called)==3


def test_reasoning_candidate_closes_thinking_before_structured_json_and_keeps_image_arguments(tmp_path,monkeypatch):
    from watermark_studio.quality_loading import load_quality_processor
    calls=[]
    inner=SimpleNamespace(tokenizer=object(),
        apply_chat_template=lambda *a,**k:calls.append((a,k)) or "rendered")
    monkeypatch.setitem(sys.modules,"transformers",SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *a,**k:inner)))
    (tmp_path/"config.json").write_text(json.dumps({"model_type":"qwen3_5"}))
    processor=load_quality_processor(tmp_path)
    chats=[{"role":"user","content":[{"type":"image","image":"photo"},{"type":"text","text":"inspect"}]}]
    assert processor.tokenizer is inner.tokenizer
    assert processor.apply_chat_template(chats,tokenize=True,return_tensors="pt")=="rendered"
    assert calls[0][0][0] is chats and calls[0][1]["enable_thinking"] is False
    (tmp_path/"config.json").write_text(json.dumps({"model_type":"qwen3_vl"}))
    assert load_quality_processor(tmp_path) is inner


@pytest.mark.parametrize("architecture",["qwen3_5","internvl"])
def test_new_candidate_quantization_is_not_silently_enabled(tmp_path,monkeypatch,architecture):
    (tmp_path/"config.json").write_text(json.dumps({"model_type":architecture}))
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(sys.modules,"transformers",SimpleNamespace(
        BitsAndBytesConfig=lambda **k:k,Qwen3VLForConditionalGeneration=object,
        Qwen3_5ForConditionalGeneration=object,InternVLForConditionalGeneration=object))
    cfg=DeploymentConfig(quality_verifier=True,quality_backend="independent",quality_quantization="nf4")
    with pytest.raises(ValueError,match="unquantized"):
        load_quality_encoder(tmp_path,cfg,"cpu")


def test_parking_reuses_readonly_host_mirror_and_rejects_shape_mismatch(monkeypatch):
    from watermark_studio.quality_loading import park_quality_encoder
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None)))
    host=SimpleNamespace(shape=(2,3),dtype="bf16")
    param=SimpleNamespace(shape=(2,3),dtype="bf16",data="gpu-storage")
    class Encoder:
        def __init__(self):
            self._quality_host_parameters={"weight":host}
        def named_parameters(self):
            return [("weight",param)]
        def to(self,device,**kwargs):
            assert device=="cpu" and param.data is host
            return self
    encoder=Encoder()
    assert park_quality_encoder(encoder) is encoder
    param.shape=(3,2)
    with pytest.raises(ValueError,match="shape/dtype"):
        park_quality_encoder(encoder)
