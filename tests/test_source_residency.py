import sys
from types import SimpleNamespace

import pytest

from watermark_studio.config import DeploymentConfig
from watermark_studio.source_residency import transformer_source_phase


@pytest.mark.parametrize("failure", [False, True])
def test_source_transformer_restores_on_success_and_failure_without_touching_editor(monkeypatch, failure):
    moves = []
    class Transformer:
        def to(self, device, **kwargs):
            moves.append(str(device))
            return self
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        synchronize=lambda: None, empty_cache=lambda: None)))
    editor = object()
    pipe = SimpleNamespace(transformer=Transformer(), text_encoder=editor)
    stats = {}
    def inspect():
        with transformer_source_phase(pipe, True, "cuda:0", stats):
            assert moves == ["cpu"] and pipe.text_encoder is editor
            if failure:
                raise ValueError("source scan failed")
    if failure:
        with pytest.raises(ValueError, match="source scan failed"):
            inspect()
    else:
        inspect()
    assert moves == ["cpu", "cuda:0"] and stats["enabled"]
    assert stats["to_host_seconds"] >= 0 and stats["to_device_seconds"] >= 0


def test_disabled_source_transformer_staging_needs_no_torch_or_transformer():
    stats = {}
    with transformer_source_phase(None, False, "cpu", stats):
        assert stats["enabled"] is False
    assert stats["to_host_seconds"] == stats["to_device_seconds"] == 0


@pytest.mark.parametrize("options", [{"quantization": "float8"}, {"offload": "model"}])
def test_staging_cannot_silently_move_unsupported_quantized_or_hooked_weights(options):
    with pytest.raises(ValueError, match="quantization=none"):
        DeploymentConfig(transformer_phase_offload=True, **options)


def test_pre_restore_callback_releases_unused_weights_before_any_cuda_allocation(monkeypatch):
    events = []
    class Transformer:
        def to(self, device, **kwargs):
            events.append(str(device))
            return self
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        synchronize=lambda: None, empty_cache=lambda: None)))
    with transformer_source_phase(SimpleNamespace(transformer=Transformer()), True,
            "cuda:0", {}, before_restore=lambda: events.append("park_verifier")):
        events.append("source_proof")
    assert events == ["cpu", "source_proof", "park_verifier", "cuda:0"]


def test_source_host_mirror_is_built_before_cuda_and_retains_original_precision():
    from watermark_studio.source_residency import prepare_transformer_host_mirror

    host = SimpleNamespace(numel=lambda: 4, element_size=lambda: 2)
    class Parameter:
        device = SimpleNamespace(type="cpu")
        def detach(self):
            return self
        def pin_memory(self):
            return host
    parameter = Parameter()
    calls = []
    model = SimpleNamespace(requires_grad_=lambda value: calls.append(value),
                            named_parameters=lambda: [("weight", parameter)])
    prepare_transformer_host_mirror(model)
    assert calls == [False] and model._source_host_parameters == {"weight": host}
    assert model._source_host_pinned_bytes == 8
    parameter.device = SimpleNamespace(type="cuda")
    with pytest.raises(ValueError, match="before CUDA"):
        prepare_transformer_host_mirror(model)
