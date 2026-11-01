import sys
from types import SimpleNamespace

import pytest

from watermark_studio.config import DeploymentConfig
from watermark_studio.source_residency import transformer_source_phase


@pytest.mark.parametrize("failure", [False, True])
def test_source_transformer_restores_on_success_and_failure_without_touching_editor(monkeypatch, failure):
    moves = []
    class Transformer:
        def to(self, device):
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
        DeploymentConfig(transformer_source_offload=True, **options)
