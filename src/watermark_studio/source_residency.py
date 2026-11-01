"""Explicit staging of an unquantized transformer around source-only inspection."""
from contextlib import contextmanager
from time import perf_counter


def prepare_transformer_host_mirror(transformer):
    """Capture immutable CPU weights BEFORE initial CUDA placement."""
    transformer.requires_grad_(False)
    mirror = {}
    pinned = 0
    for name, parameter in transformer.named_parameters():
        if parameter.device.type != "cpu":
            raise ValueError("Build source-stage mirror before CUDA placement")
        try:
            host = parameter.detach().pin_memory()
            pinned += host.numel()*host.element_size()
        except RuntimeError:
            host = parameter.detach().cpu()
        mirror[name] = host
    transformer._source_host_parameters = mirror
    transformer._source_host_pinned_bytes = pinned


@contextmanager
def transformer_source_phase(pipe, enabled, execution_device, stats, *, before_restore=None):
    """The execution device is captured BEFORE moving the pipeline's first module."""
    stats.update(enabled=bool(enabled), to_host_seconds=0., to_device_seconds=0.)
    if not enabled:
        yield
        return
    import torch

    transformer = pipe.transformer
    try:
        began = perf_counter()
        mirror = getattr(transformer, "_source_host_parameters", None)
        if mirror is not None:
            pairs = [(parameter, mirror[name]) for name, parameter in transformer.named_parameters()]
            if any(parameter.shape != host.shape or parameter.dtype != host.dtype for parameter, host in pairs):
                raise ValueError("Source-stage host weights changed shape/dtype")
            for parameter, host in pairs:
                parameter.data = host
        transformer.to("cpu", non_blocking=mirror is not None)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        stats["to_host_seconds"] = round(perf_counter()-began, 5)
        yield
    finally:
        if before_restore is not None:
            before_restore()
        began = perf_counter()
        transformer.to(execution_device, non_blocking=True)
        torch.cuda.synchronize()
        stats["to_device_seconds"] = round(perf_counter()-began, 5)
        stats["host_pinned_bytes"] = getattr(transformer, "_source_host_pinned_bytes", 0)
