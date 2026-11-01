"""Explicit staging of an unquantized transformer around source-only inspection."""
from contextlib import contextmanager
from time import perf_counter


@contextmanager
def transformer_source_phase(pipe, enabled, execution_device, stats):
    """The execution device is captured BEFORE moving the pipeline's first module."""
    stats.update(enabled=bool(enabled), to_host_seconds=0., to_device_seconds=0.)
    if not enabled:
        yield
        return
    import torch

    transformer = pipe.transformer
    try:
        began = perf_counter()
        transformer.to("cpu")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        stats["to_host_seconds"] = round(perf_counter()-began, 5)
        yield
    finally:
        began = perf_counter()
        transformer.to(execution_device)
        torch.cuda.synchronize()
        stats["to_device_seconds"] = round(perf_counter()-began, 5)
