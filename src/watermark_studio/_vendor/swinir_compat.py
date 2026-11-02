"""Torch-only inference equivalents of SwinIR's three timm utilities."""
from torch import nn
from torch.nn.init import trunc_normal_  # noqa: F401


def to_2tuple(value):
    return value if isinstance(value, tuple) else (value, value)


class DropPath(nn.Module):
    """The official evaluation path is identity; training is intentionally unavailable."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, value):
        if self.training and self.drop_prob:
            raise RuntimeError("Vendored SwinIR supports evaluation only")
        return value



def inference_mask(block, size, values):
    """Geometry-only shifted-window mask, shared once per current canvas/dtype."""
    cache = getattr(block, "_inference_mask_cache", None)
    geometry = (tuple(size), values.device.type, values.device.index, values.dtype,
                block.window_size)
    if cache is None:
        return block.calculate_mask(size).to(device=values.device, dtype=values.dtype)
    if cache.get("geometry") != geometry:
        cache.clear()
        cache.update(geometry=geometry, masks={})
    masks = cache["masks"]
    if block.shift_size not in masks:
        if len(masks) >= 2:
            raise ValueError("Only the two official SwinIR shift variants may be cached")
        masks[block.shift_size] = block.calculate_mask(size).to(device=values.device, dtype=values.dtype)
    return masks[block.shift_size]
