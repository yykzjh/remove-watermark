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
