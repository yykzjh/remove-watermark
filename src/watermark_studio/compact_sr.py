"""Optional experimental full-frame SR; never a quality approval mechanism."""
from __future__ import annotations

import hashlib
from pathlib import Path

# Real-ESRGAN v0.2.5.0 general-x4v3, also mirrored verbatim on Hugging Face.
# Verify transport contents before torch deserialization; never execute pickle code.
GENERAL_SHA256 = "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292"
WEAK_SHA256 = "1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca"
# Weak-denoise digest independently computed from the official GitHub release.
CHECKPOINTS = {"general": GENERAL_SHA256, "weak": WEAK_SHA256}


def checked_weights(path, variant="general"):
    if variant not in CHECKPOINTS:
        raise ValueError("Unknown pinned compact SR variant")
    path = Path(path).expanduser()
    if not path.is_file() or not 1_000_000 <= path.stat().st_size <= 8_000_000:
        raise ValueError("Expected the pinned compact SR checkpoint, not an arbitrary file")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != CHECKPOINTS[variant]:
        raise ValueError("Compact SR checkpoint SHA256 mismatch")
    return path


def tile_plan(size, edge=256, halo=34):
    """Disjoint output cores, with the entire 34-convolution receptive radius."""
    width, height = size
    if min(width, height) < 1 or edge < 64 or halo < 34:
        raise ValueError("Positive dimensions, edge>=64 and full receptive halo>=34 required")
    for top in range(0, height, edge):
        for left in range(0, width, edge):
            core = (left, top, min(width, left+edge), min(height, top+edge))
            expanded = (max(0, left-halo), max(0, top-halo),
                        min(width, core[2]+halo), min(height, core[3]+halo))
            yield core, expanded


def compact_model():
    """Checkpoint-compatible compact residual CNN; no BasicSR runtime required.

    Architecture specification: official Real-ESRGAN SRVGGNetCompact, BSD-3-Clause.
    See third_party/Real-ESRGAN-LICENSE.txt and the experimental report.
    """
    from torch import nn
    from torch.nn import functional

    class Compact(nn.Module):
        def __init__(self):
            super().__init__()
            layers = [nn.Conv2d(3, 64, 3, 1, 1), nn.PReLU(64)]
            for _ in range(32):
                layers.extend([nn.Conv2d(64, 64, 3, 1, 1), nn.PReLU(64)])
            layers.append(nn.Conv2d(64, 48, 3, 1, 1))
            self.body = nn.ModuleList(layers)
            self.upsampler = nn.PixelShuffle(4)

        def forward(self, value):
            result = value
            for layer in self.body:
                result = layer(result)
            return self.upsampler(result) + functional.interpolate(value, scale_factor=4,
                                                                   mode="nearest")

    return Compact()


class CompactSR:
    def __init__(self, weights, device="cuda:0", *, variant="general"):
        path = checked_weights(weights, variant)
        import torch

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("The measured experimental SR implementation requires a CUDA device")
        self.model = compact_model()
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise TypeError("Malformed compact SR state dictionary")
        state = state.get("params_ema", state.get("params", state))
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False).to(device=self.device, dtype=torch.float16)

    def enhance(self, image, *, reference_resolution=1024, tile_edge=256):
        """One full-photo SR operation, native dimensions, no diffusion inference.

        Full frame remains the reference. Tiles only bound convolution memory and
        include the full receptive context; they do not detect/crop watermarks.
        """
        import math

        import numpy as np
        from PIL import Image

        if not 256 <= reference_resolution <= 2048:
            raise ValueError("Bounded full-frame reference resolution required")
        width, height = image.size
        scale = min(1, reference_resolution/math.sqrt(width*height))
        small_size = (max(1, round(width*scale)), max(1, round(height*scale)))
        small = image.convert("RGB").resize(small_size, Image.Resampling.LANCZOS)
        # The upstream RealESRGANer accepts OpenCV BGR but reverses it to RGB
        # before the model. PIL is already RGB; no channel reversal here.
        source = self.torch.from_numpy(np.asarray(small).copy()).permute(2, 0, 1)
        source = source.unsqueeze(0).to(device=self.device, dtype=self.torch.float16)/255
        output = np.empty((small.height*4, small.width*4, 3), dtype=np.uint8)
        with self.torch.inference_mode():
            for core, expanded in tile_plan(small.size, tile_edge):
                x0, y0, x1, y1 = core
                ex0, ey0, ex1, ey1 = expanded
                prediction = self.model(source[..., ey0:ey1, ex0:ex1])
                prediction = prediction[..., (y0-ey0)*4:(y1-ey0)*4,
                                        (x0-ex0)*4:(x1-ex0)*4]
                pixels = (prediction.squeeze(0).permute(1, 2, 0).float()
                          .clamp(0, 1).mul(255).round().to(self.torch.uint8).cpu().numpy())
                output[y0*4:y1*4, x0*4:x1*4] = pixels
        return Image.fromarray(output).resize(image.size, Image.Resampling.LANCZOS)
