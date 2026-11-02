"""Offline output-only SwinIR x4 restoration; never grants quality permission.

Only generated RGB enters this operator. Original protected pixels are composed by
the caller afterward. No source image, OCR, mask, or diffusion call is accepted.
Network provenance and license: third_party/SwinIR/.
"""
from pathlib import Path


def make_swinir(profile):
    from ._vendor.swinir import SwinIR

    options = {
        "classical": {"depths": [6] * 6, "embed_dim": 180, "num_heads": [6] * 6,
                          "upsampler": "pixelshuffle"},
        "lightweight": {"depths": [6] * 4, "embed_dim": 60, "num_heads": [6] * 4,
                            "upsampler": "pixelshuffledirect"},
    }
    if profile not in options:
        raise ValueError("Only pinned classical/lightweight x4 research profiles")
    return SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.0,
                  mlp_ratio=2, resi_connection="1conv", **options[profile]).eval().requires_grad_(False)


def load_swinir(path, expected_sha256, device, profile, precision="float32"):
    """Restricted PyTorch state loading; never falls back to executable pickle."""
    import hashlib
    import re

    import torch

    if precision not in ("float32", "float16") or (precision == "float16" and not str(device).startswith("cuda")):
        raise ValueError("Explicit float32 or CUDA float16 SwinIR precision required")
    path = Path(path).resolve(strict=True)
    if not re.fullmatch(r"[a-f0-9]{64}", expected_sha256) or not 0 < path.stat().st_size < 96 * 2**20:
        raise ValueError("Bounded SHA-pinned official checkpoint required")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected_sha256:
            raise ValueError("SwinIR artifact SHA mismatch")
    values = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(values, dict):
        raise TypeError("SwinIR tensor mapping required")
    if set(values) == {"params"}:
        values = values["params"]
    if not isinstance(values, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in values.items()):
        raise ValueError("SwinIR requires tensors only")
    net = make_swinir(profile)
    expected = net.state_dict()
    if set(values) != set(expected) or any(values[k].shape != v.shape or values[k].dtype != v.dtype for k, v in expected.items()):
        raise ValueError("Every SwinIR key, shape and dtype must match")
    if any(v.is_floating_point() and not torch.isfinite(v).all().item() for v in values.values()):
        raise ValueError("Nonfinite SwinIR state")
    net.load_state_dict(values, strict=True)
    net.to(device=device, dtype=getattr(torch, precision))
    if any(v.is_floating_point() and not torch.isfinite(v).all().item() for v in net.state_dict().values()):
        raise ValueError("SwinIR state overflow after precision conversion")
    return net


def upscale_swinir(image, net, tile=256, overlap=32):
    """Official RGB/padding and overlap-average convention, bounded CPU assembly."""
    import numpy as np
    import torch
    from PIL import Image

    if image.width < 8 or image.height < 8 or image.width * image.height > 3 * 2**20:
        raise ValueError("Bounded generated RGB canvas required")
    if tile != 0 and (not 32 <= tile <= 512 or tile % 8 or not 0 <= overlap < tile or overlap % 8):
        raise ValueError("Window-aligned tile and overlap required")
    if tile == 0 and image.width * image.height > 1310720:
        raise ValueError("Untiled SwinIR is limited to the measured one-megapixel budget")
    param = next(net.parameters())
    source = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255).permute(2, 0, 1)[None]
    h, w = source.shape[-2:]
    hp, wp = (h // 8 + 1) * 8, (w // 8 + 1) * 8
    source = torch.cat([source, source.flip([2])], 2)[:, :, :hp, :]
    source = torch.cat([source, source.flip([3])], 3)[:, :, :, :wp]
    extent = min(tile or max(hp, wp), hp, wp)
    stride = extent - min(overlap, extent - 8)
    rows = list(range(0, hp - extent, stride)) + [hp - extent]
    cols = list(range(0, wp - extent, stride)) + [wp - extent]
    # For the small untiled reference, preserve the complete rectangular canvas.
    patches = [(0, 0, hp, wp)] if tile == 0 else [(y, x, extent, extent) for y in rows for x in cols]
    out = torch.zeros((1, 3, hp * 4, wp * 4), dtype=torch.float32)
    weight = torch.zeros((1, 1, hp * 4, wp * 4), dtype=torch.float32)
    with torch.inference_mode():
        for y, x, ph, pw in patches:
            patch = source[:, :, y:y + ph, x:x + pw].to(param.device, param.dtype)
            restored = net(patch)
            if restored.shape != (1, 3, ph * 4, pw * 4) or not torch.isfinite(restored).all().item():
                raise ValueError("Invalid SwinIR output; no fallback or acceptance")
            out[:, :, y * 4:(y + ph) * 4, x * 4:(x + pw) * 4] += restored.float().cpu()
            weight[:, :, y * 4:(y + ph) * 4, x * 4:(x + pw) * 4] += 1
    if not (weight > 0).all().item():
        raise ValueError("Incomplete SwinIR canvas")
    values = (out / weight)[0, :, :h * 4, :w * 4].clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray(np.rint(values * 255).astype(np.uint8))
