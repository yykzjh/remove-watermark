"""Offline small-overlay inpainting experiment; never used by the serving recipe."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def tiny_eligible(edit, maximum_area=.035, maximum_resolution=512):
    """Route only from original geometry, never from generated pixels or QA."""
    state = edit["state"]
    if state["whole_generation"] or state["source_scene"] != "outdoor":
        return False
    if edit.get("_roi_members") or edit.get("prefilled") or edit["resolution"] > maximum_resolution:
        return False
    left, top, right, bottom = edit["box"]
    return 0 < (right-left)*(bottom-top) <= maximum_area*state["source"].width*state["source"].height


def white_glyph_support(image, box):
    """Source-only white-logo stencil; ambiguous coverage keeps the full declared box."""
    left, top, right, bottom = box
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    local = rgb[top:bottom, left:right]
    bright = (local.min(axis=2) >= 195) & (local.max(axis=2)-local.min(axis=2) <= 55)
    ratio = float(bright.mean()) if bright.size else 0
    scope = Image.new("L", image.size)
    if not .015 <= ratio <= .45:
        scope.paste(255, box)
        return scope, "box_ambiguous_white_coverage"
    # Cover antialiased rims and the small drop shadow, without replacing the
    # naturally visible gaps between camera-logo letters.
    local_mask = Image.fromarray(bright.astype(np.uint8)*255)
    radius = max(2, min(6, round((bottom-top)*.035)))
    local_mask = local_mask.filter(ImageFilter.MaxFilter(2*radius+1))
    scope.paste(local_mask, (left, top))
    return scope, "white_glyph_source_stencil"


def masked_inputs(edit, mask_style="box"):
    """Use original context and a hard, half-open mask at the declared edit box."""
    image = edit["image"].convert("RGB")
    left, top, right, bottom = edit["crop"]
    if image.size != (right-left, bottom-top):
        raise ValueError("Context image and source geometry disagree")
    width, height = edit["size"]
    source = np.asarray(image.resize((width, height), Image.Resampling.LANCZOS), dtype=np.float32)/255
    box = edit["box"]
    local_box = (box[0]-left, box[1]-top, box[2]-left, box[3]-top)
    if mask_style == "white-glyph":
        mask, _ = white_glyph_support(image, local_box)
    elif mask_style == "box":
        mask = Image.new("L", image.size)
        mask.paste(255, local_box)
    else:
        raise ValueError("Unknown experimental mask style")
    mask = np.asarray(mask.resize((width, height), Image.Resampling.NEAREST)) > 0
    ph, pw = (-height) % 8, (-width) % 8
    return np.pad(source, ((0, ph), (0, pw), (0, 0)), mode="reflect"), np.pad(mask, ((0, ph), (0, pw)), mode="edge")


def native_stencil_composite(edit, patch):
    """Keep every source pixel outside the source-derived stencil at native size."""
    original = edit["image"].convert("RGB")
    left, top, _, _ = edit["crop"]
    box = edit["box"]
    local_box = (box[0]-left, box[1]-top, box[2]-left, box[3]-top)
    mask, _ = white_glyph_support(original, local_box)
    generated = patch.resize(original.size, Image.Resampling.LANCZOS)
    return Image.composite(generated, original, mask)


class TinyInpainter:
    def __init__(self, path, sha256, device, mask_style="box"):
        import torch

        path = Path(path)
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != sha256:
                raise ValueError("Auxiliary inpainting weights do not match pinned SHA256")
        self.model = torch.jit.load(str(path), map_location=device).eval()
        self.device, self.sha256, self.mask_style = device, sha256, mask_style

    def __call__(self, edit):
        import torch

        source, mask = masked_inputs(edit, self.mask_style)
        tensor = torch.from_numpy(source.transpose(2, 0, 1).copy())[None].to(self.device)
        support = torch.from_numpy(mask.astype(np.float32))[None, None].to(self.device)
        with torch.inference_mode():
            output = self.model(tensor, support)
        if output.shape != tensor.shape or not torch.isfinite(output).all():
            raise ValueError("Invalid auxiliary inpainting output")
        width, height = edit["size"]
        rgb = output[0, :, :height, :width].permute(1, 2, 0).float().cpu().numpy()
        patch = Image.fromarray(np.clip(rgb*255, 0, 255).astype(np.uint8))
        if self.mask_style == "white-glyph":
            patch = native_stencil_composite(edit, patch)
        edit["_sampling_log"] = {"backend": "experimental_big_lama", "actual_steps": 0,
                                 "weights_sha256": self.sha256, "source_geometry_only": True,
                                 "mask_style": self.mask_style, "mask_fraction": float(mask.mean())}
        return patch
