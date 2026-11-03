"""Offline small-overlay inpainting experiment; never used by the serving recipe."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def tiny_eligible(edit, maximum_area=.035):
    """Route only from original geometry, never from generated pixels or QA."""
    state = edit["state"]
    if state["whole_generation"] or state["source_scene"] != "outdoor":
        return False
    if edit.get("_roi_members") or edit.get("prefilled") or edit["resolution"] > 512:
        return False
    left, top, right, bottom = edit["box"]
    return 0 < (right-left)*(bottom-top) <= maximum_area*state["source"].width*state["source"].height


def masked_inputs(edit):
    """Use original context and a hard, half-open mask at the declared edit box."""
    image = edit["image"].convert("RGB")
    left, top, right, bottom = edit["crop"]
    if image.size != (right-left, bottom-top):
        raise ValueError("Context image and source geometry disagree")
    width, height = edit["size"]
    source = np.asarray(image.resize((width, height), Image.Resampling.LANCZOS), dtype=np.float32)/255
    box = edit["box"]
    mask = Image.new("L", image.size)
    mask.paste(255, (box[0]-left, box[1]-top, box[2]-left, box[3]-top))
    mask = np.asarray(mask.resize((width, height), Image.Resampling.NEAREST)) > 0
    ph, pw = (-height) % 8, (-width) % 8
    return np.pad(source, ((0, ph), (0, pw), (0, 0)), mode="reflect"), np.pad(mask, ((0, ph), (0, pw)), mode="edge")


class TinyInpainter:
    def __init__(self, path, sha256, device):
        import torch

        path = Path(path)
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != sha256:
                raise ValueError("Auxiliary inpainting weights do not match pinned SHA256")
        self.model = torch.jit.load(str(path), map_location=device).eval()
        self.device, self.sha256 = device, sha256

    def __call__(self, edit):
        import torch

        source, mask = masked_inputs(edit)
        tensor = torch.from_numpy(source.transpose(2, 0, 1).copy())[None].to(self.device)
        support = torch.from_numpy(mask.astype(np.float32))[None, None].to(self.device)
        with torch.inference_mode():
            output = self.model(tensor, support)
        if output.shape != tensor.shape or not torch.isfinite(output).all():
            raise ValueError("Invalid auxiliary inpainting output")
        width, height = edit["size"]
        rgb = output[0, :, :height, :width].permute(1, 2, 0).float().cpu().numpy()
        patch = Image.fromarray(np.clip(rgb*255, 0, 255).astype(np.uint8))
        edit["_sampling_log"] = {"backend": "experimental_big_lama", "actual_steps": 0,
                                 "weights_sha256": self.sha256, "source_geometry_only": True}
        return patch
