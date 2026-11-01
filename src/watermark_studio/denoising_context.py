"""Optional flow-matching context preservation for the pinned QwenImage21 pipeline."""
from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
from PIL import Image, ImageDraw

from .config import Region
from .images import pixel_box


def pixel_edit_masks(items, size):
    masks = []
    for item in items:
        regions = item.get("_denoising_context_regions")
        if not isinstance(regions, list) or len(regions) > 8:
            raise ValueError("Independent bounded source regions required for every image")
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        for value in regions:
            region = value if isinstance(value, Region) else Region.model_validate(value)
            l, t, r, b = pixel_box(region.box, size)
            draw.rectangle((l, t, r-1, b-1), fill=255)
        masks.append(np.asarray(mask, dtype=np.float32)/255)
    return np.stack(masks)


class DenoisingContext:
    """Reuse the already encoded VAE reference; never add a model/image encode."""

    def __init__(self, items, width, height, vae_scale_factor):
        self.items = items
        self.width, self.height = width, height
        self.latent_h = 2*(height//(vae_scale_factor*2))
        self.latent_w = 2*(width//(vae_scale_factor*2))
        self.pixel_masks = pixel_edit_masks(items, (width, height))
        self.completed = 0
        self.sigmas = None
        self.mask = self.clean = self.noise = self.initial = None

    def capture(self, noise, reference):
        import torch
        from torch.nn import functional

        expected = (len(self.items), self.latent_h*self.latent_w)
        if (reference is None or noise.ndim != 3 or reference.shape != noise.shape
                or tuple(noise.shape[:2]) != expected):
            raise ValueError("One matching full-frame VAE reference per independent image required")
        if noise.device != reference.device or noise.dtype != reference.dtype:
            raise ValueError("Context and target latents must share device and dtype")
        mask = torch.from_numpy(self.pixel_masks).unsqueeze(1).to(noise.device)
        mask = functional.adaptive_max_pool2d(mask, (self.latent_h, self.latent_w))
        # One latent-cell context margin protects the VAE transition; no cropping.
        mask = functional.max_pool2d(mask, 3, stride=1, padding=1)
        self.mask = mask.flatten(2).transpose(1, 2).to(noise.dtype)
        self.clean = reference.detach()
        self.noise = noise.detach().clone()
        self.initial = noise
        return noise, reference

    def schedule(self, sigmas):
        values = sigmas.detach().float().cpu().tolist()
        if (len(values) < 2 or any(not math.isfinite(s) or not 0 <= s <= 1 for s in values)
                or any(a < b for a, b in pairwise(values)) or abs(values[-1]) > 1e-7):
            raise ValueError("Descending flow sigmas with a zero terminal value required")
        if self.initial is None:
            raise ValueError("VAE reference must be captured before scheduling")
        self.sigmas = values
        # prepare_latents runs before set_timesteps in the pinned pipeline.
        # Make its returned tensor consistent with the actual shifted first sigma.
        self.initial.copy_(self.blend(self.noise, values[0]))
        self.initial = None

    def blend(self, current, sigma):
        if self.clean is None or self.mask is None:
            raise ValueError("No trusted source context captured")
        if current.shape != self.clean.shape:
            raise ValueError("Denoising latent layout changed")
        reference = (1-sigma)*self.clean + sigma*self.noise
        return self.mask*current + (1-self.mask)*reference

    def after_step(self, pipe, index, timestep, values):
        del timestep
        step = pipe.scheduler.step_index
        if (self.sigmas is None or step != index+1 or index != self.completed
                or step >= len(self.sigmas)):
            raise ValueError("Unexpected scheduler/callback position")
        current = values["latents"]
        values["latents"] = self.blend(current, self.sigmas[step])
        self.completed += 1
        return values

    def finish(self):
        if self.sigmas is None or self.completed != len(self.sigmas)-1:
            raise ValueError("Incomplete denoising context trajectory")
        fractions = self.mask.float().mean(dim=(1, 2)).detach().cpu().tolist()
        for item, fraction in zip(self.items, fractions):
            item["_denoising_context_log"] = {
                "method": "full_frame_flow_context_v1",
                "steps": self.completed, "runtime_sigmas": self.sigmas,
                "editable_latent_fraction": fraction,
                "extra_vae_encodes": 0, "extra_diffusion_forwards": 0,
            }
