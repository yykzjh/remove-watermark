"""Offline, SHA-bound decoder-only comparisons; never alters service defaults."""
import hashlib
import json
import re
from pathlib import Path


def logical_config_sha(config):
    config={k:v for k,v in config.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def strict_vae_weights(model,path,sha256):
    import torch
    from safetensors import safe_open
    path=Path(path).resolve(strict=True)
    if not re.fullmatch(r"[a-f0-9]{64}",sha256) or not 0<path.stat().st_size<=2*2**30:
        raise ValueError("Pinned bounded VAE artifact required")
    with path.open("rb") as f:
        if hashlib.file_digest(f,"sha256").hexdigest()!=sha256:
            raise ValueError("VAE artifact SHA mismatch")
    expected={k:tuple(v.shape) for k,v in model.state_dict().items()}
    with safe_open(path,framework="pt",device="cpu") as f:
        names=list(f.keys())
        if {k:tuple(f.get_slice(k).get_shape()) for k in names}!=expected:
            raise ValueError("Every VAE key and shape must match; no ignored weights")
        values={k:f.get_tensor(k) for k in names}
    if any(not v.is_floating_point() or not torch.isfinite(v).all().item() for v in values.values()):
        raise ValueError("Nonfinite or nonfloating VAE parameters")
    model.load_state_dict(values,strict=True,assign=True)
    return model.eval().requires_grad_(False)


def decoder_pair(original_model,weights,sha256,device,precision="bfloat16"):
    from types import SimpleNamespace

    import torch
    from diffusers import AutoencoderKLQwenImage21
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import QwenImage21Pipeline
    if precision not in ("bfloat16","float16"):
        raise ValueError("Explicit supported decoder precision required")
    config=json.loads((Path(original_model)/"vae/config.json").read_text())
    # Exact independently fetched publisher configuration, excluding descriptive
    # underscore metadata. Geometry, RGBA channels and latent normalization agree.
    if logical_config_sha(config)!="2c6365714ff1fd810232d2c1a037f8538f7a74b5d9018c948aeefa296ea26faa":
        raise ValueError("Original and texture-fix VAE logical configurations must agree")
    original=AutoencoderKLQwenImage21.from_pretrained(Path(original_model)/"vae",dtype=torch.bfloat16,local_files_only=True)
    with torch.device("meta"):
        modified=AutoencoderKLQwenImage21.from_config(original.config)
    modified=strict_vae_weights(modified,weights,sha256)
    original.to(device=device,dtype=torch.bfloat16)
    modified.to(device=device,dtype=getattr(torch,precision))
    def proxy(vae):
        return SimpleNamespace(vae=vae,vae_scale_factor=16,
            _unpack_latents=QwenImage21Pipeline._unpack_latents,
            image_processor=VaeImageProcessor(vae_scale_factor=16,vae_latent_channels=64))
    return proxy(original),proxy(modified)
