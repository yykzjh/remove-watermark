"""Offline-only Qwen21 Fun ControlNet probe; never enabled by the service.

Equations follow VideoX-Fun e8c314fb's qwenimage21_transformer2d_control.py.
The independent control chain is evaluated at each injection rather than stacked
up front: identical dependencies, only one running control state in memory.
No prefix KV cache: the pretrained branch changes the text stream every step.
"""
from contextlib import contextmanager
from pathlib import Path


def bounded_caption(text, max_words=80, *, reached_limit=False):
    """Compress generation conditioning to complete observed sentences.

    A caption is conditioning, never a quality verdict or identity certificate.
    Discard an unfinished tail instead of treating it as a complete fact. Never
    invent a generic scene when no usable observation was generated.
    """
    import hashlib
    import re
    text=text.strip()
    if not text or not 1<=max_words<=120:
        raise ValueError("Missing bounded source caption")
    sentences=[m.group().strip() for m in re.finditer(r".*?[.!?。！？](?=\s|$)",text,re.DOTALL)]
    selected=[];words=0
    for sentence in sentences:
        size=len(sentence.split())
        if words+size>max_words:
            break
        selected.append(sentence);words+=size
    if not selected and not reached_limit and len(text.split())<=max_words:
        selected=[text]
    if not selected:
        raise ValueError("No complete bounded source observation; no generic fallback")
    return " ".join(selected),{"source_generation_reached_cap":reached_limit,
        "conditioning_words":len(" ".join(selected).split()),
        "source_text_sha256":hashlib.sha256(text.encode()).hexdigest(),
        "quality_permission":False,"method":"complete_source_sentences_v1"}


def make_branch(config, layers):
    import torch
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TransformerBlock

    class ControlBlock(QwenImage21TransformerBlock):
        def __init__(self, first):
            dim=config.num_attention_heads*config.attention_head_dim
            super().__init__(dim,config.num_attention_heads,config.attention_head_dim,
                             config.mlp_ratio,config.eps)
            if first:
                self.before_proj=torch.nn.Linear(dim,dim)
            self.after_proj=torch.nn.Linear(dim,dim)

    branch=torch.nn.Module()
    branch.control_img_in=torch.nn.Linear(129,config.num_attention_heads*config.attention_head_dim)
    branch.control_blocks=torch.nn.ModuleList([ControlBlock(i==0) for i in range(len(layers))])
    return branch


def load_branch(transformer, path, expected_sha256):
    import hashlib
    import re
    import time

    import torch
    from safetensors import safe_open

    began=time.monotonic()
    if (len(transformer.transformer_blocks)!=32 or transformer.inner_dim!=4096
            or not re.fullmatch(r"[0-9a-f]{64}",expected_sha256)):
        raise ValueError("Pinned Qwen21 32-block architecture and artifact SHA required")
    path=Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        actual=hashlib.file_digest(stream,"sha256").hexdigest()
    if actual!=expected_sha256:
        raise ValueError("Control branch artifact SHA differs")
    layers=tuple(range(0,32,2))
    with torch.device("meta"):
        branch=make_branch(transformer.config,layers)
    expected={name:tuple(t.shape) for name,t in branch.state_dict().items()}
    with safe_open(path,framework="pt",device="cpu") as handle:
        names=list(handle.keys())  # safe_open is not a Mapping iterator.
        actual_shapes={name:tuple(handle.get_slice(name).get_shape()) for name in names}
        if expected!=actual_shapes:
            raise ValueError("Every control key and shape must match; no ignored weights")
        values={name:handle.get_tensor(name) for name in names}
        if any(not t.is_floating_point() or not torch.isfinite(t).all().item() for t in values.values()):
            raise ValueError("Nonfloating or nonfinite control weights")
    branch.load_state_dict(values,strict=True,assign=True)
    branch.requires_grad_(False).eval()
    return branch,{"sha256":actual,"tensors":len(values),"layers":list(layers),
                   "channels":129,"kv_cache":False,"seconds":round(time.monotonic()-began,3)}


@contextmanager
def control_forward(transformer, branch, context, scale=1.):
    """Single worker serial forward scope; remove all hooks even on a failed step."""
    import math

    import torch
    if (not math.isfinite(scale) or not 0<=scale<=2 or context.ndim!=3 or context.shape[-1]!=129
            or not torch.isfinite(context).all().item()):
        raise ValueError("Finite packed B,T,129 context and scale0..2 required")
    count=len(transformer.transformer_blocks)
    layers=tuple(range(0,count,2))
    if len(branch.control_blocks)!=len(layers):
        raise ValueError("Every other base block needs one matching control block")
    running={}
    handles=[]

    def begin(module,args,kwargs):
        if kwargs.get("kv_cache") is not None or kwargs.get("kv_cache_mode") is not None:
            raise ValueError("Fun ControlNet recomputes the full stream; prefix KV cache is invalid")
        running.clear()

    def first(module,args,kwargs):
        if scale==0:
            return
        x=kwargs.get("hidden_states",args[0] if args else None)
        target=kwargs.get("target_token_mask")
        if (x is None or target is None or target.ndim!=1 or x.shape[0]!=context.shape[0]
                or int(target.sum())!=context.shape[1] or target.shape[0]!=x.shape[1]):
            raise ValueError("Full target token positions must match each independent control row")
        c=torch.zeros_like(x)
        c[:,target]=branch.control_img_in(context.to(device=x.device,dtype=x.dtype))
        running["c"]=branch.control_blocks[0].before_proj(c)+x
        running["kwargs"]={key:kwargs.get(key) for key in (
            "modulation","rotary_emb","attention_mask","target_token_mask","segments","key_valid")}

    def inject(index):
        def hook(module,args,kwargs,output):
            if scale==0:
                return output
            block=branch.control_blocks[index]
            running["c"]=block(running["c"],**running["kwargs"])
            return output+block.after_proj(running["c"])*scale
        return hook

    try:
        handles.append(transformer.register_forward_pre_hook(begin,with_kwargs=True))
        handles.append(transformer.transformer_blocks[0].register_forward_pre_hook(first,with_kwargs=True))
        for index,layer in enumerate(layers):
            handles.append(transformer.transformer_blocks[layer].register_forward_hook(inject(index),with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()
        running.clear()


def packed_inpaint_context(pipe, image, mask, width, height):
    """Official pure-inpaint 0-control|keep-mask|masked-RGBA-VAE order, full frame."""
    import torch
    from torch.nn import functional as F
    device=pipe._execution_device
    dtype=pipe.transformer.dtype
    # Mask white regenerates pixels; the branch's latent mask is inverted (1=keep).
    pixels=pipe.image_processor.preprocess(image.convert("RGB"),height=height,width=width).to(device,dtype)
    mask=mask.convert("L").resize((width,height),resample=0)
    import numpy as np
    edit=torch.from_numpy(np.asarray(mask).copy()).to(device=device).ge(128)[None,None].to(dtype)
    masked=pixels*(edit<.5)
    rgba=torch.cat([masked,torch.ones_like(masked[:,:1])],dim=1).unsqueeze(2)
    latent=pipe._encode_vae_image(rgba,generator=None)
    keep=F.interpolate(1-edit,size=latent.shape[-2:],mode="nearest").unsqueeze(2)
    combined=torch.cat([torch.zeros_like(latent),keep,latent],dim=1)
    return pipe._pack_latents(combined,1,129,*latent.shape[-2:])


@contextmanager
def source_flow_context(pipe,source,regions,width,height,enabled=False):
    """Anchor the unmasked target trajectory without adding a generation pass.

    Pure ControlNet has no original-image prefix to reuse. It therefore needs
    one explicitly recorded full-source VAE encode; an original-image prefix
    can be reused when present. Never return that extra reference as a new
    prefix, or alter the frozen Turbo sigma schedule.
    """
    if not enabled:
        yield {},{}
        return
    import torch

    from .batching import override_methods
    from .denoising_context import DenoisingContext
    if type(pipe.scheduler).__name__!="FlowMatchEulerDiscreteScheduler" or getattr(pipe.scheduler.config,"stochastic_sampling",False):
        raise ValueError("Deterministic flow scheduler required")
    item={"_denoising_context_regions":[r.model_dump() for r in regions]}
    ctx=DenoisingContext([item],width,height,pipe.vae_scale_factor)
    original_prepare=pipe.prepare_latents;original_schedule=pipe.scheduler.set_timesteps
    audit={"enabled":True,"extra_vae_encodes":0,"extra_diffusion_forwards":0}
    def prepare(images,*args,**kwargs):
        noise,prefix=original_prepare(images,*args,**kwargs)
        reference=prefix
        if reference is None:
            pixels=pipe.image_processor.preprocess(source.convert("RGB"),height=height,width=width).to(noise.device,noise.dtype)
            rgba=torch.cat([pixels,torch.ones_like(pixels[:,:1])],dim=1).unsqueeze(2)
            reference=pipe._encode_vae_image(rgba,generator=None)
            reference=pipe._pack_latents(reference,1,64,*reference.shape[-2:]).to(noise.device,noise.dtype)
            audit["extra_vae_encodes"]=1
        ctx.capture(noise,reference)
        return noise,prefix
    def schedule(*args,**kwargs):
        result=original_schedule(*args,**kwargs)
        ctx.schedule(pipe.scheduler.sigmas)
        return result
    with override_methods(pipe,prepare_latents=prepare),override_methods(pipe.scheduler,set_timesteps=schedule):
        yield {"callback_on_step_end":ctx.after_step,"callback_on_step_end_tensor_inputs":["latents"]},audit
        ctx.finish()
        item["_denoising_context_log"]["extra_vae_encodes"]=audit["extra_vae_encodes"]
        audit.update(item["_denoising_context_log"])
