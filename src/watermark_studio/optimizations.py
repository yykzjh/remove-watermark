"""Opt-in optimizations for the pinned QwenImage21 implementation."""
import torch
from diffusers.models.transformers.transformer_qwenimage21 import (
    _qwenimage21_prepare_qkv,
    apply_rotary_emb_qwen,
)


def fuse_projections(transformer):
    for block in transformer.transformer_blocks:
        attn = block.attn
        if hasattr(attn, "to_qkv"):
            continue
        merged = torch.nn.Linear(attn.to_q.in_features, 3 * attn.to_q.out_features,
                                 bias=False, device=attn.to_q.weight.device,
                                 dtype=attn.to_q.weight.dtype)
        with torch.no_grad():
            merged.weight.copy_(torch.cat([attn.to_q.weight, attn.to_k.weight, attn.to_v.weight]))
        attn.to_qkv = merged
        del attn.to_q, attn.to_k, attn.to_v


def prepare_qkv(attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice):
    if not hasattr(attn, "to_qkv"):
        return _qwenimage21_prepare_qkv(
            attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice
        )
    query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
    query = attn.norm_q(query.unflatten(-1, (attn.heads, -1))).to(value.dtype)
    key = attn.norm_k(key.unflatten(-1, (attn.heads, -1))).to(value.dtype)
    value = value.unflatten(-1, (attn.heads, -1))
    if rotary_emb is not None:
        query = apply_rotary_emb_qwen(query, rotary_emb, use_real=False)
        key = apply_rotary_emb_qwen(key, rotary_emb, use_real=False)
    if layer_cache is not None:
        if kv_cache_mode == "extract" and cache_write_slice is not None:
            layer_cache.store(key[:, cache_write_slice].clone(), value[:, cache_write_slice].clone())
        elif kv_cache_mode == "cached":
            cached_k, cached_v = layer_cache.get()
            key = torch.cat([cached_k, key], dim=1)
            value = torch.cat([cached_v, value], dim=1)
    return query, key, value, query.shape[1]


def quantize_transformer(transformer, mode):
    try:
        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            Int8WeightOnlyConfig,
            quantize_,
        )
    except ImportError as error:
        raise RuntimeError("Quantization requires uv sync --extra server --extra quantization") from error
    config = Float8DynamicActivationFloat8WeightConfig() if mode == "float8" else Int8WeightOnlyConfig()
    # Leave input/output and modulation projections in the model's original precision.
    quantize_(transformer, config, filter_fn=lambda module, name:
              isinstance(module, torch.nn.Linear) and name.startswith("transformer_blocks."))


def lean_encode(pipe, items):
    """Independent images, no LM head, no unused hidden states or generation KV cache."""
    from PIL import Image
    prompts, images = [], []
    for item in items:
        prompts.append(pipe.prompt_template_ti2i.format(item["prompt"] or " "))
        image = item["image"]
        if image.mode == "RGBA":
            white = Image.new("RGB", image.size, "white")
            white.paste(image, mask=image.getchannel("A"))
            image = white
        images.append(image)
    inputs = pipe.processor(text=prompts, images=images, padding=True, padding_side="left",
                            return_tensors="pt").to(pipe._execution_device)
    kwargs = {key: inputs[key] for key in
              ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")
              if key in inputs}
    model = pipe.text_encoder.model
    language = getattr(model, "language_model", model)
    handle = language.norm.register_forward_hook(lambda module, args, output: args[0])
    try:
        hidden = model(**kwargs, output_hidden_states=False, use_cache=False,
                       return_dict=True).last_hidden_state
    finally:
        handle.remove()
    rows = []
    for index, valid in enumerate(inputs.attention_mask.bool()):
        embedding = hidden[index][valid][pipe._drop_idx:]
        ids = inputs.input_ids[index][valid][pipe._drop_idx:]
        rows.append((embedding[None], torch.ones((1, len(ids)), device=hidden.device, dtype=torch.long),
                     (ids == pipe._img_token_id)[None]))
    return rows
