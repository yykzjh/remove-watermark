"""Independent-image batching for pinned QwenImage21. Never broadcast reference pixels."""
from contextlib import contextmanager


def dimensions(image, resolution):
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
    width, height, _ = calculate_dimensions(resolution * resolution, image.width / image.height)
    return width, height


@contextmanager
def override_methods(pipe, **methods):
    previous = {name: pipe.__dict__.get(name) for name in methods}
    owned = {name: name in pipe.__dict__ for name in methods}
    try:
        for name, method in methods.items():
            setattr(pipe, name, method)
        yield
    finally:
        for name in methods:
            if owned[name]:
                setattr(pipe, name, previous[name])
            else:
                delattr(pipe, name)


def independent_batch(pipe, items, resolution, kv_cache):
    """Each item has its own image, prompt and seed. A worker calls this serially."""
    import torch
    from torch.nn import functional

    sizes = [dimensions(item["image"], resolution) for item in items]
    if len(set(sizes)) != 1:
        raise ValueError("Independent editing batch requires identical inference dimensions")
    width, height = sizes[0]
    original_prepare = pipe.prepare_latents
    encoded, vae_images, prepared = [], [], []
    with torch.inference_mode():
        for item in items:
            image = item["image"].convert("RGBA")
            resized = pipe.image_processor.resize(image, width=width, height=height)
            prepared.append({"prompt": item["prompt"], "image": resized})
            if getattr(pipe, "_watermark_prompt_encoder", "stock") == "stock":
                encoded.append(pipe.encode_prompt(prompt=item["prompt"], image=[resized]))
            vae_images.append(
                pipe.image_processor.preprocess(image, width=width, height=height).unsqueeze(2)
            )
        if getattr(pipe, "_watermark_prompt_encoder", "stock") == "lean":
            from .optimizations import lean_encode
            chunk = getattr(pipe, "_watermark_encoder_batch_size", 1)
            for offset in range(0, len(prepared), chunk):
                encoded.extend(lean_encode(pipe, prepared[offset:offset + chunk]))
        max_length = max(row[0].shape[1] for row in encoded)
        embeddings, masks, image_masks = [], [], []
        for embedding, mask, image_mask in encoded:
            length = embedding.shape[1]
            if mask is None:
                mask = torch.ones(embedding.shape[:2], device=embedding.device, dtype=torch.long)
            embeddings.append(functional.pad(embedding, (0, 0, 0, max_length - length)))
            masks.append(functional.pad(mask, (0, max_length - length)))
            image_masks.append(functional.pad(image_mask, (0, max_length - length)))
        # Transformer reads spatial/rotary layout from row zero: fail closed on mismatch.
        if any(not torch.equal(image_masks[0], mask) for mask in image_masks[1:]):
            raise ValueError("Image-token positions differ; independent batch layout is unsafe")
        combined_mask = torch.cat(masks)
        if combined_mask.all():
            combined_mask = None
        combined = (torch.cat(embeddings), combined_mask, torch.cat(image_masks))
        vae_batch = torch.cat(vae_images)

        def encode_prompt(**kwargs):
            return combined

        def prepare_latents(images, *args, **kwargs):
            return original_prepare([vae_batch], *args, **kwargs)

        generators = [torch.Generator("cpu").manual_seed(item["seed"]) for item in items]
        with override_methods(pipe, encode_prompt=encode_prompt, prepare_latents=prepare_latents):
            return pipe(
                prompt=[item["prompt"] for item in items], image=items[0]["image"],
                height=height, width=width, output_resolution=resolution,
                true_cfg_scale=1.0, use_kv_cache=kv_cache, generator=generators,
            ).images
