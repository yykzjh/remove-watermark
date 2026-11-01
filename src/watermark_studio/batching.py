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



def preencode_window(pipe, items, limit, encode=None):
    """Batch independent VLM prefill across a window, even when diffusion uses B1."""
    import torch

    if encode is None:
        from .optimizations import lean_encode
        encode=lean_encode
    stats={"cached_images":0,"oom_splits":0,"fallback_images":0}
    def run(group):
        failure=None
        try:
            prepared=[{"prompt":item["prompt"],"image":pipe.image_processor.resize(
                item["image"].convert("RGBA"),width=item["size"][0],height=item["size"][1])} for item in group]
            with torch.inference_mode():
                rows=encode(pipe,prepared)
            if len(rows)!=len(group) or any(not isinstance(row,tuple) or len(row)!=3 for row in rows):
                raise ValueError("Prefill count/protocol differs from independent inputs")
            for item,row in zip(group,rows):
                item["_encoded_prompt"]=row
                item["_prefill_batch_size"]=len(group)
            stats["cached_images"]+=len(group)
            return
        except Exception as error:  # noqa: BLE001 - split before falling back to normal per-image encoding
            failure=isinstance(error,torch.cuda.OutOfMemoryError)
        torch.cuda.empty_cache()
        if len(group)>1:
            stats["oom_splits"]+=int(failure)
            middle=len(group)//2
            run(group[:middle]);run(group[middle:])
        else:
            stats["fallback_images"]+=1
    for offset in range(0,len(items),limit):
        run(items[offset:offset+limit])
    return stats

def independent_batch(pipe, items, resolution, kv_cache):
    """Each item has its own image, prompt and seed. A worker calls this serially."""
    import torch
    from torch.nn import functional

    sizes = [dimensions(item["image"], resolution) for item in items]
    if len(set(sizes)) != 1:
        raise ValueError("Independent editing batch requires identical inference dimensions")
    width, height = sizes[0]
    original_prepare = pipe.prepare_latents
    encoded = [item.get("_encoded_prompt") for item in items]
    vae_images, prepared = [], []
    with torch.inference_mode():
        for index,item in enumerate(items):
            image = item["image"].convert("RGBA")
            resized = (pipe.image_processor.resize(image,width=width,height=height)
                       if encoded[index] is None else None)
            prepared.append({"prompt":item["prompt"],"image":resized})
            if encoded[index] is None and getattr(pipe,"_watermark_prompt_encoder","stock")=="stock":
                encoded[index]=pipe.encode_prompt(prompt=item["prompt"],image=[resized])
                item["_prefill_batch_size"]=1
            vae_images.append(
                pipe.image_processor.preprocess(image, width=width, height=height).unsqueeze(2)
            )
        if getattr(pipe, "_watermark_prompt_encoder", "stock") == "lean":
            from .optimizations import lean_encode
            chunk = getattr(pipe, "_watermark_encoder_batch_size", 1)
            missing=[i for i,row in enumerate(encoded) if row is None]
            for offset in range(0,len(missing),chunk):
                indices=missing[offset:offset+chunk]
                rows=lean_encode(pipe,[prepared[i] for i in indices])
                if len(rows)!=len(indices):
                    raise ValueError("Lean encoder returned wrong independent prefill count")
                for index,row in zip(indices,rows):
                    encoded[index]=row
                    items[index]["_prefill_batch_size"]=len(indices)
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
            separate=getattr(pipe,"_watermark_separate_vae_decode",False)
            output=pipe(
                prompt=[item["prompt"] for item in items], image=items[0]["image"],
                height=height, width=width, output_resolution=resolution,
                true_cfg_scale=1.0, use_kv_cache=kv_cache, generator=generators,
                output_type="latent" if separate else "pil",
            ).images
        # Returning from the pipeline releases its local prefix K/V and denoising
        # tensors before allocating VAE activations; sampling arithmetic is unchanged.
        return decode_latents(pipe,output,height,width) if separate else output


def decode_latents(pipe,latents,height,width):
    """Exact decode arithmetic from the pinned QwenImage21 pipeline."""
    import torch
    latent_h=2*(int(height)//(pipe.vae_scale_factor*2))
    latent_w=2*(int(width)//(pipe.vae_scale_factor*2))
    if latents.ndim!=3 or latents.shape[1]!=latent_h*latent_w:
        raise ValueError("Latent sequence does not match the requested full-frame grid")
    with torch.inference_mode():
        latents=pipe._unpack_latents(latents,height,width,pipe.vae_scale_factor).to(pipe.vae.dtype)
        mean=torch.tensor(pipe.vae.config.latents_mean).view(1,pipe.vae.config.z_dim,1,1,1).to(latents.device,latents.dtype)
        std=torch.tensor(pipe.vae.config.latents_std).view(1,pipe.vae.config.z_dim,1,1,1).to(latents.device,latents.dtype)
        latents=latents*std+mean
        image=pipe.vae.decode(latents,return_dict=False)[0][:,:,0]
        return pipe.image_processor.postprocess(image,output_type="pil")
