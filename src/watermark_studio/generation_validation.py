"""Reject non-finite numerical generations before converting them into valid-looking pixels."""
from contextlib import ExitStack, contextmanager

import numpy as np


class NonFiniteGenerationError(RuntimeError):
    pass


def require_finite(value,stage):
    if isinstance(value,np.ndarray):
        finite=bool(np.isfinite(value).all())
    else:
        import torch

        finite=bool(torch.isfinite(value).all().item())
    if not finite:
        raise NonFiniteGenerationError(f"Non-finite {stage}; no image may be written")


@contextmanager
def finite_generation_scope(pipe):
    from .batching import override_methods

    with ExitStack() as stack:
        vae=getattr(pipe,"vae",None)
        decode=getattr(vae,"decode",None)
        if callable(decode):
            def checked_decode(*args,**kwargs):
                latents=args[0] if args else kwargs.get("z",kwargs.get("latents"))
                if latents is not None:
                    require_finite(latents,"VAE input")
                result=decode(*args,**kwargs)
                image=result[0] if isinstance(result,tuple) else getattr(result,"sample",None)
                if image is not None:
                    require_finite(image,"VAE output")
                return result
            stack.enter_context(override_methods(vae,decode=checked_decode))
        processor=getattr(pipe,"image_processor",None)
        convert=getattr(processor,"numpy_to_pil",None)
        if callable(convert):
            def checked_convert(images,*args,**kwargs):
                require_finite(np.asarray(images),"image conversion input")
                return convert(images,*args,**kwargs)
            stack.enter_context(override_methods(processor,numpy_to_pil=checked_convert))
        yield
