"""Fixed lighting-contrast diagnostic views; never a restored output or proof."""
import numpy as np
from PIL import Image, ImageFilter


def lighting_view(photo, max_edge=512):
    if not 64 <= max_edge <= 1024:
        raise ValueError("Bounded diagnostic size required")
    gray=photo.convert("L")
    gray.thumbnail((max_edge,max_edge),Image.Resampling.LANCZOS)
    fine=gray.filter(ImageFilter.GaussianBlur(2.))
    smooth=fine.filter(ImageFilter.GaussianBlur(20.))
    detail=np.log(16+np.asarray(fine,dtype=np.float32))-np.log(16+np.asarray(smooth,dtype=np.float32))
    # Fixed gain, never per-image normalization that manufactures contrast.
    result=np.clip(np.rint(128+900*detail),0,255).astype(np.uint8)
    return Image.fromarray(result).convert("RGB")
