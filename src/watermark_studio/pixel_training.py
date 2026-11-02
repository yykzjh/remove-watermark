"""Optional offline RGB supervision through a frozen native Qwen21 decoder."""
from __future__ import annotations


def packed_to_vae(packed,height,width,mean,std):
    if packed.ndim!=3 or packed.shape[1]!=(height//16)*(width//16) or height%32 or width%32:
        raise ValueError("Complete aligned Qwen21 pixel supervision grid required")
    image=packed.transpose(1,2).reshape(packed.shape[0],packed.shape[2],1,height//16,width//16)
    if mean.shape!=(1,packed.shape[2],1,1,1) or std.shape!=mean.shape:
        raise ValueError("Matching native VAE normalization required")
    return image*std+mean


def exact_pixel_objective(prediction,target,mask):
    """True clean RGB L1 plus local texture amplitude; never a QA score."""
    import torch
    from torch.nn import functional
    if (prediction.ndim!=4 or prediction.shape!=target.shape or prediction.shape[1]!=3
            or mask.shape!=(prediction.shape[0],1,*prediction.shape[2:])
            or prediction.shape[2]%8 or prediction.shape[3]%8):
        raise ValueError("Aligned complete RGB target and exact pixel support required")
    if (not torch.isfinite(target).all() or not torch.isfinite(mask).all()
            or not ((mask>=0)&(mask<=1)).all()):
        raise ValueError("Finite true clean pixels and support in0..1 required")
    weights=1+3*mask.float()
    def average(value,weight):
        return (value*weight).sum()/(value.shape[1]*weight.sum())
    l1=average((prediction.float()-target.float()).abs(),weights)
    def texture(image):
        dx=functional.pad((image[...,1:]-image[...,:-1]).abs(),(0,1,0,0))
        dy=functional.pad((image[...,1:,:]-image[...,:-1,:]).abs(),(0,0,0,1))
        return functional.avg_pool2d(dx+dy,8)
    roughness=average((texture(prediction.float())-texture(target.float())).abs(),
        functional.avg_pool2d(weights,8))
    return l1+.25*roughness


def decode_exact_pixels(vae,packed,height,width,*,checkpoint=False):
    import torch
    from torch.utils.checkpoint import checkpoint as gradient_checkpoint
    dtype=vae.dtype
    mean=torch.tensor(vae.config.latents_mean,device=packed.device,dtype=dtype).view(1,-1,1,1,1)
    std=torch.tensor(vae.config.latents_std,device=packed.device,dtype=dtype).view(1,-1,1,1,1)
    normalized=packed_to_vae(packed.to(dtype),height,width,mean,std)
    def decode(value):
        return vae.decode(value,return_dict=False)[0][:,:3,0]
    result=gradient_checkpoint(decode,normalized,use_reentrant=False) if checkpoint else decode(normalized)
    if torch.is_grad_enabled() and packed.requires_grad and not result.requires_grad:
        raise ValueError("Native frozen decoder disconnected the RGB supervision graph")
    return result
