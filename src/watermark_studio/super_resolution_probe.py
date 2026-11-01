"""Offline learned x4 restoration; no diffusion generation or quality permission.

RRDB equations adapted from BasicSR, Copyright 2018-2022 BasicSR Authors,
Apache-2.0; see third_party/BasicSR/LICENSE.txt and NOTICE.txt.
Modified: inference-only construction, strict artifact checks, bounded tiles.
"""
from pathlib import Path


def make_rrdbnet(features=64,growth=32,blocks=23):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Dense(nn.Module):
        def __init__(self):
            super().__init__()
            for i in range(5):
                setattr(self,"conv"+str(i+1),nn.Conv2d(features+i*growth,growth if i<4 else features,3,1,1))
        def forward(self,x):
            values=[x]
            for i in range(4):
                values.append(F.leaky_relu(getattr(self,"conv"+str(i+1))(torch.cat(values,1)),.2,inplace=True))
            return self.conv5(torch.cat(values,1))*.2+x

    class Residual(nn.Module):
        def __init__(self):
            super().__init__()
            self.rdb1=Dense();self.rdb2=Dense();self.rdb3=Dense()
        def forward(self,x):
            return self.rdb3(self.rdb2(self.rdb1(x)))*.2+x

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_first=nn.Conv2d(3,features,3,1,1)
            self.body=nn.Sequential(*[Residual() for _ in range(blocks)])
            self.conv_body=nn.Conv2d(features,features,3,1,1)
            self.conv_up1=nn.Conv2d(features,features,3,1,1)
            self.conv_up2=nn.Conv2d(features,features,3,1,1)
            self.conv_hr=nn.Conv2d(features,features,3,1,1)
            self.conv_last=nn.Conv2d(features,3,3,1,1)
        def forward(self,x):
            x=self.conv_first(x)
            x=x+self.conv_body(self.body(x))
            x=F.leaky_relu(self.conv_up1(F.interpolate(x,scale_factor=2,mode="nearest")),.2,inplace=True)
            x=F.leaky_relu(self.conv_up2(F.interpolate(x,scale_factor=2,mode="nearest")),.2,inplace=True)
            return self.conv_last(F.leaky_relu(self.conv_hr(x),.2,inplace=True))
    return Network().eval().requires_grad_(False)


def load_upscaler(path,expected_sha256,device):
    import hashlib
    import re

    import torch
    from safetensors import safe_open
    path=Path(path).resolve(strict=True)
    if not re.fullmatch(r"[a-f0-9]{64}",expected_sha256) or not 0<path.stat().st_size<128*2**20:
        raise ValueError("Pinned bounded safetensors restoration artifact required")
    with path.open("rb") as f:
        if hashlib.file_digest(f,"sha256").hexdigest()!=expected_sha256:
            raise ValueError("Restoration artifact SHA mismatch")
    with torch.device("meta"):
        net=make_rrdbnet()
    expected={k:tuple(v.shape) for k,v in net.state_dict().items()}
    with safe_open(path,framework="pt",device="cpu") as f:
        names=list(f.keys())  # safe_open is not a Mapping iterator.
        if {k:tuple(f.get_slice(k).get_shape()) for k in names}!=expected:
            raise ValueError("Every RGB x4 RRDB key and shape must match")
        values={k:f.get_tensor(k) for k in names}
    if any(not v.is_floating_point() or not torch.isfinite(v).all().item() for v in values.values()):
        raise ValueError("Nonfinite or nonfloating restoration parameters")
    net.load_state_dict(values,strict=True,assign=True)
    net.to(device=device,dtype=torch.float16 if str(device).startswith("cuda") else torch.float32)
    if any(not torch.isfinite(v).all().item() for v in net.parameters()):
        raise ValueError("Restoration parameters overflow the selected precision")
    return net


def upscale_tiled(image,net,tile=256,padding=24):
    """Tile the SR operator only. The diffusion input remains the full photograph."""
    import numpy as np
    import torch
    from PIL import Image
    if not 32<=tile<=512 or not 8<=padding<=64 or image.width*image.height>3*2**20:
        raise ValueError("Bounded full-frame restoration canvas and tile required")
    source=np.asarray(image.convert("RGB"),dtype=np.float32)/255
    h,w=source.shape[:2];out=np.empty((h*4,w*4,3),dtype=np.uint8)
    param=next(net.parameters())
    with torch.inference_mode():
        for top in range(0,h,tile):
            for left in range(0,w,tile):
                right,bottom=min(w,left+tile),min(h,top+tile)
                x0,y0=max(0,left-padding),max(0,top-padding)
                x1,y1=min(w,right+padding),min(h,bottom+padding)
                patch=torch.from_numpy(source[y0:y1,x0:x1].copy()).permute(2,0,1)[None].to(param.device,param.dtype)
                restored=net(patch)
                if restored.shape!=(1,3,(y1-y0)*4,(x1-x0)*4) or not torch.isfinite(restored).all().item():
                    raise ValueError("Invalid restoration output; no resizing fallback")
                restored=restored[0,:,4*(top-y0):4*(bottom-y0),4*(left-x0):4*(right-x0)]
                pixels=restored.clamp(0,1).permute(1,2,0).float().cpu().numpy()
                out[top*4:bottom*4,left*4:right*4]=np.rint(pixels*255).astype(np.uint8)
    return Image.fromarray(out)
