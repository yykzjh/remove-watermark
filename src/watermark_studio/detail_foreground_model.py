"""Small paired pixel foreground veto pilot; a negative prediction is never a quality certificate."""
from __future__ import annotations

import numpy as np

MODEL_TYPE="detail_foreground_unet_v1"
TILE=256
THRESHOLD=.01
DILATION=4


def build_foreground_model():
    import torch
    from torch import nn
    from torch.nn import functional as F

    def block(before,after):
        return nn.Sequential(nn.Conv2d(before,after,3,padding=1),nn.GroupNorm(4,after),nn.SiLU(),
            nn.Conv2d(after,after,3,padding=1),nn.GroupNorm(4,after),nn.SiLU())
    class ForegroundUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.a=block(6,24);self.b=block(24,48);self.c=block(48,96);self.mid=block(96,128)
            self.u2=block(224,96);self.u1=block(144,48);self.u0=block(72,24)
            self.head=nn.Conv2d(24,1,1)
        def forward(self,values):
            a=self.a(values);b=self.b(F.avg_pool2d(a,2));c=self.c(F.avg_pool2d(b,2))
            mid=self.mid(F.avg_pool2d(c,2))
            up=lambda value,skip:F.interpolate(value,size=skip.shape[-2:],mode="bilinear",align_corners=False)
            c=self.u2(torch.cat([up(mid,c),c],dim=1))
            b=self.u1(torch.cat([up(c,b),b],dim=1))
            return self.head(self.u0(torch.cat([up(b,a),a],dim=1)))
    return ForegroundUNet()


def foreground_scores(model,source,reference,box,device,*,batch_size=16):
    """Native classifier tiles only; never rescales/crops a diffusion input."""
    import torch

    if source.size!=reference.size or not 1<=batch_size<=32:
        raise ValueError("Matching complete source/reference frames and bounded classifier batch required")
    l,t,r,b=box
    if not (0<=l<r<=source.width and 0<=t<b<=source.height):
        raise ValueError("Classifier body box escaped its original frame")
    arrays=[np.asarray(im.convert("RGB").crop(box)).copy() for im in (source,reference)]
    height,width=arrays[0].shape[:2]
    arrays=[np.pad(a,((0,max(0,TILE-height)),(0,max(0,TILE-width)),(0,0)),mode="edge") for a in arrays]
    h,w=arrays[0].shape[:2]
    def starts(size):
        return sorted({*range(0,max(1,size-TILE+1),TILE-64),size-TILE})
    positions=[(y,x) for y in starts(h) for x in starts(w)]
    scores=np.zeros((h,w),np.float32)
    with torch.inference_mode():
        for offset in range(0,len(positions),batch_size):
            group=positions[offset:offset+batch_size]
            batch=np.stack([np.concatenate([a[y:y+TILE,x:x+TILE] for a in arrays],axis=2)
                .transpose(2,0,1) for y,x in group])
            tensor=torch.from_numpy(batch).to(device=device,dtype=torch.float32).div(127.5).sub(1)
            predicted=model(tensor).float().sigmoid()
            if predicted.shape!=(len(group),1,TILE,TILE) or not torch.isfinite(predicted).all().item():
                raise ValueError("Foreground classifier returned invalid pixel scores")
            for (y,x),value in zip(group,predicted[:,0].cpu().numpy(),strict=True):
                np.maximum(scores[y:y+TILE,x:x+TILE],value,out=scores[y:y+TILE,x:x+TILE])
    full=np.ones((source.height,source.width),np.float32)
    full[t:b,l:r]=scores[:height,:width]
    return full
