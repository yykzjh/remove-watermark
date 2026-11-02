"""Exact-target, source-isolated synthetic defect pairs for an offline fast guard."""
from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

CLASSES=("clean","shadow_or_tint","repair_blur","residual_overlay","changed_visible_content")


def _box(clean,mask,metadata,rng):
    width,height=clean.size
    panel=metadata.get("render",{}).get("panel") or metadata.get("panel")
    if panel:return tuple(panel)
    locations=np.argwhere(np.asarray(mask)>0)
    if locations.size:
        y,x=locations[rng.randrange(len(locations))]
    else:x,y=rng.uniform(.2,.8)*width,rng.uniform(.2,.8)*height
    bw,bh=round(width*rng.uniform(.22,.42)),round(height*rng.uniform(.16,.32))
    l=max(0,min(width-bw,round(x-bw/2)));t=max(0,min(height-bh,round(y-bh/2)))
    return l,t,l+bw,t+bh


def synthetic_case(source,clean,mask,metadata,label,seed,*,preserved_marks=False,shadow_profile="legacy"):
    """Only injected defects are positive; exact clean target is never model-generated."""
    if source.size!=clean.size or mask.size!=clean.size or label not in range(5):
        raise ValueError("Aligned exact clean/source/mask and explicit class required")
    if shadow_profile not in {"legacy","mixed-v1"}:raise ValueError("Explicit supported shadow profile required")
    rng=random.Random(seed);source=source.convert("RGB");clean=clean.convert("RGB")
    box=_box(clean,mask,metadata,rng);l,t,r,b=box
    marking=None
    if preserved_marks and rng.random()<.6:
        occupied=np.asarray(mask)>0
        for _ in range(128):
            mw,mh=rng.randint(28,64),rng.randint(16,28)
            x=rng.randint(max(0,l-(r-l)//3),min(clean.width-mw,max(0,r+(r-l)//3-mw)))
            y=rng.randint(max(0,t-(b-t)//3),min(clean.height-mh,max(0,b+(b-t)//3-mh)))
            if occupied[y:y+mh,x:x+mw].any():continue
            stamp=Image.new("RGB",(mw,mh),rng.choice(["#b4b6b3","#e6e6db","#cbcecc"]))
            ImageDraw.Draw(stamp).text((2,1),rng.choice(["WE","TX130","120cm","B-08"]),
                font=ImageFont.load_default(size=rng.randint(12,18)),fill=rng.choice(["#a52a2a","#182e42","#161616"]))
            source=source.copy();clean=clean.copy()
            source.paste(stamp,(x,y));clean.paste(stamp,(x,y))
            marking={"box":[x,y,x+mw,y+mh],"same_pixels_in_source_and_clean":True};break
    output=clean.copy();severity={}
    if label==1:
        # Include subtle 8-bit shifts and bright/color residuals, not only black.
        alpha=rng.uniform(.045,.24)
        color=rng.choice([(0,0,0),(255,255,255),(32,70,120),(170,170,185)])
        layer=Image.new("RGB",(r-l,b-t),color)
        output.paste(Image.blend(clean.crop(box),layer,alpha),box)
        severity={"alpha":alpha,"foreground":color}
        if shadow_profile=="mixed-v1" and rng.random()<.5:
            delta=rng.uniform(8.,48.)*rng.choice([-1,1])
            offset=np.array([delta+rng.uniform(-.15,.15)*abs(delta) for _ in range(3)])
            pixels=np.asarray(clean.crop(box),dtype=np.float32)
            output.paste(Image.fromarray(np.clip(np.rint(pixels+offset),0,255).astype(np.uint8)),box)
            severity={"additive_rgb_offset":offset.tolist(),"profile":"mixed-v1"}
    elif label==2:
        radius=rng.uniform(1.1,5.0)
        output.paste(clean.crop(box).filter(ImageFilter.GaussianBlur(radius)),box)
        severity={"blur_radius":radius}
    elif label==3:
        alpha=rng.uniform(.18,.92)
        output.paste(Image.blend(clean.crop(box),source.crop(box),alpha),box)
        severity={"retained_source_fraction":alpha}
    elif label==4:
        # Visible content changes only OUTSIDE the exact original overlay;
        # hidden ground truth must not become an impossible semantic label.
        occupied=np.asarray(mask)>0;placed=False
        for _ in range(128):
            x=rng.randrange(clean.width);y=rng.randrange(clean.height)
            rw,rh=rng.randint(8,30),rng.randint(8,30)
            region=(x,y,min(clean.width,x+rw),min(clean.height,y+rh))
            if occupied[region[1]:region[3],region[0]:region[2]].any():continue
            # Keep the added detail in the guard's context, near the candidate.
            if not (max(0,l-round((r-l)*.35))<=x and region[2]<=min(clean.width,r+round((r-l)*.35)) and
                    max(0,t-round((b-t)*.35))<=y and region[3]<=min(clean.height,b+round((b-t)*.35))):continue
            draw=ImageDraw.Draw(output)
            draw.ellipse(region,fill=tuple(rng.randint(0,255) for _ in range(3)))
            severity={"changed_visible_box":region};placed=True;break
        if not placed:raise ValueError("No known-visible context for a content-change control")
    difference=np.abs(np.asarray(output,dtype=np.int16)-np.asarray(clean,dtype=np.int16))
    if label and not np.any(difference>2):
        # E.g. blurring a featureless clean crop is NOT a defect example.
        alternate=synthetic_case(source,clean,mask,metadata,1,seed+7919,shadow_profile=shadow_profile)
        alternate[3]["preserved_marking"]=marking
        alternate[3]["fallback_from_label"]=label
        return alternate
    if label==0 and rng.random()<.5:
        stream=io.BytesIO();output.save(stream,format="JPEG",quality=rng.randint(95,100))
        stream.seek(0)
        with Image.open(stream) as image:output=image.convert("RGB")
        severity={"clean_jpeg":True}
    return source,output,box,{"label":label,"name":CLASSES[label],"severity":severity,
        "exact_clean_target":True,"diffusion_generations":0,"preserved_marking":marking}


def boundary_guard_input(before,after,box,resolution=256):
    """Four aligned inward-normal boundary strips, source/result paired; no approvals."""
    if before.size!=after.size or resolution not in (256,384):
        raise ValueError("Aligned frames and bounded strip resolution required")
    l,t,r,b=map(float,box);width,height=before.size
    if not np.isfinite([l,t,r,b]).all() or not 0<=l<r<=width or not 0<=t<b<=height:
        raise ValueError("Bounded finite pixel box required")
    strip=resolution//4;result=np.zeros((7,resolution,resolution),np.float32)
    result[6]=-1  # unavailable image-border side, distinct from a measured outside band
    mean=np.array([.485,.456,.406],np.float32);std=np.array([.229,.224,.225],np.float32)
    images=[after.convert("RGB"),before.convert("RGB")]
    valid_image=Image.new("L",before.size,255)
    for index,side in enumerate(("left","right","top","bottom")):
        vertical=index<2;coordinate=(l,r,t,b)[index];extent=width if vertical else height
        if coordinate<=1 or coordinate>=extent-1:continue
        half=max(8.,.12*((r-l) if vertical else (b-t)),.02*max(width,height))
        if side=="left":transform=(2*half/strip,0,l-half,0,.7*(b-t)/resolution,t+.15*(b-t))
        elif side=="right":transform=(-2*half/strip,0,r+half,0,.7*(b-t)/resolution,t+.15*(b-t))
        elif side=="top":transform=(0,.7*(r-l)/resolution,l+.15*(r-l),2*half/strip,0,t-half)
        else:transform=(0,.7*(r-l)/resolution,l+.15*(r-l),-2*half/strip,0,b+half)
        valid=np.asarray(valid_image.transform((strip,resolution),Image.Transform.AFFINE,transform,
            Image.Resampling.NEAREST,fillcolor=0),dtype=np.float32)/255.
        start=index*strip;end=start+strip
        for image_index,image in enumerate(images):
            values=np.asarray(image.transform((strip,resolution),Image.Transform.AFFINE,transform,
                Image.Resampling.BILINEAR,fillcolor=(124,116,104)),dtype=np.float32)/255.
            value=((values-mean)/std).transpose(2,0,1)
            result[image_index*3:image_index*3+3,:,start:end]=value*valid[None]
        inside=np.broadcast_to(np.arange(strip)[None]>=strip//2,(resolution,strip)).astype(np.float32)
        result[6,:,start:end]=np.where(valid>0,inside,-1)
    if not np.isfinite(result).all():raise ValueError("Nonfinite boundary guard input")
    return result


def guard_input(before,after,box,resolution=256,*,profile="context"):
    """Shared versioned native preprocessing; no generation or user-selected mask."""
    if profile=="boundary-strips-v1":return boundary_guard_input(before,after,box,resolution)
    if profile!="context":raise ValueError("Unknown guard input profile")
    if before.size!=after.size or not 128<=resolution<=384:
        raise ValueError("Common-size bounded guard input required")
    l,t,r,b=map(float,box);width,height=before.size
    if not 0<=l<r<=width or not 0<=t<b<=height:raise ValueError("Bounded pixel box required")
    margin=.45
    pad_x=max((r-l)*margin,width*.04);pad_y=max((b-t)*margin,height*.04)
    crop=(max(0,int(l-pad_x)),max(0,int(t-pad_y)),min(width,int(r+pad_x+.999)),
        min(height,int(b+pad_y+.999)))
    mask=Image.new("L",before.size,0);ImageDraw.Draw(mask).rectangle((l,t,r-1,b-1),fill=255)
    arrays=[]
    mean=np.array([.485,.456,.406],np.float32);std=np.array([.229,.224,.225],np.float32)
    for image in (after,before):
        pixels=np.asarray(image.crop(crop).convert("RGB").resize((resolution,resolution),Image.Resampling.BILINEAR),
            dtype=np.float32)/255.
        arrays.append(((pixels-mean)/std).transpose(2,0,1))
    support=np.asarray(mask.crop(crop).resize((resolution,resolution),Image.Resampling.NEAREST),dtype=np.float32)[None]/255.
    value=np.concatenate([*arrays,support],axis=0)
    if not np.isfinite(value).all():raise ValueError("Nonfinite guard input")
    return value.copy()


def build_guard(pretrained=False):
    """Pretraining is an explicit training-only option; inference never downloads."""
    import torch
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18
    model=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None,progress=False)
    original=model.conv1
    model.conv1=nn.Conv2d(7,original.out_channels,original.kernel_size,original.stride,original.padding,bias=False)
    with torch.no_grad():
        model.conv1.weight.zero_();model.conv1.weight[:,:3].copy_(original.weight)
    model.fc=nn.Linear(model.fc.in_features,len(CLASSES))
    return model

def physical_shadow_background(clean,box,seed):
    """Known pre-existing scene shade, applied BEFORE the original digital overlay."""
    rng=random.Random(seed);l,t,r,b=box;w,h=clean.size
    # The shade continues across the candidate, giving genuine scene context;
    # it is not a post-repair rectangle with exactly the watermark boundary.
    x0=max(0,l-rng.uniform(.1,.45)*(r-l));y0=max(0,t-rng.uniform(.1,.45)*(b-t))
    x1=min(w,r+rng.uniform(.1,.45)*(r-l));y1=min(h,b+rng.uniform(.1,.45)*(b-t))
    alpha=Image.new("L",clean.size,0)
    points=[(x0,y0),(x1,min(h,y0+rng.uniform(-.1,.1)*h)),(x1,y1),(x0,max(0,y1+rng.uniform(-.1,.1)*h))]
    ImageDraw.Draw(alpha).polygon(points,fill=round(rng.uniform(.07,.28)*255))
    alpha=alpha.filter(ImageFilter.GaussianBlur(rng.uniform(1.,6.)))
    shade=Image.new("RGB",clean.size,rng.choice([(0,0,0),(12,20,35),(35,30,20)]))
    return Image.composite(shade,clean.convert("RGB"),alpha)


def guard_decision(scores,defect_types=CLASSES[1:],threshold=.995):
    """Select only validated heads without renormalizing or ever granting a pass."""
    values=np.asarray(scores,dtype=float)
    if (values.shape!=(len(CLASSES),) or not np.isfinite(values).all() or
            not ((values>=0)&(values<=1)).all() or abs(values.sum()-1)>.002 or threshold!=.995):
        raise ValueError("Finite complete softmax and precommitted experimental threshold required")
    if (not defect_types or len(set(defect_types))!=len(defect_types) or
            any(name not in CLASSES[1:] for name in defect_types)):
        raise ValueError("Explicit unique defect heads required")
    indices=[CLASSES.index(name) for name in defect_types]
    kind=max(indices,key=lambda index:values[index]);score=float(values[kind])
    return {"type":CLASSES[kind],"score":score,"scores":[float(v) for v in values],
        "would_veto":score>=threshold,"quality_permission":False}
