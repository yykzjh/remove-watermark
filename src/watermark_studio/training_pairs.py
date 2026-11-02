"""Exact-pixel synthetic editing pairs from independently screened clean photo crops."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def clean_crop(size,regions,aspect=4/3,padding=.04,min_side=384):
    """Largest aspect-constrained rectangle outside every expanded source hypothesis."""
    width,height=size;blocked=[]
    if not .5<=aspect<=2 or not .025<=padding<=.1 or not 1<=min_side:
        raise ValueError("Bounded clean-crop geometry required")
    for region in regions:
        x0,y0,x1,y1=region.box
        blocked.append((max(0,math.floor((x0-padding)*width)),max(0,math.floor((y0-padding)*height)),
            min(width,math.ceil((x1+padding)*width)),min(height,math.ceil((y1+padding)*height))))
    xs=sorted({0,width,*[v for b in blocked for v in (b[0],b[2])]})
    ys=sorted({0,height,*[v for b in blocked for v in (b[1],b[3])]})
    occupied=np.zeros((len(ys)-1,len(xs)-1),bool)
    for j in range(len(ys)-1):
        for i in range(len(xs)-1):
            occupied[j,i]=any(max(xs[i],l)<min(xs[i+1],r) and max(ys[j],t)<min(ys[j+1],b)
                for l,t,r,b in blocked)
    best=None;area=0
    for left in range(len(xs)-1):
        for right in range(left+1,len(xs)):
            free=~occupied[:,left:right].any(axis=1)
            edges=np.diff(np.r_[False,free,False].astype(np.int8))
            for top,bottom in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
                aw,ah=xs[right]-xs[left],ys[bottom]-ys[top]
                cw,ch=min(aw,math.floor(ah*aspect)),min(ah,math.floor(aw/aspect))
                if min(cw,ch)<min_side or cw*ch<=area:continue
                l=xs[left]+(aw-cw)//2;t=ys[top]+(ah-ch)//2
                best=(l,t,l+cw,t+ch);area=cw*ch
    return best if area>=.12*width*height else None


def render_pair(clean,variant,rng,font_path,*,return_overlay=False):
    """Draw only a known digital overlay; the target clean photograph is immutable."""
    if clean.mode!="RGB" or variant not in range(4):
        raise ValueError("RGB clean target and explicit 0..3 variant required")
    width,height=clean.size;layer=Image.new("RGBA",clean.size,(0,0,0,0))
    draw=ImageDraw.Draw(layer);render_meta={}
    def font(size):return ImageFont.truetype(str(font_path),max(7,int(size)),index=2)
    stamp=max(8,round(height*.031))
    if variant==3:
        values=(clean.copy(),Image.new("L",clean.size,0),{"kind":"identity","overlay_groups":0})
        return (*values,layer) if return_overlay else values
    if variant in (0,1):
        cw=round(width*rng.uniform(.36,.55));ch=round(height*rng.uniform(.24,.36))
        corners=[(.018,.98-ch/height),(.98-cw/width,.018),(.035,.035),(.97-cw/width,.97-ch/height)]
        x,y=rng.choice(corners);l,t=round(x*width),round(y*height);r,b=l+cw,t+ch
        opacity=rng.choice([.25,.45,.70,.85,.93])
        fg=rng.choice([(224,224,224),(241,241,241),(207,215,225)])
        extra=round(width*rng.uniform(0,.035)) if variant==0 else 0
        br=min(width-3,r+extra)
        draw.rounded_rectangle((l,t,br,b),radius=max(3,round(cw*.025)),fill=(*fg,round(opacity*255)))
        top=t
        if variant==0:
            hh=max(16,round(ch*.25))
            blue=rng.choice([(32,112,215),(28,83,173),(15,143,178)])
            draw.rectangle((l,t,r,t+hh),fill=(*blue,rng.randint(220,255)))
            title="高标准农田建设"
            draw.text((l+5,t+2),title,font=font(min(hh-4,cw/len(title)*.9)),fill=(255,255,255,255))
            top=t+hh
        rows=["施工内容：现场检查","拍摄时间：2025.06.04","地点：建设项目现场","备注：施工质量验收"]
        fh=max(7,min((b-top-6)/len(rows)*.68,cw/17))
        for i,text in enumerate(rows):
            draw.text((l+4,top+3+i*(b-top-4)/len(rows)),text,font=font(fh),fill=(12,12,12,255))
        kind=("blue_gray_wide" if extra else "blue_gray") if variant==0 else "gray_information"
        render_meta={"panel":[l,t,br,b],"header_right":r,"opacity":opacity,"foreground":fg}
    else:
        kind="floating_stamp"
        text="拍摄于 2025年06月04日"
        draw.text((max(3,width-round(stamp*14)),max(3,round(height*.06))),text,
            font=font(stamp),fill=(255,255,255,210),stroke_width=1,stroke_fill=(35,35,35,150))
    # Complete app cluster, including secondary lines and a fine verification code.
    size=max(8,round(height*.026));x=max(2,width-round(size*7.6));y=max(2,height-round(size*4.3))
    for i,text in enumerate(["今日水印","相机 真实时间","防伪码 "+str(rng.randrange(10**7,10**8))]):
        draw.text((x,y+i*(size+2)),text,font=font(size if i<2 else size*.72),
            fill=(255,255,255,240),stroke_width=1,stroke_fill=(30,30,30,140))
    mask=layer.getchannel("A").point(lambda value:255 if value else 0)
    result=Image.alpha_composite(clean.convert("RGBA"),layer).convert("RGB")
    if np.any(np.any(np.asarray(result)!=np.asarray(clean),axis=2)&(np.asarray(mask)==0)):
        raise ValueError("Synthetic edits escaped their exact overlay mask")
    values=(result,mask,{"kind":kind,"overlay_groups":2,"render":render_meta})
    return (*values,layer) if return_overlay else values
