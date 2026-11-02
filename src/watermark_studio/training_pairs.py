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


def render_camera_pair(clean,variant,rng,font_path,*,return_overlay=False):
    """Versioned camera-app overlays; never changes the legacy renderer."""
    if clean.mode!="RGB" or variant not in range(4) or min(clean.size)<128:
        raise ValueError("RGB target >=128px and explicit 0..3 variant required")
    width,height=clean.size
    layer=Image.new("RGBA",clean.size,(0,0,0,0));draw=ImageDraw.Draw(layer)
    def font(size):return ImageFont.truetype(str(font_path),max(7,round(size)),index=2)
    meta={"profile":"camera-v2","overlay_groups":0}
    if variant!=3:
        if variant in (0,1):
            cw=round(width*rng.uniform(.32,.56))
            rows=rng.choice([4,5,6]);ch=round(height*rng.uniform(.22,.37))
            # Most photographs use a lower-left card, with explicit alternatives.
            corner=rng.choices(["bottom_left","top_left","bottom_right","top_right"],[7,1,1,1])[0]
            x=rng.uniform(.012,.03)
            y=rng.uniform(.012,.03)
            l=round(width*x) if "left" in corner else width-cw-round(width*x)
            t=height-ch-round(height*y) if "bottom" in corner else round(height*y)
            r,b=l+cw,t+ch
            opacity=rng.choice([.30,.45,.60,.75,.85,.93])
            foreground=rng.choice([(226,226,226),(242,242,242),(215,220,225)])
            radius=max(3,round(cw*.025))
            draw.rounded_rectangle((l,t,r-1,b-1),radius,fill=(*foreground,round(255*opacity)))
            hh=max(20,round(ch*.23)) if variant==0 else 0
            footer=max(9,round(ch*.07)) if variant==0 and rng.random()<.65 else 0
            if hh:
                blue=rng.choice([(45,121,221),(32,104,209),(31,97,190)])
                draw.rectangle((l,t,r-1,t+hh-1),fill=(*blue,rng.randint(205,255)))
                cell=min(hh,round(cw*.14))
                draw.rectangle((l,t,l+cell-1,t+hh-1),fill=(250,250,250,255))
                pad=max(2,round(cell*.16));mid=l+cell//2
                # Generic green/blue engineering emblem, not a physical scene sign.
                draw.polygon([(mid,t+pad),(l+pad,t+hh-pad),(mid,t+hh-pad)],fill=(21,164,101,255))
                draw.polygon([(mid,t+pad),(mid,t+hh-pad),(l+cell-pad,t+hh-pad)],fill=(20,113,215,255))
                title=rng.choice(["高标准农田建设","建设项目施工记录","工程建设现场检查"])
                fs=min(hh*.66,(cw-cell-8)/len(title))
                draw.text((l+cell+4,t+max(0,(hh-fs)*.2)),title,font=font(fs),fill=(255,255,255,255))
            content=[
                rng.choice(["施工内容：田块整治","工程项目：高标准农田","施工内容：机耕道路"]),
                "拍摄时间：2025.07.21 14:35",
                rng.choice(["地点：乡村建设项目现场","地    点：田块施工现场"]),
                rng.choice(["备注：施工质量检查","施工工序：混凝土浇筑","备    注：现场踏勘"]),
                rng.choice(["施工单位：工程建设单位","项目名称：农田建设工程"]),
                rng.choice(["天气：多云","现场记录：施工中"])]
            gap=(ch-hh-footer-6)/rows;fs=min(gap*.67,(cw-10)/max(len(v) for v in content[:rows]))
            for index,text in enumerate(content[:rows]):
                draw.text((l+4,t+hh+2+index*gap),text,font=font(fs),fill=(10,10,10,255))
            if footer:
                draw.rectangle((l,b-footer,r-1,b-1),fill=(29,112,214,235))
                code="现场记录  "+str(rng.randrange(10000000,99999999))
                draw.text((l+3,b-footer),code,font=font(min(footer*.7,(cw-6)/len(code))),fill=(255,255,255,255))
            meta.update(panel=[l,t,r,b],header_height=hh,footer_height=footer,rows=rows,
                opacity=opacity,foreground=foreground,corner=corner)
        else:
            text=rng.choice(["现场拍摄 2025.07.21","2025年07月21日 14:35","工程记录  真实时间"])
            fs=height*rng.uniform(.025,.037)
            draw.text((round(width*.04),round(height*rng.choice([.04,.88]))),text,font=font(fs),
                fill=(255,255,255,235),stroke_width=1,stroke_fill=(35,35,35,140))
        size=max(8,round(height*rng.uniform(.023,.033)))
        brand=rng.choice([["今日水印","相机 真实时间","防伪码 "+str(rng.randrange(10**9,10**10))],
            ["今日水印","— 相机 —"],["工程记录","真实时间地点"]])
        x=max(2,width-round(size*7.6));y=max(2,height-round((size+2)*(len(brand)+.8)))
        for i,text in enumerate(brand):
            draw.text((x,y+i*(size+2)),text,font=font(size if i<2 else size*.65),
                fill=(255,255,255,240),stroke_width=1,stroke_fill=(30,30,30,140))
        meta["overlay_groups"]=2
    mask=layer.getchannel("A").point(lambda value:255 if value else 0)
    result=Image.alpha_composite(clean.convert("RGBA"),layer).convert("RGB")
    if np.any(np.any(np.asarray(result)!=np.asarray(clean),axis=2)&(np.asarray(mask)==0)):
        raise ValueError("Versioned synthetic edit escaped exact mask")
    values=(result,mask,{"kind":"identity" if variant==3 else "camera_app_v2",**meta})
    return (*values,layer) if return_overlay else values


def render_profile_pair(clean,variant,rng,font_path,*,profile="legacy",return_overlay=False):
    """Explicit renderer identity is required when replaying a derived dataset."""
    if profile not in {"legacy","camera-v2"}:
        raise ValueError("Unknown renderer profile")
    renderer=render_pair if profile=="legacy" else render_camera_pair
    return renderer(clean,variant,rng,font_path,return_overlay=return_overlay)
