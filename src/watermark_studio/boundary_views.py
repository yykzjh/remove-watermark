"""Source-mask aligned ribbons for optional, read-only naturalness inspection."""
from __future__ import annotations

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .images import pixel_box


def boundary_ribbons(after, regions):
    """Show normal/low-frequency boundary context, never infer a defect or mask."""
    panels=[r for r in regions if (r.box[2]-r.box[0])*(r.box[3]-r.box[1])>=.06]
    if not panels:
        return None, []
    region=max(panels,key=lambda r:(r.box[2]-r.box[0])*(r.box[3]-r.box[1]))
    l,t,r,b=pixel_box(region.box,after.size)
    depth=max(24,min(192,round(min(r-l,b-t)*.20)))
    candidates=[
        ("top",(max(0,l-depth),max(0,t-depth),min(after.width,r+depth),min(after.height,t+depth)),t>0,False),
        ("bottom",(max(0,l-depth),max(0,b-depth),min(after.width,r+depth),min(after.height,b+depth)),b<after.height,False),
        ("left",(max(0,l-depth),max(0,t-depth),min(after.width,l+depth),min(after.height,b+depth)),l>0,True),
        ("right",(max(0,r-depth),max(0,t-depth),min(after.width,r+depth),min(after.height,b+depth)),r<after.width,True),
    ]
    patches=[]
    for side,box,available,rotate in candidates:
        if not available or box[0]>=box[2] or box[1]>=box[3]:
            continue
        patch=after.crop(box).convert("RGB")
        if rotate:
            patch=patch.transpose(Image.Transpose.ROTATE_90)
        patches.append((side,box,patch))
    if not patches:
        return None, []
    canvas=Image.new("RGB",(640,96*len(patches)),(240,240,240))
    boxes=[]
    for i,(side,box,patch) in enumerate(patches):
        normal=ImageOps.contain(patch,(320,96),Image.Resampling.LANCZOS)
        diagnostic=ImageEnhance.Contrast(normal.convert("L").filter(
            ImageFilter.GaussianBlur(2))).enhance(2).convert("RGB")
        y=i*96+(96-normal.height)//2
        canvas.paste(normal,((320-normal.width)//2,y))
        canvas.paste(diagnostic,(320+(320-normal.width)//2,y))
        boxes.append({"side":side,"box":[v/s for v,s in zip(
            box,(after.width,after.height,after.width,after.height))]})
    return canvas,boxes
