"""Offline crop-coordinate adapter for the existing source-flow denoising probe."""
from watermark_studio.config import Region
from watermark_studio.images import pixel_box


def prepare_roi_context(edits,margin):
    if type(margin) is not int or not 0<=margin<=16:
        raise ValueError("A bounded integer latent-cell margin is required")
    audits=[]
    for edit in edits:
        state=edit["state"]
        if state["whole_generation"]:
            continue
        width,height=state["source"].size
        left,top,right,bottom=edit["crop"]
        if not 0<=left<right<=width or not 0<=top<bottom<=height:
            raise ValueError("Crop must be inside its own original photograph")
        if edit["image"].size!=(right-left,bottom-top):
            raise ValueError("ROI reference geometry changed")
        native=[(region,pixel_box(region.box,(width,height))) for region in state["regions"]]
        for member in edit.get("_roi_members",[edit]):
            if member["state"] is not state:
                raise ValueError("Cannot broadcast another source context")
            x0,y0,x1,y1=member["box"]
            if not any(a<=x0<x1<=c and b<=y0<y1<=d for _,(a,b,c,d) in native):
                raise ValueError("Do not lock an unproven repair footprint back to the source")
        mapped=[]
        for region,(a,b,c,d) in native:
            x0,y0,x1,y1=max(left,a),max(top,b),min(right,c),min(bottom,d)
            if x0<x1 and y0<y1:
                mapped.append(Region(kind=region.kind,box=(
                    (x0-left)/(right-left),(y0-top)/(bottom-top),
                    (x1-left)/(right-left),(y1-top)/(bottom-top))).model_dump())
        if not mapped:
            raise ValueError("ROI source masks cannot be empty")
        edit["_denoising_context_regions"]=mapped
        edit["_denoising_context_margin"]=margin
        audits.append({"file":state["task"]["relative"],"index":edit["index"],
            "scope":"original_ROI_crop","crop":list(edit["crop"]),
            "reference_size":list(edit["image"].size),"context_margin_cells":margin,
            "mapped_regions":mapped,"extra_encodes":0,"extra_generation_candidates":0})
    return audits
