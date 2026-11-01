"""Full-frame gray initialization of independently proven digital card cores."""
import hashlib

from .images import pixel_box
from .source_panel_rims import effective_rim_scope


def gray_whole_reference(source, regions, core_mapping, proofs):
    """Prepare model input only. Original pixels remain the sole QA reference."""
    cores=[{"mask":list(mask),"core":list(core)} for mask,core in core_mapping.items()]
    scoped, mapped, invalid=effective_rim_scope(source,regions,cores,proofs,header_core=True)
    if invalid:
        raise ValueError("Gray conditioning requires intact source provenance")
    approved=[item for item in proofs or [] if item.get("approved") is True]
    if approved:
        fingerprint=hashlib.sha256(source.convert("RGB").tobytes()).hexdigest()
        if any(item.get("source_rgb_sha256")!=fingerprint for item in approved):
            raise ValueError("Gray conditioning requires the full original source RGB proof")
    allowed={tuple(item["guard_region"]["box"]) for item in approved}
    kinds={tuple(region.box):region.kind for region in scoped}
    result=source.copy()
    applied=[]
    for item in mapped:
        mask,core=tuple(item["mask"]),item["core"]
        if mask not in allowed or kinds.get(mask)!="panel" or (core[2]-core[0])*(core[3]-core[1])<.03:
            continue
        box=pixel_box(core,source.size)
        result.paste((128,128,128),box)
        applied.append(list(box))
    if not applied:
        return source,False
    result.info["gray_conditioning"]={"method":"source_proven_gray_full_frame_v1",
        "rectangles":applied,"quality_permission":False}
    return result,True
