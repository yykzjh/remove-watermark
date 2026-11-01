"""Full-frame gray initialization of independently proven digital overlay cores."""
import hashlib

from .images import pixel_box
from .source_panel_rims import effective_rim_scope


def gray_whole_reference(source, regions, core_mapping, proofs, certificates=()):
    """Prepare model input only. Original pixels remain the sole QA reference."""
    cores=[{"mask":list(mask),"core":list(core)} for mask,core in core_mapping.items()]
    scoped, mapped, invalid=effective_rim_scope(source,regions,cores,proofs,header_core=True)
    if invalid:
        raise ValueError("Gray conditioning requires intact source provenance")
    approved=[item for item in proofs or [] if item.get("approved") is True]
    fingerprint=(hashlib.sha256(source.convert("RGB").tobytes()).hexdigest()
                 if approved or any(c.get("approved") is True for c in certificates or []) else None)
    if approved and any(item.get("source_rgb_sha256")!=fingerprint for item in approved):
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
    from .source_digital_proof import digital_overlay
    original_rims={tuple(item["mask"]) for item in approved}
    original_regions={tuple(region.box):region for region in regions}
    for certificate in certificates or []:
        if certificate.get("approved") is not True:
            continue
        mask=tuple(certificate.get("mask",[]))
        if mask in original_rims:
            continue
        if (certificate.get("requires_review") or certificate.get("weights_relation")!="different_weights"
                or not digital_overlay(certificate.get("editor")) or not digital_overlay(certificate.get("independent"))
                or certificate.get("source_size")!=list(source.size)
                or certificate.get("source_rgb_sha256")!=fingerprint
                or mask not in original_regions or original_regions[mask].kind!=certificate.get("kind")
                or tuple(certificate.get("core") or ())!=tuple(core_mapping.get(mask) or ())):
            raise ValueError("Invalid remaining-overlay source certificate")
        core=core_mapping.get(mask)
        if (core is None or not mask[0]<=core[0]<core[2]<=mask[2]
                or not mask[1]<=core[1]<core[3]<=mask[3] or (mask[2]-mask[0])*(mask[3]-mask[1])>.4):
            raise ValueError("Invalid remaining-overlay source core")
        box=pixel_box(core,source.size)
        result.paste((128,128,128),box)
        applied.append(list(box))
    if not applied:
        return source,False
    result.info["gray_conditioning"]={"method":"source_proven_gray_full_frame_v1",
        "rectangles":applied,"quality_permission":False}
    return result,True
