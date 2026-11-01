"""Source-only certificates for gray conditioning, never output-quality approvals."""
from __future__ import annotations

import hashlib
import time

from .cross_review import _classify
from .source_vetting import source_carrier_chats


def digital_overlay(report):
    return (isinstance(report,dict)
            and report.get("digital_overlay") is True
            and report.get("physical_scene_text") is False
            and report.get("object_attachment") is False
            and report.get("surface_evidence") is False
            and report.get("uncertain") is False
            and isinstance(report.get("evidence"),str) and bool(report["evidence"].strip()))


def gray_source_cores(raw):
    """Keep raw detected bounds for initialization, including small camera marks.

    This is separate from composition feather cores: small overlays still use
    their fully opaque padded masks when preserving original output pixels.
    """
    from .detection import parse_detection
    result={}
    for region in parse_detection(raw):
        l,t,r,b=region.box
        inside=[c.box for c in parse_detection(raw,relative_padding=0)
                if l<=c.box[0]<c.box[2]<=r and t<=c.box[1]<c.box[3]<=b]
        if inside:
            result[region.box]=(min(c[0] for c in inside),min(c[1] for c in inside),
                                max(c[2] for c in inside),max(c[3] for c in inside))
    return result


def certify_gray_regions(photo, regions, cores, rims, editor, editor_processor,
                         verifier, verifier_processor, device, weights_relation):
    """Independently prove every remaining overlay before graying its raw core."""
    skip={tuple(r["mask"]) for r in rims or [] if r.get("approved") is True}
    targets=[r for r in regions if tuple(r.box) not in skip]
    fingerprint=hashlib.sha256(photo.convert("RGB").tobytes()).hexdigest() if targets else None
    audits=[]
    for region in targets:
        began=time.monotonic()
        core=cores.get(tuple(region.box))
        item={"mask":list(region.box),"kind":region.kind,"core":list(core) if core is not None else None,
              "approved":False,"version":"gray-source-digital-v1","weights_relation":weights_relation,
              "source_size":list(photo.size),"source_rgb_sha256":fingerprint}
        try:
            if weights_relation!="different_weights" or editor is verifier:
                raise ValueError("Distinct independent source weights required")
            if (core is None or not region.box[0]<=core[0]<core[2]<=region.box[2]
                    or not region.box[1]<=core[1]<core[3]<=region.box[3]
                    or (region.box[2]-region.box[0])*(region.box[3]-region.box[1])>.4):
                raise ValueError("Missing bounded raw source core")
            chats=source_carrier_chats(photo,region.box)
            chats[0][0]["content"][-1]["text"]+=(
                "\n下层照片有纹理不等于文字附着于下层物体。载体字段只针对待核实信息本身；"
                "不能因去水印任务或候选框而认定数字覆盖，需有可见叠加关系/布局证据。")
            item["editor"]=_classify(chats,editor,editor_processor,device)[0]
            if not digital_overlay(item["editor"]):
                raise ValueError("Editor did not establish unambiguous digital identity")
            item["independent"]=_classify(chats,verifier,verifier_processor,device)[0]
            item["approved"]=digital_overlay(item["independent"])
            if not item["approved"]:
                raise ValueError("Independent source verifier did not confirm digital identity")
            item["reason"]="two_source_digital_proofs"
        except Exception as error:  # noqa: BLE001 - source ambiguity blocks blind repainting
            item["reason"]=str(error)[:240]
            item["requires_review"]=True
        item["seconds"]=round(time.monotonic()-began,5)
        audits.append(item)
    return audits
