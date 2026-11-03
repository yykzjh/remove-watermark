"""Experimental bounded inquiry into a watermark-only dispute; never waives other vetoes."""
from __future__ import annotations


def eligible(verdict,context):
    explanation=verdict.get("model_explanation") or {}
    outside=verdict.get("outside_repair_change") or {}
    return (
        context.get("background_preserved") is True
        and context.get("source_carrier_ambiguous") is not True
        and context.get("source_scene")=="outdoor"
        and verdict.get("status")=="review"
        and verdict.get("watermark_removed") is False
        and verdict.get("natural") is True
        and verdict.get("content_preserved") is True
        and verdict.get("uncertain") is False
        and not verdict.get("guard_codes")
        and outside.get("tiles")==0
        and verdict.get("residual_regions")==[]
        and verdict.get("evidence_conflict") is True
        and explanation.get("status")=="ok"
        and explanation.get("assessment")=="no_clear_defect"
        and explanation.get("findings")==[]
    )

def carrier_context_box(box,size):
    """A text line needs object context above/below it, not a narrow strip."""
    import math
    width,height=size
    left,top,right,bottom=box
    if not (0<=left<right<=width and 0<=top<bottom<=height):
        raise ValueError("Native text geometry must stay inside the photograph")
    side=max(96,math.ceil(max(right-left,bottom-top)*2.6))
    cw,ch=min(width,side),min(height,side)
    x=max(0,min(width-cw,round((left+right-cw)/2)))
    y=max(0,min(height-ch,round((top+bottom-ch)/2)))
    return (x,y,x+cw,y+ch)


def reconsider(before,after,verdict,editor,editor_processor,validator,validator_processor,device,
               context,weights_relation,text_preserver,resolution=768,max_tokens=128):
    import time
    from copy import deepcopy

    from PIL import Image

    from .config import Region
    from .evidence import refresh_evidence
    from .images import pixel_box
    from .quality import output_detail_views, output_review_prompt, preview

    log={"method":"preserved_physical_text_native_scans_v2","attempted":False,"status":"skipped"}
    verdict["physical_text_review"]=log
    if not eligible(verdict,context):
        return
    if (weights_relation!="different_weights" or editor is None or editor is validator
            or text_preserver is None):
        log["reason"]="independent_weights_and_pixel_witness_required"
        return
    start=time.monotonic();log["attempted"]=True
    try:
        witness=text_preserver.inspect(before,after);log["text_preservation"]=witness
        if witness.get("status")!="unchanged":
            log.update(status="unresolved",reason="visible_text_not_proven_unchanged")
            return
        content=[{"type":"text","text":"完整结果照片："},
                 {"type":"image","image":preview(after,max(1024,resolution))}]
        # Supply bounded carrier context, not isolated glyphs or previous conclusions.
        boxes=sorted((r["box"] for r in witness["text_boxes"]),
            key=lambda b:(b[2]-b[0])*(b[3]-b[1]),reverse=True)[:4]
        for box in boxes:
            crop=carrier_context_box(pixel_box(box,after.size),after.size)
            content.extend([{"type":"text","text":f"结果文字周围的实体上下文，整图文字范围{box}，裁切边缘不是水印。"},
                {"type":"image","image":preview(after.crop(crop),512)}])
        regions=[Region.model_validate(r) for r in verdict["source_regions"]]
        aligned=after if after.size==before.size else after.resize(before.size,Image.Resampling.LANCZOS)
        details,detail_boxes=output_detail_views(before,aligned,4,regions)
        for detail,box in zip(details,detail_boxes):
            content.extend([{"type":"text","text":f"结果原覆盖区域及细节，整图范围{box}："},
                            {"type":"image","image":detail}])
        from .detection import locate_watermarks_batch
        source_views=[after]
        source_scopes=[[0.,0.,1.,1.]]
        if max(after.size)>1400:
            for box in ((0,0,.56,.56),(.44,0,1,.56),(0,.44,.56,1),(.44,.44,1,1)):
                source_scopes.append(list(box))
                source_views.append(after.crop(pixel_box(box,after.size)))
        for box in boxes:
            crop=carrier_context_box(pixel_box(box,after.size),after.size)
            source_scopes.append([crop[0]/after.width,crop[1]/after.height,
                                  crop[2]/after.width,crop[3]/after.height])
            source_views.append(after.crop(crop))
        scanned=locate_watermarks_batch(source_views,editor,editor_processor,device,
            max(1280,resolution),1)
        if len(scanned)!=len(source_views):
            raise ValueError("Independent native scan count mismatch")
        log["independent_editor_native_scans"]=[]
        for scope,scan in zip(source_scopes,scanned):
            if isinstance(scan,Exception):
                raise scan
            residual=[r.model_dump() for r in scan[0]]
            log["independent_editor_native_scans"].append({"scope":scope,
                "residuals":residual,"raw":scan[1][:1600]})
        if any(r["residuals"] for r in log["independent_editor_native_scans"]):
            log.update(status="unresolved",reason="editor_native_residual_detected")
            return
        from .grounded_review import inspect_grounded
        # Independent judge receives the original image evidence, not editor prose.
        independent=[dict(item) for item in content]
        independent.append({"type":"text","text":output_review_prompt()+
            "\n没有缺陷且能确定时area必须是none，真实实物文字的位置不是缺陷位置。"})
        values,_=inspect_grounded([[{"role":"user","content":independent}]],
            ("watermark_removed","natural","uncertain"),validator,validator_processor,device,384)
        check=values[0];log["independent_validator_output"]=check
        if not (check.get("watermark_removed") is True and check.get("natural") is True
                and check.get("uncertain") is False and check.get("area")=="none"):
            log.update(status="unresolved",reason="independent_grounded_output_not_confirmed")
            return
        from .detection import locate_watermarks_batch
        scan=locate_watermarks_batch([after],validator,validator_processor,device,max(1024,resolution),1)[0]
        if isinstance(scan,Exception):
            raise scan
        residual=[r.model_dump() for r in scan[0]]
        log["fresh_unwhitelisted_residuals"]=residual
        if residual:
            log.update(status="unresolved",reason="fresh_residuals_detected")
            return
        # All original content/naturalness/uncertainty and deterministic vetoes stay.
        if not eligible(verdict,context):
            raise RuntimeError("Initial verdict changed during the inquiry")
        updated=deepcopy(verdict)
        updated["pre_physical_text_review"]={k:deepcopy(verdict.get(k)) for k in
            ("status","watermark_removed","natural","content_preserved","uncertain","reason",
             "judgment_basis","model_explanation","evidence_conflict")}
        updated.update(status="passed",watermark_removed=True,evidence_conflict=False,
            reason="结果可见文字的原图像素一致性证据、双独立模型的整图与原生细节数字覆盖层检查、无豁免残留扫描均通过；原自然度和内容检查保留。")
        updated["model_explanation"]={"status":"adjudicated","assessment":"no_clear_defect",
            "summary":updated["reason"],"findings":[]}
        log.update(status="corrected",reason="unchanged_printing_and_independent_native_scans")
        updated["physical_text_review"]=log
        refresh_evidence(updated)
        verdict.update(updated)
    except Exception as error:  # noqa: BLE001 - inquiry errors never permit approval
        log.update(status="error",reason="inquiry_failed",error=f"{type(error).__name__}: {error}"[:400])
    finally:
        log["seconds"]=round(time.monotonic()-start,6)
        verdict["seconds"]=round(verdict.get("seconds",0)+log["seconds"],3)
