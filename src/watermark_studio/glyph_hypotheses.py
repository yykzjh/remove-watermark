"""Diagnostic-only native glyph geometry for coarse physical-text hypotheses."""
from __future__ import annotations


def overlapping_physical_candidates(before,after,proposals,verdict,detector,feather=12):
    """No verdict waiver: all output glyphs must be unchanged outside every edit."""
    import hashlib

    from .cross_review import _overlap, candidate_boxes, preserved_patch
    from .evidence import valid_box

    result={"method":"overlap_native_glyph_diagnostic_v1","findings":[],"proposals":[],
        "quality_permission":False,"source_size":list(before.size),
        "source_rgb_sha256":hashlib.sha256(before.convert("RGB").tobytes()).hexdigest(),
        "output_rgb_sha256":hashlib.sha256(after.convert("RGB").tobytes()).hexdigest()}
    if before.size!=after.size or not proposals or len(proposals)>2:
        return result|{"reason":"bounded_same_resolution_proposals_required"}
    raw=[valid_box(p.get("box")) for p in proposals]
    if any(p.get("issue")!="physical_text" or box is None or
            (box[2]-box[0])*(box[3]-box[1])>.15 or max(box[2]-box[0],box[3]-box[1])>.65
            for p,box in zip(proposals,raw,strict=True)):
        return result|{"reason":"explicit_bounded_physical_hypotheses_only"}
    # Only a coarsely placed physical-text candidate is eligible for diagnosis;
    # a claimed residual inside the edited area must never move to another label.
    if not any(candidate_boxes(verdict|{"residual_regions":[{"box":box}]},before.size,feather)[1]
            =="overlaps_original_overlay_or_feather" for box in raw):
        return result|{"reason":"no_coarse_edit_overlap"}
    findings=[]
    for proposal,box in zip(proposals,raw,strict=True):
        observed=detector.inspect(after,[box])
        trace={"hypothesis":proposal,"output_glyphs":observed};result["proposals"].append(trace)
        if observed.get("status")!="text_present":
            return result|{"reason":"every_hypothesis_requires_output_native_glyphs"}
        glyphs=[g for region in observed.get("regions",[]) for g in region.get("textlike_boxes",[]) if g]
        if not glyphs or len(glyphs)>2:
            return result|{"reason":"missing_or_ambiguous_native_glyphs"}
        for glyph in glyphs:
            glyph=valid_box(glyph)
            if glyph is None or not _overlap(glyph,box):
                return result|{"reason":"glyph_does_not_anchor_original_hypothesis"}
            padx,pady=4/before.width,4/before.height
            if (glyph[0]<box[0]-padx or glyph[1]<box[1]-pady or
                    glyph[2]>box[2]+padx or glyph[3]>box[3]+pady):
                return result|{"reason":"native_glyph_escapes_coarse_hypothesis"}
            precise=[max(0,glyph[0]-padx),max(0,glyph[1]-pady),
                min(1,glyph[2]+padx),min(1,glyph[3]+pady)]
            bounded,reason=candidate_boxes(verdict|{"residual_regions":[{"box":precise}]},before.size,feather)
            # One dirty output glyph vetoes the entire proposal set, including
            # other perfectly preserved real labels. Never choose only safe ones.
            if reason:return result|{"reason":"output_glyph_intersects_original_edit"}
            source=detector.inspect(before,bounded)
            pixels=preserved_patch(before,after,precise)
            trace.setdefault("anchors",[]).append({"box":precise,"source":source,"pixels":pixels})
            if source.get("status")!="text_present" or pixels.get("unchanged") is not True:
                return result|{"reason":"original_glyph_or_pixel_identity_not_established"}
            if not any(_overlap(precise,f["box"]) for f in findings):
                findings.append({"issue":"physical_text","box":precise,
                    "observation":"候选原生笔画在编辑范围外且原图保留；仍须双方载体证明及完整复验。"})
            if len(findings)>2:return result|{"reason":"too_many_glyph_candidates"}
    return result|{"findings":findings,"reason":"geometry_candidates_only_all_carrier_and_global_checks_still_required"}
