"""Experimental bounded inquiry into a watermark-only dispute; never waives other vetoes."""
from __future__ import annotations


def eligible(verdict,context):
    explanation=verdict.get("model_explanation") or {}
    outside=verdict.get("outside_repair_change") or {}
    return (
        context.get("background_preserved") is True
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
