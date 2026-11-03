import numpy as np
from PIL import Image, ImageDraw

from watermark_studio import detection, evidence, quality
from watermark_studio.config import Region


def models_pass(monkeypatch):
    def inspect(chats, fields, *args, **kwargs):
        return [{**{key: True for key in fields}, "uncertain": False, "area": "none"} for _ in chats], 0.
    monkeypatch.setattr(quality, "_structured_vqa", inspect)
    monkeypatch.setattr(detection, "locate_watermarks_batch",
                        lambda images, *args, **kwargs: [([], "") for _ in images])
    monkeypatch.setattr(evidence, "attach_evidence", lambda *args, **kwargs: None)


def pair():
    clean = Image.fromarray(np.random.default_rng(781).integers(30, 190, (480, 640, 3), dtype=np.uint8))
    marked = clean.copy()
    ImageDraw.Draw(marked).rectangle((64, 48, 511, 383), fill=(220, 220, 220))
    return marked, clean


def test_external_broad_outdoor_panel_cannot_bypass_source_policy_even_if_every_model_approves(monkeypatch):
    models_pass(monkeypatch)
    result = quality.verify_batch(
        [pair()], object(), object(), "cpu", source_regions=[[Region(kind="panel", box=(.1, .1, .8, .8))]],
        editor_context=[{"source_scene": "outdoor"}], detail_patches=0)[0]
    assert result["model_status"] == "passed"
    assert result["status"] == "review" and result["uncertain"]
    assert "source_physical_ambiguity" in result["guard_codes"]
    assert result["source_footprint_validation"]["policy"] == "source-broad-consensus-v1"


def test_explicit_manual_custom_scope_does_not_claim_to_be_an_auto_detected_broad_panel(monkeypatch):
    models_pass(monkeypatch)
    result = quality.verify_batch(
        [pair()], object(), object(), "cpu", source_regions=[[Region(kind="custom", box=(.1, .1, .8, .8))]],
        editor_context=[{"source_scene": "outdoor"}], detail_patches=0)[0]
    assert "source_footprint_validation" not in result
    assert "source_physical_ambiguity" not in result.get("guard_codes", [])
