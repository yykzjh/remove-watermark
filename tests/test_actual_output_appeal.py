"""Physical-text agreement must never override an independent image-quality veto."""
import runpy
from pathlib import Path

import pytest

support = runpy.run_path(str(Path(__file__).parents[1]/"scripts/probe-ocr-bound-counterfactual.py"))["actual_views_support"]


@pytest.mark.parametrize("which,patch", [
    ("base", {"natural": False}), ("base", {"content_preserved": False}),
    ("base", {"uncertain": True}), ("base", {"guard_codes": ["repair_texture_collapse"]}),
    ("presence", {"overlay_present": True}), ("presence", {"uncertain": True}),
    ("editor", {"watermark_removed": False}), ("editor", {"natural": False}),
    ("editor", {"uncertain": True}), ("editor", {"watermark_removed": 1}),
])
def test_any_actual_output_veto_blocks_agreement(which, patch):
    base = {"natural": True, "content_preserved": True, "uncertain": False}
    presence = {"overlay_present": False, "uncertain": False}
    editor = {"watermark_removed": True, "natural": True, "uncertain": False}
    assert support(base, presence, editor)
    {"base": base, "presence": presence, "editor": editor}[which].update(patch)
    assert not support(base, presence, editor)
