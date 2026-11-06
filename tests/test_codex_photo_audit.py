import copy

import pytest

from watermark_studio.codex_photo_audit import freeze_plan, validate_verdict


def verdict():
    return {"acceptable": True, "watermark_removed": True, "natural_fill": True,
            "content_preserved": True, "uncertain": False, "defects": [],
            "evidence": ["Both overlays have disappeared; surrounding structures remain continuous."]}


@pytest.mark.parametrize("key", ["watermark_removed", "natural_fill", "content_preserved", "uncertain"])
def test_inconsistent_pass_cannot_be_accepted(key):
    value = verdict()
    value[key] = not value[key]
    with pytest.raises(ValueError, match="contradicts"):
        validate_verdict(value)


def test_no_evidence_is_not_a_valid_pass():
    value = verdict()
    value["evidence"] = []
    with pytest.raises(ValueError, match="evidence"):
        validate_verdict(value)


def test_sample_is_frozen_and_cannot_be_regraded_after_repairs(tmp_path):
    records = [{"file": str(n) + ".jpg", "sha256": str(n), "quality_status": "passed"}
               for n in range(1200)]
    records.append({"file": "review.jpg", "quality_status": "review"})
    plan = freeze_plan(records, tmp_path)
    assert len(set(plan["sample"])) == 1000
    assert plan["initially_rejected"] == ["review.jpg"]
    assert freeze_plan(records, tmp_path) == plan
    changed = copy.deepcopy(records)
    changed[0]["sha256"] = "repaired bytes"
    with pytest.raises(ValueError, match="changed"):
        freeze_plan(changed, tmp_path)
