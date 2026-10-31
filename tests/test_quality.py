import json

import pytest
from PIL import Image

from watermark_studio.config import ProcessingConfig
from watermark_studio.quality import parse_review, pixel_change


def verdict(**changes):
    return json.dumps({"watermark_removed": True, "natural": True,
                       "content_preserved": True, "uncertain": False, **changes})


def test_pass_requires_all_checks_and_no_uncertainty():
    assert parse_review(verdict())["status"] == "passed"
    for flag in ("watermark_removed", "natural", "content_preserved"):
        assert parse_review(verdict(**{flag: False}))["status"] == "review"
    assert parse_review(verdict(uncertain=True))["status"] == "review"


@pytest.mark.parametrize("text", ['{}', 'not json', '{"watermark_removed":"true"}'])
def test_incomplete_or_invalid_verdict_is_never_a_pass(text):
    with pytest.raises((ValueError, TypeError)):
        parse_review(text)


def test_size_change_alone_has_no_change_in_constant_color():
    assert pixel_change(Image.new("RGB", (400, 300), "blue"),
                        Image.new("RGB", (800, 600), "blue")) == 0


def test_quality_is_opt_in_and_parameters_are_bounded():
    assert not ProcessingConfig().quality_check
    with pytest.raises(ValueError):
        ProcessingConfig(quality_batch_size=16)
