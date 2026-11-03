import runpy
from pathlib import Path

import pytest

from watermark_studio.native_text import normalized_candidates


def test_rotated_and_axis_aligned_native_candidates_map_to_the_same_actual_photo():
    quad=[[10,20],[90,25],[85,60],[5,55]]
    rows=normalized_candidates([quad,[100,30,140,70]],[.8,.6],(200,100))
    assert rows[0]["box"]==[.025,.2,.45,.6]
    assert rows[1]["box"]==[.5,.3,.7,.7]
    assert rows[0]["score"]==.8


@pytest.mark.parametrize("boxes,scores", [
    ([[0,0,300,50]],[.9]),
    ([[0,0,20,20]],[float("nan")]),
    ([[0,0,0,10]],[.8]),
    ([[0,0,20,20]],[]),
])
def test_bad_ocr_geometry_or_scores_cannot_turn_into_a_clear_scan(boxes,scores):
    with pytest.raises(ValueError):
        normalized_candidates(boxes,scores,(200,100))


def test_giant_or_wrong_location_boxes_cannot_count_as_injected_text_coverage():
    script=Path(__file__).parents[1]/"scripts/probe-native-ocr-coverage.py"
    covered=runpy.run_path(str(script))["location_covered"]
    target=[.6,.8,.8,.85]
    assert covered(target,target)
    assert not covered(target,[0,0,1,1])
    assert not covered(target,[.01,.01,.1,.1])
