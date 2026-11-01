from itertools import pairwise

import pytest

from watermark_studio.probe_schedules import tail_sigma_grid


def test_tail_probe_preserves_every_trained_sigma_and_only_adds_explicit_positive_tail():
    original=[1.,.978453,.95418,.926626,.89508,.845148,.704534,.414568]
    untouched=list(original)
    for profile,size in (("native",8),("tail9",9),("tail10",10),("tail12",12)):
        result=tail_sigma_grid(original,profile)
        assert len(result)==size and result[:8]==original
        assert result[-1]>0 and all(a>b for a,b in pairwise(result))
        assert original==untouched
    assert tail_sigma_grid(original,"tail10")[8:]==[original[-1]/2,original[-1]/4]


@pytest.mark.parametrize("values",[[1.,.5],list(range(8)),[1,.9,.8,.7,.6,.5,.4,0],
    [1,.9,.8,.7,.6,.5,.4,float("nan")],[1,.9,.8,.7,.6,.5,.4,True],[1,.9,.8,.7,.6,.5,.4,.4]])
def test_invalid_or_non_native_grid_cannot_be_silently_replaced(values):
    with pytest.raises(ValueError):
        tail_sigma_grid(values,"tail9")
