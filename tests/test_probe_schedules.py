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


def test_midpoint_probes_keep_trained_nodes_and_exact_forward_budget():
    original=[1.,.978453,.95418,.926626,.89508,.845148,.704534,.414568]
    for profile,count in (("mid10",10),("mid12",12)):
        result=tail_sigma_grid(original,profile)
        assert len(result)==count and all(value in result for value in original)
        assert all(a>b for a,b in pairwise(result))
        assert (.704534+.414568)/2 in result and .414568/2 in result
        assert result[-1]>0
    assert tail_sigma_grid(original,"mid12")[-1]==.414568/4


def test_published_three_step_grid_is_not_a_tail_refinement():
    original=[1.,.978453,.95418,.926626,.89508,.845148,.704534,.414568]
    assert tail_sigma_grid(original,"turturbo3")==[1.,.845148,.414568]


def test_base_probe_steps_cannot_mix_runtime_batches_or_replace_turbo(tmp_path):
    import json

    from watermark_studio.probe_schedules import (
        probe_inference_step_options,
        validate_base_probe,
    )
    assert probe_inference_step_options([{},{}])=={}
    assert probe_inference_step_options([{"_probe_num_inference_steps":16}]*2)=={"num_inference_steps":16}
    for values in ([16,None],[16,20],[True],[16.0],[0],[8]):
        with pytest.raises(ValueError):
            probe_inference_step_options([{"_probe_num_inference_steps":v} for v in values])
    index=tmp_path/"model_index.json"
    index.write_text(json.dumps({"_class_name":"QwenImage21Pipeline"}))
    assert validate_base_probe(tmp_path,40)["requested_steps"]==40
    index.write_text(json.dumps({"_class_name":"QwenImage21Pipeline","sample_sigmas":[1.,.5]}))
    with pytest.raises(ValueError):
        validate_base_probe(tmp_path,40)
