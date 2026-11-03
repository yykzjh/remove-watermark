import copy
import runpy
from pathlib import Path

import pytest

attach=runpy.run_path(str(Path(__file__).parents[1]/"scripts/benchmark_monitor.py"))["attachment_health"]


def fixture():
    health={"optimization":{"roi_steps":12},"quality_weight_identity":{"relation":"different_weights"},
            "code_revision":"original","gpus":[0,1,2,3],"backend":"qwen"}
    saved={"job":"existing","options":{"seed":42},"selection_sha256":"frozen","health":health}
    live=copy.deepcopy(health)|{"job":{"id":"existing","state":"running"}}
    return saved,live


def test_attach_reuses_original_health_without_mutating_the_record():
    saved,live=fixture()
    result=attach(saved,live,"existing",{"seed":42},"frozen")
    result["optimization"]["roi_steps"]=99
    assert saved["health"]["optimization"]["roi_steps"]==12


@pytest.mark.parametrize("change",["job","options","cohort","weights","optimization"])
def test_another_job_cohort_or_changed_deployment_cannot_attach(change):
    saved,live=fixture();options={"seed":42};cohort="frozen"
    if change=="job":live["job"]["id"]="different"
    if change=="options":options={"seed":43}
    if change=="cohort":cohort="different"
    if change=="weights":live["quality_weight_identity"]={"relation":"same"}
    if change=="optimization":live["optimization"]["roi_steps"]=8
    with pytest.raises(ValueError):
        attach(saved,live,"existing",options,cohort)
