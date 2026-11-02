from itertools import islice, pairwise

import numpy as np
import pytest

from watermark_studio.training_sampling import (
    crop_groups,
    explicit_training_sigmas,
    lower_balanced_cycle,
    lower_balanced_validation,
)


def rows(split,count=12):
    return [{"id":str(i),"split":split,"source_original_sha256":hex(i),
        **({"training_crop_kind":"lower"} if i<4 else {})} for i in range(count)]


def test_lower_sampling_retains_roles_and_visits_each_group_without_replacement():
    data=rows("train");cache=[{k:r[k] for k in ("id","split","source_original_sha256")} for r in data]
    sampled=list(islice(lower_balanced_cycle(cache,data,77),16))
    assert all(int(r["id"])<4 for r in sampled[::2])
    assert all(int(r["id"])>=4 for r in sampled[1::2])
    assert len({r["id"] for r in sampled[:8:2]})==4
    assert len({r["id"] for r in sampled[1:16:2]})==8
    assert sampled==list(islice(lower_balanced_cycle(cache,data,77),16))
    with pytest.raises(ValueError):crop_groups([{**cache[0],"split":"validation"}],data,"train")
    with pytest.raises(ValueError):crop_groups([{**cache[0],"source_original_sha256":"wrong"}],data,"train")


def test_balanced_validation_preserves_original_validation_roles():
    data=rows("validation")
    chosen=lower_balanced_validation(data,data,8)
    assert len(chosen)==8 and sum(r.get("training_crop_kind")=="lower" for r in chosen)==4
    assert all(r["split"]=="validation" for r in chosen)
    with pytest.raises(ValueError):lower_balanced_validation(data,data,10)


def test_explicit_sigma_profiles_keep_positive_descending_static_nodes_and_reject_dynamic_changes():
    config={"_class_name":"FlowMatchEulerDiscreteScheduler","shift":1,"use_dynamic_shifting":False}
    original={"sample_sigmas":[1.,.978453,.95418,.926626,.89508,.845148,.704534,.414568]}
    assert explicit_training_sigmas("continuous",{}, {}) is None
    for steps in (12,24,40):
        values=explicit_training_sigmas("uniform"+str(steps),config,original)
        assert values==np.linspace(1,1/steps,steps).astype(np.float32).tolist()
        assert values[0]==1 and values[-1]>0 and all(a>b for a,b in pairwise(values))
    assert len(explicit_training_sigmas("turbo8",config,original))==8
    with pytest.raises(ValueError):explicit_training_sigmas("uniform12",{**config,"shift":3},original)
    with pytest.raises(ValueError):explicit_training_sigmas("uniform12",{**config,"use_dynamic_shifting":True},original)


def test_uniform_profile_matches_actual_diffusers_euler_cpu_schedule():
    pytest.importorskip("torch")
    diffusers=pytest.importorskip("diffusers")
    config={"_class_name":"FlowMatchEulerDiscreteScheduler","shift":1,"use_dynamic_shifting":False}
    scheduler=diffusers.FlowMatchEulerDiscreteScheduler(shift=1,use_dynamic_shifting=False)
    for steps in (12,24,40):
        scheduler.set_timesteps(steps,device="cpu",sigmas=np.linspace(1,1/steps,steps))
        assert explicit_training_sigmas("uniform"+str(steps),config,{})==scheduler.sigmas[:-1].tolist()
