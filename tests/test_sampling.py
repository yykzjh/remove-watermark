from types import SimpleNamespace

import pytest

from watermark_studio.config import DeploymentConfig
from watermark_studio.deployment_cli import parse_deployment
from watermark_studio.sampling import (
    configure_uniform_sampling,
    runtime_step_options,
    verify_sampling_steps,
)


class Pipe:
    def __init__(self):
        self.config=SimpleNamespace(sample_sigmas=[1.,.98,.95,.93,.9,.85,.7,.4])
        self.scheduler=SimpleNamespace(timesteps=list(range(8)))
    def register_to_config(self,**values):
        for name,value in values.items():setattr(self.config,name,value)


def test_deployment_step_cli_defaults_to_checkpoint_and_roundtrips_explicit_counts(tmp_path):
    path=tmp_path/"base.toml";path.write_text("")
    config,_=parse_deployment(["--config",str(path)])
    assert config.uniform_steps==0
    for count in (12,16,20,24,40):
        config,_=parse_deployment(["--config",str(path),"--uniform-steps",str(count)])
        assert DeploymentConfig.model_validate_json(config.model_dump_json()).uniform_steps==count
    for count in ("8","13","12.0"):
        with pytest.raises(SystemExit):
            parse_deployment(["--config",str(path),"--uniform-steps",count])


def test_uniform_deployment_must_clear_only_instance_schedule_and_check_actual_step_count():
    default=Pipe();deployed=Pipe();original=list(default.config.sample_sigmas)
    assert runtime_step_options(default)=={} and default.config.sample_sigmas==original
    deployed._watermark_uniform_steps=12
    with pytest.raises(ValueError,match="override"):
        runtime_step_options(deployed)
    info=configure_uniform_sampling(deployed,12)
    assert info["original_sigmas"]==original and info["checkpoint_modified"] is False
    assert default.config.sample_sigmas==original
    options=runtime_step_options(deployed)
    assert options=={"num_inference_steps":12}
    with pytest.raises(RuntimeError,match="Actual"):
        verify_sampling_steps(deployed,options)
    deployed.scheduler.timesteps=list(range(12))
    verify_sampling_steps(deployed,options)


def test_explicit_deployment_schedule_rejects_conflicting_per_item_probe():
    pipe=Pipe();configure_uniform_sampling(pipe,24);pipe._watermark_uniform_steps=24
    assert runtime_step_options(pipe,{"num_inference_steps":24})=={"num_inference_steps":24}
    with pytest.raises(ValueError,match="cannot override"):
        runtime_step_options(pipe,{"num_inference_steps":12})
