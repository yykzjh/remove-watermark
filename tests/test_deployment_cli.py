import json

import pytest

from watermark_studio.config import DeploymentConfig
from watermark_studio.deployment_cli import make_parser, parse_deployment, render_config


def test_all_deployment_fields_have_cli_options_and_override_toml(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text('gpus=[1,2,3]\ncompile=true\nvae_tiling=true\nallowed_roots=["/old"]\n')
    config, printing = parse_deployment([
        "--config", str(path), "--parallelism", "single", "--gpus", "5",
        "--no-compile", "--no-vae-tiling", "--allowed-roots", "/new path", "/mnt",
        "--port", "18765", "--print-config",
    ])
    assert printing and config.gpus == [5] and config.capacity == 1
    assert not config.compile and not config.vae_tiling
    assert config.allowed_roots == ["/new path", "/mnt"] and config.port == 18765
    assert json.loads(render_config(config))["parallelism"] == "single"
    destinations = {action.dest for action in make_parser()._actions}
    assert set(DeploymentConfig.model_fields) <= destinations


@pytest.mark.parametrize("flags", [
    ["--parallelism", "single", "--gpus", "1,2"],
    ["--attention", "flex", "--no-compile"],
    ["--compile", "--offload", "model"],
    ["--gpus", "1,1"],
    ["--gpus", "oops"],
    ["--replicas-per-gpu", "9"],
    ["--parallelism", "tensor"],
])
def test_invalid_deployment_combinations_fail_before_start(tmp_path, flags):
    path = tmp_path / "base.toml"
    path.write_text("")
    with pytest.raises(SystemExit) as error:
        parse_deployment(["--config", str(path), *flags])
    assert error.value.code == 2
