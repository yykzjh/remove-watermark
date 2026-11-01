import hashlib

import pytest

from watermark_studio.decoder_probe import logical_config_sha


def test_decoder_compatibility_digest_ignores_descriptive_metadata_but_not_latent_scaling():
    config={"z_dim":64,"latents_std":[.2,.3],"_diffusers_version":"old"}
    assert logical_config_sha(config)==logical_config_sha(config|{"_diffusers_version":"new"})
    assert logical_config_sha(config)!=logical_config_sha(config|{"latents_std":[.3,.3]})


def test_decoder_weight_loader_requires_digest_all_keys_and_finite_tensors(tmp_path):
    torch=pytest.importorskip("torch")
    save=pytest.importorskip("safetensors.torch").save_file
    from watermark_studio.decoder_probe import strict_vae_weights
    weights=tmp_path/"weights.safetensors"
    source=torch.nn.Linear(2,3)
    save(source.state_dict(),weights)
    sha=lambda:hashlib.sha256(weights.read_bytes()).hexdigest()
    loaded=strict_vae_weights(torch.nn.Linear(2,3),weights,sha())
    assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in source.state_dict().items())
    with pytest.raises(ValueError,match="SHA"):
        strict_vae_weights(torch.nn.Linear(2,3),weights,"0"*64)
    save({"weight":source.weight.detach()},weights)
    with pytest.raises(ValueError,match="Every"):
        strict_vae_weights(torch.nn.Linear(2,3),weights,sha())
    values=source.state_dict();values["bias"]=torch.full_like(values["bias"],float("nan"))
    save(values,weights)
    with pytest.raises(ValueError,match="Nonfinite"):
        strict_vae_weights(torch.nn.Linear(2,3),weights,sha())
