import pytest


def tiny_model(torch):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.transformer_blocks=torch.nn.ModuleList()
            for _ in range(2):
                attn=torch.nn.Module()
                for name in ["to_q","to_k","to_v"]:setattr(attn,name,torch.nn.Linear(12,12))
                attn.to_out=torch.nn.ModuleList([torch.nn.Linear(12,12)])
                block=torch.nn.Module();block.attn=attn;self.transformer_blocks.append(block)
        def forward(self,x):
            for block in self.transformer_blocks:
                x=block.attn.to_out[0](block.attn.to_q(x)+block.attn.to_k(x)+block.attn.to_v(x))
            return x
    return Tiny()


def test_zero_adapter_is_identity_and_only_lora_parameters_receive_updates(tmp_path):
    torch=pytest.importorskip("torch")
    from safetensors.torch import save_file

    from watermark_studio.adapters import fuse_cpu_adapter
    from watermark_studio.lora_training import adapter_tensors, install_attention_lora
    torch.manual_seed(42);model=tiny_model(torch);original=tiny_model(torch);original.load_state_dict(model.state_dict())
    x=torch.randn(2,5,12);before=model(x).detach()
    audit=install_attention_lora(model,rank=4,alpha=4)
    assert len(audit["targets"])==8 and torch.equal(model(x),before)
    base={name:p.detach().clone() for name,p in model.named_parameters() if not p.requires_grad}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    for _ in range(2):
        optimizer.zero_grad();model(x).square().mean().backward();optimizer.step()
    assert all(p.grad is None and torch.equal(p,base[n]) for n,p in model.named_parameters() if not p.requires_grad)
    path=tmp_path/"pilot.safetensors";save_file(adapter_tensors(model),path)
    fuse_cpu_adapter(original,path,alpha=4)
    assert torch.allclose(model(x),original(x),atol=2e-6,rtol=2e-6)


def test_flow_loss_has_exact_noise_minus_clean_target_and_preserves_outside_weight():
    torch=pytest.importorskip("torch")
    from watermark_studio.lora_training import flow_objective
    clean=torch.randn(1,12,8);noise=torch.randn_like(clean);mask=torch.zeros((1,12))
    assert flow_objective(noise-clean,noise,clean,mask).item()==0
    error=(noise-clean).clone();error[:,0]+=1
    assert flow_objective(error,noise,clean,mask)>0
    with pytest.raises(ValueError):
        flow_objective(error,noise,clean,mask,outside_weight=0)
