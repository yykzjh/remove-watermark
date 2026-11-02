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


def test_attention_mlp_scope_trains_and_fuses_each_separate_swiglu_projection(tmp_path):
    import copy

    import pytest
    torch=pytest.importorskip("torch")
    from torch import nn

    from watermark_studio.adapters import fuse_cpu_adapter
    from watermark_studio.lora_training import adapter_tensors, install_attention_lora
    class Mlp(nn.Module):
        def __init__(self):
            super().__init__();self.proj=nn.Linear(4,12,bias=False)
            self.gate_layer=nn.Linear(4,12,bias=False);self.out=nn.Linear(12,4,bias=False)
        def forward(self,value):
            return self.out(torch.nn.functional.silu(self.gate_layer(value))*self.proj(value))
    class Attention(nn.Module):
        def __init__(self):
            super().__init__();self.to_q=nn.Linear(4,4,bias=False);self.to_k=nn.Linear(4,4,bias=False)
            self.to_v=nn.Linear(4,4,bias=False);self.to_out=nn.ModuleList([nn.Linear(4,4,bias=False)])
        def forward(self,value):return self.to_out[0](self.to_q(value)+self.to_k(value)+self.to_v(value))
    class Block(nn.Module):
        def __init__(self):
            super().__init__();self.attn=Attention();self.img_mlp=Mlp()
        def forward(self,value):return value+self.img_mlp(self.attn(value))
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.transformer_blocks=nn.ModuleList([Block()])
        def forward(self,value):return self.transformer_blocks[0](value)
    model=Model();base=copy.deepcopy(model)
    spec=install_attention_lora(model,rank=4,alpha=4,include_mlp=True)
    assert spec["scope"]=="attention_mlp" and len(spec["targets"])==7
    parameters=[v for v in model.parameters() if v.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=.01)
    inputs=torch.randn(2,5,4)
    for _ in range(3):
        optimizer.zero_grad();model(inputs).square().mean().backward();optimizer.step()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    safe=pytest.importorskip("safetensors.torch")
    path=tmp_path/"lora.safetensors";safe.save_file(adapter_tensors(model),str(path))
    fused=fuse_cpu_adapter(base,path,alpha=4)
    assert fused["pairs"]==7
    torch.testing.assert_close(model(inputs),base(inputs),rtol=1e-5,atol=1e-6)


def test_pinned_warm_start_only_updates_matching_lora_matrices(tmp_path):
    import copy
    import hashlib

    torch=pytest.importorskip("torch")
    from safetensors.torch import save_file

    from watermark_studio.lora_training import (
        adapter_tensors,
        initialize_adapter,
        install_attention_lora,
    )
    first=tiny_model(torch);second=copy.deepcopy(first)
    install_attention_lora(first,rank=4,alpha=4);install_attention_lora(second,rank=4,alpha=4)
    with torch.no_grad():
        for module in first.modules():
            if getattr(module,"_watermark_training_lora",False):module.lora_B.fill_(.01)
    path=tmp_path/"init.safetensors"
    save_file(adapter_tensors(first),str(path),metadata={"rank":"4","alpha":"4","model_index_sha256":"c"*64})
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    before={n:p.detach().clone() for n,p in second.named_parameters() if not p.requires_grad}
    with pytest.raises(ValueError,match="bytes"):initialize_adapter(second,path,"0"*64,"c"*64)
    with pytest.raises(ValueError,match="identity"):initialize_adapter(second,path,sha,"d"*64)
    audit=initialize_adapter(second,path,sha,"c"*64)
    assert audit["matrices"]==16
    for name,value in second.named_parameters():
        if not value.requires_grad:assert torch.equal(value,before[name])
    x=torch.randn(1,5,12);torch.testing.assert_close(first(x),second(x))
