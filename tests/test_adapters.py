import pytest

from watermark_studio.adapters import adapter_plan


def shapes():
    return {"diffusion_model.block.q.lora_A.weight":[2,3],
            "diffusion_model.block.q.lora_B.weight":[4,2]}


def test_adapter_plan_requires_explicit_alpha_and_exact_target():
    plan=adapter_plan(shapes(),{"block.q.weight":[4,3]},alpha=2)
    assert plan==[{"a":"diffusion_model.block.q.lora_A.weight",
                  "b":"diffusion_model.block.q.lora_B.weight",
                  "target":"block.q.weight","rank":2,"alpha":2}]
    with pytest.raises(ValueError,match="Missing explicit alpha"):
        adapter_plan(shapes(),{"block.q.weight":[4,3]})
    assert adapter_plan(shapes(),{"block.q.weight":[4,3]},metadata_alpha="4")[0]["alpha"]==4


@pytest.mark.parametrize("change",["unknown","shape","missing","extra","duplicate","alpha"])
def test_bad_adapter_cannot_enter_mutation_phase(change):
    values, targets, alpha=shapes(),{"block.q.weight":[4,3]},2
    if change=="unknown":targets={}
    elif change=="shape":targets={"block.q.weight":[3,4]}
    elif change=="missing":values.pop("diffusion_model.block.q.lora_B.weight")
    elif change=="extra":values["mystery.weight"]=[2,2]
    elif change=="duplicate":
        values.update({k.replace("diffusion_model.","transformer."):v for k,v in list(values.items())})
    else:alpha=float("nan")
    with pytest.raises(ValueError):
        adapter_plan(values,targets,alpha=alpha)


def test_real_cpu_fusion_uses_alpha_over_rank_and_no_dynamic_modules(tmp_path):
    torch=pytest.importorskip("torch")
    safetensors=pytest.importorskip("safetensors.torch")
    from watermark_studio.adapters import fuse_cpu_adapter
    model=torch.nn.Module();model.block=torch.nn.Module();model.block.q=torch.nn.Linear(3,4,bias=False)
    before=model.block.q.weight.detach().clone()
    a=torch.arange(6,dtype=torch.float32).reshape(2,3)/10
    b=torch.arange(8,dtype=torch.float32).reshape(4,2)/10
    path=tmp_path/"adapter.safetensors"
    safetensors.save_file({"diffusion_model.block.q.lora_A.weight":a,
                          "diffusion_model.block.q.lora_B.weight":b},str(path))
    report=fuse_cpu_adapter(model,path,scale=.5,alpha=4)
    assert torch.allclose(model.block.q.weight,before+b@a)
    assert report["pairs"]==1 and report["runtime_adapter_forwards"]==0
    assert next(iter(model.named_parameters()))[0]=="block.q.weight"


def test_failed_preflight_or_checksum_never_changes_a_parameter(tmp_path):
    torch=pytest.importorskip("torch")
    safetensors=pytest.importorskip("safetensors.torch")
    from watermark_studio.adapters import fuse_cpu_adapter
    model=torch.nn.Module();model.block=torch.nn.Module();model.block.q=torch.nn.Linear(3,4,bias=False)
    before=model.block.q.weight.detach().clone()
    path=tmp_path/"adapter.safetensors"
    safetensors.save_file({"diffusion_model.block.q.lora_A.weight":torch.zeros(2,3),
                          "diffusion_model.block.q.lora_B.weight":torch.full((4,2),float("nan"))},str(path))
    for kwargs in ({"expected_sha256":"0"*64},{}):
        with pytest.raises(ValueError):
            fuse_cpu_adapter(model,path,alpha=2,**kwargs)
        assert torch.equal(model.block.q.weight,before)



def test_fused_gate_up_plan_splits_rows_gate_first_then_projection():
    values={"diffusion_model.block.img_mlp.gate_up.lora_A.weight":[2,3],
            "diffusion_model.block.img_mlp.gate_up.lora_B.weight":[8,2]}
    target={"block.img_mlp.gate_layer.weight":[4,3],"block.img_mlp.proj.weight":[4,3]}
    item=adapter_plan(values,target,alpha=2)[0]
    assert item["split_targets"]==[
        {"target":"block.img_mlp.gate_layer.weight","rows":[0,4]},
        {"target":"block.img_mlp.proj.weight","rows":[4,8]}]
    with pytest.raises(ValueError):
        adapter_plan(values,{"block.img_mlp.gate_layer.weight":[4,3]},alpha=2)
    with pytest.raises(ValueError):
        adapter_plan(shapes()|{"diffusion_model.block.q.alpha":[]},{"block.q.weight":[4,3]},
                     metadata_alpha=2,pair_alphas={"diffusion_model.block.q.alpha":0})


def test_real_fused_swiglu_lora_matches_both_split_matrices(tmp_path):
    torch=pytest.importorskip("torch")
    safetensors=pytest.importorskip("safetensors.torch")
    from watermark_studio.adapters import fuse_cpu_adapter
    model=torch.nn.Module();model.block=torch.nn.Module();model.block.img_mlp=torch.nn.Module()
    mlp=model.block.img_mlp
    mlp.gate_layer=torch.nn.Linear(3,4,bias=False);mlp.proj=torch.nn.Linear(3,4,bias=False)
    gate,up=mlp.gate_layer.weight.detach().clone(),mlp.proj.weight.detach().clone()
    a=torch.arange(6,dtype=torch.float32).reshape(2,3)/10
    b=torch.arange(16,dtype=torch.float32).reshape(8,2)/10
    path=tmp_path/"split.safetensors"
    safetensors.save_file({"diffusion_model.block.img_mlp.gate_up.lora_A.weight":a,
                          "diffusion_model.block.img_mlp.gate_up.lora_B.weight":b},str(path))
    report=fuse_cpu_adapter(model,path,alpha=2)
    assert torch.allclose(mlp.gate_layer.weight,gate+(b@a)[:4])
    assert torch.allclose(mlp.proj.weight,up+(b@a)[4:])
    x=torch.tensor([[.1,.3,.5]])
    expected=torch.nn.functional.silu(x@(gate+(b@a)[:4]).T)*(x@(up+(b@a)[4:]).T)
    assert torch.allclose(torch.nn.functional.silu(mlp.gate_layer(x))*mlp.proj(x),expected)
    assert report["split_gate_up_pairs"]==1 and report["target_weights"]==2
