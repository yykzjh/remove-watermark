"""Real Torch SDPA/cache invariants for an offline conditioning ablation."""
from types import SimpleNamespace

import pytest

torch=pytest.importorskip("torch")
pytest.importorskip("diffusers")
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel
from torch import nn
from torch.nn import functional as F

from watermark_studio.attention import IndependentSDPAProcessor
from watermark_studio.reference_cache import ReferencePrefixCache
from watermark_studio.reference_exclusion import reference_exclusion


class Cache:
    def store(self,key,value):self.values=(key,value)
    def get(self):return self.values


def attn_layer():
    return SimpleNamespace(to_q=nn.Linear(8,8,bias=False),to_k=nn.Linear(8,8,bias=False),
        to_v=nn.Linear(8,8,bias=False),heads=2,norm_q=nn.Identity(),norm_k=nn.Identity(),
        to_out=[nn.Identity(),nn.Identity()])


@pytest.mark.parametrize("exclude,exclude_prefix,compact",[
    (False,False,False),(True,False,False),(True,True,False),(True,False,True),(True,True,True)])
def test_target_reference_exclusion_is_exact_sdpa_and_preserves_prefix_and_cache(exclude,exclude_prefix,compact):
    torch.manual_seed(93)
    attn=attn_layer();processor=IndependentSDPAProcessor("native");cache=Cache()
    hidden=torch.randn(2,14,8);valid=torch.ones(2,14,dtype=torch.bool);valid[1,1]=False
    segments=[(0,2,True),(2,6,False),(6,8,True)]
    keep=torch.ones(2,14,dtype=torch.bool)
    if exclude:
        keep[0,2:3]=False;keep[1,3:6]=False
        processor._watermark_target_key_keep=keep
        processor._watermark_target_query_count=6
        processor._watermark_exclude_prefix_keys=exclude_prefix
        if compact:processor._watermark_prefix_cache_plan=ReferencePrefixCache(keep,6)
    allowed=torch.zeros(2,14,14,dtype=torch.bool)
    for start,end,is_text in segments:
        allowed[:,start:end,:start]=True
        allowed[:,start:end,start:end]=(torch.tril(torch.ones(end-start,end-start,dtype=torch.bool))
            if is_text else True)
    allowed[:,8:]=keep[:,None,:];allowed &= valid[:,None,:]
    if exclude_prefix:allowed &= keep[:,None,:]
    q,k,v=[layer(hidden).unflatten(-1,(2,4)).transpose(1,2)
        for layer in (attn.to_q,attn.to_k,attn.to_v)]
    expected=F.scaled_dot_product_attention(q,k,v,attn_mask=allowed[:,None]).transpose(1,2).flatten(2)
    full=processor(attn,hidden,layer_cache=cache,kv_cache_mode="extract",
        cache_write_slice=slice(0,8),segments=segments,key_valid=valid)
    torch.testing.assert_close(full,expected,atol=2e-6,rtol=2e-5)
    cached=processor(attn,hidden[:,8:],layer_cache=cache,kv_cache_mode="cached",
        attention_mask=valid[:,None,None,:])
    torch.testing.assert_close(cached,expected[:,8:],atol=2e-6,rtol=2e-5)
    # Exclusion must never perturb source/text prefix computation or stored K/V.
    baseline=IndependentSDPAProcessor("native")(attn,hidden,segments=segments,key_valid=valid)
    if not exclude_prefix:
        torch.testing.assert_close(full[:,:8],baseline[:,:8],atol=0,rtol=0)
    else:
        assert not torch.equal(full[:,2:8],baseline[:,2:8])
    prefix=k.transpose(1,2)[:,:8]
    if compact:
        prefix=processor._watermark_prefix_cache_plan.pack(prefix)
        assert cache.values[0].shape[1]==7
    torch.testing.assert_close(cache.values[0],prefix,atol=0,rtol=0)


class Transformer:
    build_token_metadata=staticmethod(QwenImage21Transformer2DModel.build_token_metadata)
    def __init__(self):
        self.processor=IndependentSDPAProcessor("native")
        self.transformer_blocks=[SimpleNamespace(attn=SimpleNamespace(processor=self.processor))]
    def forward(self,hidden_states,img_mask,img_shapes,timestep=None):
        keep=getattr(self.processor,"_watermark_target_key_keep",None)
        return None if keep is None else keep.clone()


@pytest.mark.parametrize("exclude_prefix",[False,True])
@pytest.mark.parametrize("compact",[False,True])
def test_exact_joint_layout_independent_masks_and_exception_cleanup(exclude_prefix,compact):
    transformer=Transformer();pipe=SimpleNamespace(transformer=transformer)
    # Four VLM slots per4x4-token image, with two interleaved text positions.
    img_mask=torch.tensor([[False,True,True,True,True,False,True,True,True,True]]*2)
    shapes=[[(1,4,4),(1,4,4)]]*2
    items=[{"_reference_exclusion_regions":[{"box":[0,0,.5,.5]}]},
           {"_reference_exclusion_regions":[{"box":[.5,.5,1,1]}]}]
    old_forward=transformer.forward
    with pytest.raises(RuntimeError,match="test cleanup"), reference_exclusion(pipe,items,enabled=True,
            exclude_prefix=exclude_prefix,compact_cache=compact) as audit:
        keep=transformer.forward(torch.zeros(2,32,8),img_mask,shapes)
        assert keep.shape==(2,34) and audit["forward_calls"]==1
        assert audit["excluded_source_fractions"]==[.25,.25]
        # source positions1..16, text positions0/17, target positions18..33
        assert not keep[0,1] and keep[1,1] and keep[0,16] and not keep[1,16]
        assert keep[:,[0,17]].all() and keep[:,18:].all()
        assert audit["whole_source_pixels_unchanged"]
        assert audit["prefix_computation_unchanged"] is not exclude_prefix
        if compact:
            assert audit["original_prefix_tokens"]==18 and audit["compact_prefix_tokens"]==14
        raise RuntimeError("test cleanup")
    assert transformer.forward==old_forward
    assert not hasattr(transformer.processor,"_watermark_target_key_keep")
    assert not hasattr(transformer.processor,"_watermark_target_query_count")
    assert not hasattr(transformer.processor,"_watermark_exclude_prefix_keys")
    assert not hasattr(transformer.processor,"_watermark_prefix_cache_plan")


def test_disabled_scope_does_not_modify_processor_or_forward():
    transformer=Transformer();pipe=SimpleNamespace(transformer=transformer);original=transformer.forward
    with reference_exclusion(pipe,[],enabled=False) as audit:
        assert transformer.forward==original and audit["forward_calls"]==0
        assert not hasattr(transformer.processor,"_watermark_target_key_keep")


def test_reference_margin_dilates_only_source_keys_and_rejects_all_source_removal():
    transformer=Transformer();pipe=SimpleNamespace(transformer=transformer)
    mask=torch.tensor([[False,True,True,True,True,False,True,True,True,True]])
    shapes=[[(1,4,4),(1,4,4)]]
    items=[{"_reference_exclusion_regions":[{"box":[0,0,.5,.5]}]}]
    with reference_exclusion(pipe,items,enabled=True,margin_cells=1) as audit:
        keep=transformer.forward(torch.zeros(1,32,8),mask,shapes)
        assert audit["excluded_source_fractions"]==[.5625]
        assert (~keep[0,1:17]).sum().item()==9
        assert keep[:,[0,17]].all() and keep[:,18:].all()
    with pytest.raises(ValueError,match="partial"), reference_exclusion(
            pipe,items,enabled=True,margin_cells=2):
        transformer.forward(torch.zeros(1,32,8),mask,shapes)
    assert not hasattr(transformer.processor,"_watermark_target_key_keep")


def test_late_reference_schedule_keeps_constant_prefix_and_uses_actual_normalized_sigma():
    transformer=Transformer();pipe=SimpleNamespace(transformer=transformer)
    mask=torch.tensor([[False,True,True,True,True,False,True,True,True,True]])
    shapes=[[(1,4,4),(1,4,4)]]
    items=[{"_reference_exclusion_regions":[{"box":[0,0,.5,.5]}]}]
    with reference_exclusion(pipe,items,enabled=True,max_sigma=.5) as audit:
        results=[transformer.forward(torch.zeros(1,32,8),mask,shapes,
            timestep=torch.tensor([sigma])) for sigma in [.9,.5,.1]]
        assert results[0] is None and results[1].shape==(1,34)
        assert torch.equal(results[1],results[2])
        assert audit["forward_calls"]==3 and audit["active_forward_calls"]==2
        assert audit["prefix_computation_unchanged"] and not audit["compact_reference_cache"]
    with pytest.raises(ValueError,match="unchanged full prefix"), reference_exclusion(
            pipe,items,enabled=True,exclude_prefix=True,max_sigma=.5):
        pass
    with pytest.raises(ValueError,match="unchanged full prefix"), reference_exclusion(
            pipe,items,enabled=True,compact_cache=True,max_sigma=.5):
        pass
