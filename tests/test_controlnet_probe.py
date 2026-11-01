from types import SimpleNamespace

import pytest


def tiny_model():
    torch=pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TransformerBlock

    from watermark_studio.controlnet_probe import make_branch
    torch.manual_seed(451)
    config=SimpleNamespace(num_attention_heads=2,attention_head_dim=8,mlp_ratio=3,eps=1e-6)
    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks=torch.nn.ModuleList([QwenImage21TransformerBlock(16,2,8) for _ in range(4)])
        def forward(self,hidden_states,**kwargs):
            for block in self.transformer_blocks:
                hidden_states=block(hidden_states=hidden_states,**kwargs)
            return hidden_states
    return torch,Base().eval(),make_branch(config,(0,2)).eval()


def test_streamed_control_injection_matches_all_hints_equation_and_batch_rows():
    from watermark_studio.controlnet_probe import control_forward
    torch,base,branch=tiny_model()
    x=torch.randn(2,8,16);context=torch.randn(2,6,129)
    target=torch.tensor([False,False,True,True,True,True,True,True])
    kwargs={"modulation":torch.randn(3,64),"target_token_mask":target}
    with torch.inference_mode():
        c=torch.zeros_like(x);c[:,target]=branch.control_img_in(context)
        c=branch.control_blocks[0].before_proj(c)+x
        hints=[]
        for block in branch.control_blocks:
            c=block(c,**kwargs);hints.append(block.after_proj(c))
        reference=x.clone()
        for i,block in enumerate(base.transformer_blocks):
            reference=block(reference,**kwargs)
            if i%2==0:
                reference=reference+hints[i//2]*.75
        with control_forward(base,branch,context,.75):
            result=base(x,**kwargs)
        assert torch.equal(result,reference)
        switched=kwargs|{"modulation":kwargs["modulation"][[1,0,2]]}
        with control_forward(base,branch,context.flip(0),.75):
            swapped=base(x.flip(0),**switched)
        assert torch.allclose(result,swapped.flip(0),atol=1e-6,rtol=1e-6)
    assert not base._forward_pre_hooks and all(not b._forward_hooks for b in base.transformer_blocks)


def test_zero_strength_is_exact_no_op_and_failed_scope_removes_every_hook():
    from watermark_studio.controlnet_probe import control_forward
    torch,base,branch=tiny_model()
    x=torch.randn(1,8,16);kwargs={"modulation":torch.randn(2,64),
        "target_token_mask":torch.tensor([False,False,True,True,True,True,True,True])}
    with torch.inference_mode():
        reference=base(x,**kwargs)
        with control_forward(base,branch,torch.randn(1,6,129),0):
            assert torch.equal(base(x,**kwargs),reference)
        with pytest.raises(ValueError,match="token positions"),control_forward(base,branch,torch.randn(1,7,129)):
            base(x,**kwargs)
        with pytest.raises(ValueError,match="KV cache"),control_forward(base,branch,torch.randn(1,6,129)):
            base(x,**kwargs,kv_cache=object())
    assert not base._forward_pre_hooks
    assert all(not b._forward_hooks and not b._forward_pre_hooks for b in base.transformer_blocks)


def test_inpaint_context_uses_full_masked_rgba_and_keep_mask_channel():
    from PIL import Image

    from watermark_studio.controlnet_probe import packed_inpaint_context
    torch=pytest.importorskip("torch")
    seen={}
    class Processor:
        def preprocess(self,image,**kw):
            assert image.size==(16,16)
            return torch.full((1,3,16,16),.7)
    class Pipe:
        _execution_device="cpu"
        transformer=SimpleNamespace(dtype=torch.float32)
        image_processor=Processor()
        def _encode_vae_image(self,rgba,generator=None):
            seen["rgba"]=rgba.clone()
            return torch.ones(1,64,1,2,2)*.4
        def _pack_latents(self,x,b,c,h,w):
            return x.permute(0,2,3,4,1).reshape(b,-1,c)
    image=Image.new("RGB",(16,16),"green");original=image.tobytes()
    mask=Image.new("L",image.size,0);mask.paste(255,(0,0,8,16))
    result=packed_inpaint_context(Pipe(),image,mask,16,16)
    assert result.shape==(1,4,129) and torch.equal(result[:,:,:64],torch.zeros(1,4,64))
    assert torch.equal(result[0,:,64],torch.tensor([0.,1.,0.,1.]))
    assert torch.allclose(result[:,:,65:],torch.full((1,4,64),.4))
    assert seen["rgba"].shape==(1,4,1,16,16)
    assert seen["rgba"][:,:3,:,:,0:8].count_nonzero()==0
    assert torch.all(seen["rgba"][:,3]==1) and image.tobytes()==original
