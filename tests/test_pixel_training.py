import pytest


def test_pixel_loss_has_true_target_minimum_and_detects_shadow_and_texture_loss():
    torch=pytest.importorskip("torch")
    from watermark_studio.pixel_training import exact_pixel_objective
    clean=torch.rand(1,3,32,32)*2-1;mask=torch.zeros(1,1,32,32);mask[:,:,8:24,8:24]=1
    assert exact_pixel_objective(clean,clean,mask)==0
    shadow=clean.clone();shadow[:,:,8:24,8:24]-=.12;shadow.requires_grad_(True)
    loss=exact_pixel_objective(shadow,clean,mask);loss.backward()
    assert loss>0 and shadow.grad is not None and shadow.grad.abs().sum()>0
    blurred=torch.nn.functional.avg_pool2d(clean,3,1,1)
    assert exact_pixel_objective(blurred,clean,mask)>0
    with pytest.raises(ValueError):exact_pixel_objective(clean,clean,mask+1)


def test_decoder_input_restores_native_channel_first_layout_with_gradient():
    torch=pytest.importorskip("torch")
    from watermark_studio.pixel_training import packed_to_vae
    packed=torch.randn(1,16,4,requires_grad=True);mean=torch.arange(4.).reshape(1,4,1,1,1);std=torch.ones_like(mean)*2
    decoded=packed_to_vae(packed,64,64,mean,std)
    assert decoded.shape==(1,4,1,4,4)
    torch.testing.assert_close(decoded[:,:,0].flatten(2).transpose(1,2),packed*2+mean.flatten())
    decoded.sum().backward();assert torch.all(packed.grad==2)


def test_checkpointed_frozen_vae_preserves_rgb_gradient_to_packed_dit_output():
    from types import SimpleNamespace

    torch=pytest.importorskip("torch")
    from watermark_studio.pixel_training import decode_exact_pixels
    class Decoder(torch.nn.Module):
        dtype=torch.float32
        config=SimpleNamespace(latents_mean=[0.]*4,latents_std=[1.]*4)
        def __init__(self):
            super().__init__();self.layer=torch.nn.ConvTranspose2d(4,4,16,stride=16,bias=False)
        def decode(self,value,return_dict=False):
            return (self.layer(value[:,:,0]).unsqueeze(2),)
    vae=Decoder().eval().requires_grad_(False)
    original=torch.randn(1,16,4);gradients=[]
    for checkpoint in [False,True]:
        packed=original.clone().requires_grad_(True)
        rgb=decode_exact_pixels(vae,packed,64,64,checkpoint=checkpoint)
        assert rgb.shape==(1,3,64,64)
        rgb.square().mean().backward()
        assert packed.grad is not None and torch.isfinite(packed.grad).all() and packed.grad.abs().sum()>0
        gradients.append(packed.grad)
        assert all(p.grad is None and not p.requires_grad for p in vae.parameters())
    torch.testing.assert_close(*gradients)
