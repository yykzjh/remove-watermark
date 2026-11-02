import numpy as np
import pytest
from PIL import Image, ImageFilter

from watermark_studio.transmitted_detail import recover_patch_detail


def photos(foreground=220):
    y,x=np.indices((96,128))
    pattern=110+18*np.sin(x*.6)+14*np.cos(y*.7)
    clean=np.repeat(pattern[...,None],3,axis=2).round().astype(np.uint8)
    source=(clean.astype(float)*.5+foreground*.5).round().astype(np.uint8)
    target=Image.fromarray(clean).filter(ImageFilter.GaussianBlur(3))
    return clean,Image.fromarray(source),target


def test_detail_uses_transmitted_real_structure_without_copying_foreground_color():
    clean,source,target=photos()
    support=np.ones(clean.shape[:2])
    out,audit=recover_patch_detail(source,target,support,transmission=.5,radius=12,max_delta=32)
    before=np.mean((np.asarray(target,dtype=float)-clean)**2)
    after=np.mean((np.asarray(out,dtype=float)-clean)**2)
    assert after<before*.15
    assert audit["quality_permission"] is False and audit["extra_diffusion_inferences"]==0
    _,other,_=photos(240)
    second,_=recover_patch_detail(other,target,support,transmission=.5,radius=12,max_delta=32)
    assert np.abs(np.asarray(second,dtype=int)-np.asarray(out,dtype=int)).max()<=1


def test_excluded_text_never_bleeds_through_masked_filter_or_changes_protected_pixels():
    clean,source,target=photos()
    support=np.ones(clean.shape[:2]);support[20:50,40:70]=0;support[:5]=0
    painted=np.asarray(source).copy();painted[20:50,40:70]=0
    first,_=recover_patch_detail(source,target,support,radius=12)
    second,_=recover_patch_detail(Image.fromarray(painted),target,support,radius=12)
    assert np.array_equal(np.asarray(first),np.asarray(second))
    assert np.array_equal(np.asarray(second)[support==0],np.asarray(target)[support==0])


def test_invalid_assumptions_cannot_silently_apply_recovery():
    clean,source,target=photos();support=np.ones(clean.shape[:2])
    with pytest.raises(ValueError):recover_patch_detail(source,target,support,transmission=.1)
    with pytest.raises(ValueError):recover_patch_detail(source,target,support*float("nan"))

def test_coarse_transmission_estimate_rejects_healthy_and_unrelated_references():
    from PIL import ImageFilter

    from watermark_studio.transmitted_detail import estimate_patch_transmission
    y,x=np.mgrid[:128,:128]
    base=130+35*np.sin(x/128*2*np.pi)+22*np.cos(y/128*2*np.pi)+8*((x+y)%2*2-1)
    clean=np.repeat(base[...,None],3,axis=2)
    photo=Image.fromarray(np.rint(clean).astype(np.uint8))
    source=Image.fromarray(np.rint(.55*clean+.45*235).astype(np.uint8))
    support=np.ones((128,128))
    degraded=photo.filter(ImageFilter.GaussianBlur(2))
    estimate=estimate_patch_transmission(source,degraded,support,radius=16)
    assert estimate["eligible"] and abs(estimate["transmission"]-.55)<.04
    assert estimate["quality_permission"] is False
    healthy=estimate_patch_transmission(source,photo,support,radius=16)
    assert not healthy["eligible"] and healthy["reason"]=="no_observable_detail_deficit"
    unrelated=Image.fromarray(np.rint(255-clean).astype(np.uint8))
    assert not estimate_patch_transmission(source,unrelated,support,radius=16)["eligible"]


def test_attenuation_aware_evidence_compares_background_detail_without_accepting_healthy_targets():
    import io

    from watermark_studio.transmitted_detail import estimate_patch_transmission
    y,x=np.mgrid[:192,:256]
    base=125+35*np.sin(x/256*2*np.pi)+25*np.cos(y/192*2*np.pi)+20*((x+y)%2*2-1)
    clean=np.repeat(base[...,None],3,axis=2)
    photo=Image.fromarray(np.rint(clean).astype(np.uint8))
    source=Image.fromarray(np.rint(.3*clean+.7*235).astype(np.uint8))
    support=np.ones((192,256))
    degraded=photo.filter(ImageFilter.GaussianBlur(.35))
    old=estimate_patch_transmission(source,degraded,support,radius=24)
    new=estimate_patch_transmission(source,degraded,support,radius=24,attenuation_aware=True)
    assert not old["eligible"] and old["reason"]=="no_observable_detail_deficit"
    assert new["eligible"] and new["source_detail_rms"]<new["generated_detail_rms"]
    assert new["quality_permission"] is False and new["source_rgb_uncertainty_floor"]==2
    assert not estimate_patch_transmission(source,photo,support,radius=24,attenuation_aware=True)["eligible"]
    stream=io.BytesIO();photo.save(stream,format="JPEG",quality=95);stream.seek(0)
    with Image.open(stream) as encoded:
        healthy=encoded.convert("RGB")
    assert not estimate_patch_transmission(source,healthy,support,radius=24,attenuation_aware=True)["eligible"]


def test_low_transmission_is_explicit_and_retains_noise_fit_and_pixel_scope_bounds():
    import pytest

    y,x=np.mgrid[:192,:256]
    base=70+.25*x+.18*y+35*np.sin(x*.7)+12*np.cos(y*.5)
    clean=Image.fromarray(np.repeat(np.clip(np.rint(base),0,255).astype(np.uint8)[...,None],3,axis=2))
    original=Image.fromarray(np.rint(.15*np.asarray(clean).astype(float)+.85*220).astype(np.uint8))
    generated=clean.filter(ImageFilter.GaussianBlur(2))
    support=np.ones((192,256));support[:10]=0;support[-10:]=0
    from watermark_studio.transmitted_detail import estimate_patch_transmission
    default=estimate_patch_transmission(original,generated,support,attenuation_aware=True)
    assert not default["eligible"] and default["transmission_lower_bound"]==.25
    trial=estimate_patch_transmission(original,generated,support,attenuation_aware=True,
        low_transmission_experiment=True)
    assert trial["eligible"] and .12<trial["transmission"]<.18
    assert trial["source_rgb_uncertainty_floor"]==2 and trial["quality_permission"] is False
    with pytest.raises(ValueError,match="bounded"):
        recover_patch_detail(original,generated,support,transmission=.15)
    after,audit=recover_patch_detail(original,generated,support,transmission=trial["transmission"],
        low_transmission_experiment=True)
    before=np.asarray(generated).astype(float);actual=np.asarray(after).astype(float)
    assert np.array_equal(actual[support==0],before[support==0])
    assert np.max(np.abs(actual-before))<=24.5
    target=np.asarray(clean).astype(float)
    assert np.square(actual-target).mean()<np.square(before-target).mean()
    assert audit["quality_permission"] is False
