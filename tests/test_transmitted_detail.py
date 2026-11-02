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
