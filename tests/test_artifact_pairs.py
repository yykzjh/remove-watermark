import random

import numpy as np
from PIL import Image, ImageDraw

from watermark_studio.artifact_pairs import guard_input, synthetic_case


def scene():
    rng=np.random.default_rng(77)
    clean=Image.fromarray(rng.integers(20,220,(256,320,3),dtype=np.uint8))
    source=clean.copy();ImageDraw.Draw(source).rectangle((10,160,150,248),fill="lightgray")
    mask=Image.new("L",clean.size,0);ImageDraw.Draw(mask).rectangle((10,160,150,248),fill=255)
    return source,clean,mask,{"render":{"panel":[10,160,151,249]}}


def test_synthetic_defects_do_not_modify_clean_targets_or_use_generation():
    source,clean,mask,meta=scene();before=clean.tobytes()
    for label in range(5):
        original,output,box,info=synthetic_case(source,clean,mask,meta,label,891+label)
        assert clean.tobytes()==before and original.tobytes()==source.tobytes()
        assert info["diffusion_generations"]==0 and info["exact_clean_target"]
        if info["label"]:
            assert np.any(np.abs(np.asarray(output,dtype=float)-np.asarray(clean,dtype=float))>2)
        tensor=guard_input(original,output,box,256)
        assert tensor.shape==(7,256,256) and np.isfinite(tensor).all()
        assert set(np.unique(tensor[-1]))<={0,1}


def test_featureless_blur_is_not_falsely_labeled_a_visible_blur_defect():
    clean=Image.new("RGB",(320,256),"gray")
    source=clean.copy();mask=Image.new("L",clean.size,0)
    _,output,_,info=synthetic_case(source,clean,mask,{},2,random.Random(1).randrange(2**32))
    assert info["label"]==1 and output.tobytes()!=clean.tobytes()


def test_visible_content_damage_is_outside_actual_watermark_support():
    source,clean,mask,meta=scene()
    _,output,_,info=synthetic_case(source,clean,mask,meta,4,1882)
    changed=np.any(np.asarray(output)!=np.asarray(clean),axis=2)
    assert info["label"]==4 and changed.any()
    assert not (changed & (np.asarray(mask)>0)).any()


def test_preserved_scene_marking_augmentation_never_modifies_original_watermark_pixels():
    source,clean,mask,meta=scene();source_bytes=source.tobytes();clean_bytes=clean.tobytes()
    found=0
    for seed in range(20):
        original,output,_,info=synthetic_case(source,clean,mask,meta,0,seed,preserved_marks=True)
        marking=info["preserved_marking"]
        assert source.tobytes()==source_bytes and clean.tobytes()==clean_bytes
        if marking:
            l,t,r,b=marking["box"]
            assert not np.asarray(mask)[t:b,l:r].any()
            assert np.array_equal(np.asarray(original)[np.asarray(mask)>0],np.asarray(source)[np.asarray(mask)>0])
            if not info["severity"].get("clean_jpeg"):
                assert np.array_equal(np.asarray(original)[t:b,l:r],np.asarray(output)[t:b,l:r])
            found+=1
    assert found>0


def test_paired_guard_checkpoint_round_trip_keeps_seven_channel_layout(tmp_path):
    import pytest
    torch=pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    safe=pytest.importorskip("safetensors.torch")
    from watermark_studio.artifact_pairs import build_guard
    torch.set_num_threads(1);model=build_guard(False).eval()
    value=torch.zeros(1,7,128,128)
    with torch.no_grad():expected=model(value)
    path=tmp_path/"guard.safetensors"
    safe.save_file({k:v.contiguous() for k,v in model.state_dict().items()},str(path))
    restored=build_guard(False).eval();restored.load_state_dict(safe.load_file(str(path)),strict=True)
    with torch.no_grad():actual=restored(value)
    assert actual.shape==(1,5) and torch.isfinite(actual).all()
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
