import numpy as np
import pytest
from PIL import Image

from watermark_studio.artifact_pairs import boundary_guard_input, guard_input, synthetic_case


def test_all_four_edges_have_the_same_outward_to_inward_orientation_without_resizing_the_frame():
    source=Image.new("RGB",(512,384),(130,130,130))
    candidate=source.copy();candidate.paste((110,110,110),(96,80,416,320))
    old=source.tobytes();after=candidate.tobytes()
    value=boundary_guard_input(source,candidate,(96,80,416,320),256)
    assert value.shape==(7,256,256) and np.isfinite(value).all()
    assert source.tobytes()==old and candidate.tobytes()==after
    for side in range(4):
        band=value[:3,:,side*64:(side+1)*64]
        restored=band*np.array([.229,.224,.225])[:,None,None]+np.array([.485,.456,.406])[:,None,None]
        assert abs(float(restored[:,32:-32,8:20].mean())*255-130)<.01
        assert abs(float(restored[:,32:-32,44:56].mean())*255-110)<.01
        assert np.all(value[6,:,side*64:side*64+32]==0)
        assert np.all(value[6,:,side*64+32:(side+1)*64]==1)


def test_missing_image_border_is_explicitly_unobservable():
    source=Image.new("RGB",(512,384),(130,130,130))
    value=boundary_guard_input(source,source,(0,80,416,320))
    assert np.all(value[:6,:,:64]==0)
    assert np.all(value[6,:,:64]==-1)
    assert np.all(value[6,:,64:96]==0)
    with pytest.raises(ValueError):boundary_guard_input(source,source,(0,float("nan"),100,300))
    with pytest.raises(ValueError):guard_input(source,source,(0,0,100,100),profile="wrong")


def test_mixed_shadow_examples_and_flat_camera_panel_metadata_preserve_exact_source_and_outside_pixels():
    clean=Image.new("RGB",(512,384),(130,150,120));source=clean.copy()
    box=(48,72,260,300);source.paste((200,200,200),box)
    mask=Image.new("L",clean.size,0);mask.paste(255,box)
    seen_additive=False
    for seed in range(16):
        before,after,actual,info=synthetic_case(source,clean,mask,
            {"profile":"camera-v2","panel":list(box)},1,seed,shadow_profile="mixed-v1")
        assert tuple(actual)==box and before.tobytes()==source.tobytes()
        outside=np.ones((384,512),bool);outside[72:300,48:260]=False
        assert np.array_equal(np.asarray(after)[outside],np.asarray(clean)[outside])
        seen_additive|="additive_rgb_offset" in info["severity"]
    assert seen_additive
