from copy import deepcopy

import numpy as np
import pytest
from PIL import Image

from watermark_studio.config import Region
from watermark_studio.full_frame_sampling import (
    full_frame_mapping,
    map_region,
    restore_full_frame,
    sample_full_frame,
    validate_mapping,
)


def test_full_frame_budget_preserves_every_border_and_positive_bounded_density():
    r=Region(kind="panel",box=(.02,.70,.42,.98))
    mapping=full_frame_mapping([r],(4000,3000),.30)
    validate_mapping(mapping)
    for name in ("x","y"):
        a=mapping[name]
        assert a["source"][0]==a["model"][0]==0
        assert a["source"][-1]==a["model"][-1]==1
        slopes=np.diff(a["model"])/np.diff(a["source"])
        assert slopes.min()>=.65-1e-12 and slopes.max()<=2+1e-12
    warped=map_region(r,mapping)
    assert (warped.box[2]-warped.box[0])*(warped.box[3]-warped.box[1])>2*(.4*.28)
    assert not mapping["cropped_diffusion_input"]


def test_mapping_restores_smooth_geometry_and_corner_content_without_crop():
    yy,xx=np.mgrid[:300,:400]
    original=Image.fromarray(np.stack([xx/2,yy/2,(xx+yy)/4],axis=-1).astype(np.uint8))
    mapping=full_frame_mapping([Region(kind="panel",box=(.02,.7,.42,.98))],original.size,.30)
    warped=sample_full_frame(original,mapping)
    restored=restore_full_frame(warped,original.size,mapping)
    assert warped.size==restored.size==original.size
    delta=np.abs(np.asarray(restored).astype(float)-np.asarray(original))
    assert delta.mean()<1.2
    assert max(delta[0,0].max(),delta[-1,-1].max(),delta[0,-1].max(),delta[-1,0].max())<=3
    # Coordinates remain monotone and recover their original positions exactly.
    for name in ("x","y"):
        a=mapping[name];points=np.linspace(0,1,101)
        inverse=np.interp(np.interp(points,a["source"],a["model"]),a["model"],a["source"])
        assert np.allclose(points,inverse,atol=1e-12)


def test_disabled_or_missing_large_overlay_keeps_existing_native_path_bit_exact():
    original=Image.fromarray(np.arange(12*16*3,dtype=np.uint8).reshape(12,16,3))
    # Small standalone app code never triggers a large-panel reallocation.
    assert full_frame_mapping([Region(kind="logo",box=(.8,.9,1,1))],(4000,3000),.30) is None
    assert full_frame_mapping([],original.size if original.height>=16 else (16,16),0) is None
    assert sample_full_frame(original,None) is original
    expected=original.resize((32,24),Image.Resampling.LANCZOS)
    assert restore_full_frame(original,(32,24),None).tobytes()==expected.tobytes()


def test_changed_or_incomplete_sampling_record_cannot_silently_reproject():
    mapping=full_frame_mapping([Region(kind="panel",box=(.1,.6,.5,.9))],(400,300),.30)
    damaged=deepcopy(mapping);damaged["x"]["model"][1]+=.01
    with pytest.raises(ValueError):
        validate_mapping(damaged)
    with pytest.raises(ValueError):
        sample_full_frame(Image.new("RGB",(200,150)),mapping)
    with pytest.raises(ValueError):
        restore_full_frame(Image.new("RGB",(400,300)),(200,150),mapping)
    for value in (-.1,.36,float("nan")):
        with pytest.raises(ValueError):
            full_frame_mapping([], (400,300),value)


def test_mapped_automatic_mask_matches_the_actual_reprojected_source_pixels():
    from PIL import ImageDraw
    region=Region(kind="panel",box=(.18,.58,.45,.86))
    source=Image.new("RGB",(400,300),"white")
    ImageDraw.Draw(source).rectangle((72,174,179,257),fill=(200,0,0))
    mapping=full_frame_mapping([region],source.size,.30)
    warped=sample_full_frame(source,mapping)
    rgb=np.asarray(warped)
    yy,xx=np.where((rgb[...,0]>100)&(rgb[...,1]<50)&(rgb[...,2]<50))
    observed=np.array([xx.min()/400,yy.min()/300,(xx.max()+1)/400,(yy.max()+1)/300])
    expected=np.asarray(map_region(region,mapping).box)
    assert np.max(np.abs(observed-expected))<.006
