import runpy
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from watermark_studio.config import Region
from watermark_studio.denoising_context import pixel_edit_masks

prepare=runpy.run_path(str(Path(__file__).parents[1]/"scripts/roi_context_probe.py"))["prepare_roi_context"]


def item():
    return {"state":{"whole_generation":False,"source":Image.new("RGB",(400,300),"gray"),
        "task":{"relative":"town/village/photo.png"},
        "regions":[Region(box=(0,.6,.4,1)),Region(box=(.5,.8,.9,1))]},
        "box":(0,180,160,300),"crop":(0,100,300,300),
        "image":Image.new("RGB",(300,200),"gray"),"index":0,"prompt":"unchanged","seed":42}


def test_crop_mapping_matches_the_exact_native_source_mask_and_keeps_all_other_inputs():
    edit=item();original=edit["state"]["source"].tobytes()
    inputs=(edit["box"],edit["crop"],edit["prompt"],edit["seed"],edit["image"].tobytes())
    audit=prepare([edit],4)
    source=pixel_edit_masks([{"_denoising_context_regions":[r.model_dump() for r in edit["state"]["regions"]]}],(400,300))[0]
    mapped=pixel_edit_masks([edit],(300,200))[0]
    assert np.array_equal(mapped,source[100:300,:300])
    assert audit[0]["reference_size"]==[300,200] and audit[0]["context_margin_cells"]==4
    assert (edit["box"],edit["crop"],edit["prompt"],edit["seed"],edit["image"].tobytes())==inputs
    assert edit["state"]["source"].tobytes()==original


def test_unsafe_footprint_cannot_silently_restore_watermark_pixels():
    edit=item();edit["box"]=(0,180,200,300)
    with pytest.raises(ValueError,match="unproven repair"):
        prepare([edit],4)
    edit=item();edit["image"]=Image.new("RGB",(100,100))
    with pytest.raises(ValueError,match="geometry"):
        prepare([edit],4)


def test_independent_sources_are_not_broadcast_and_whole_generation_is_left_alone():
    first=item();second=item()
    second["state"]["regions"]=[Region(box=(.5,.8,.9,1))]
    second["box"]=(200,240,360,300);second["crop"]=(100,100,400,300)
    prepare([first,second],1)
    masks=pixel_edit_masks([first,second],(300,200))
    assert not np.array_equal(masks[0],masks[1])
    whole=item();whole["state"]["whole_generation"]=True
    previous=deepcopy(whole["state"]["regions"])
    assert prepare([whole],1)==[] and "_denoising_context_regions" not in whole
    assert whole["state"]["regions"]==previous
