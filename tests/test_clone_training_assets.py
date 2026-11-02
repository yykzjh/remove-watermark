from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from watermark_studio.training_features import clone_pair_assets, digest


def test_parent_assets_are_rebound_inside_derived_root_without_changing_source_roles_or_bytes(tmp_path):
    parent=tmp_path/"parent";parent.mkdir();source=parent/"rgb.png";mask=parent/"mask.png"
    Image.fromarray(np.random.default_rng(8).integers(0,256,(96,128,3),dtype=np.uint8)).save(source)
    Image.new("L",(128,96),255).save(mask)
    row={"id":"unchanged","split":"validation","source_original":"original",
        "source_original_sha256":"c"*64,"input":str(source),"target":str(source),"mask":str(mask),
        "input_sha256":digest(source),"target_sha256":digest(source),"mask_sha256":digest(mask)}
    frozen=dict(row);out=tmp_path/"derived"
    result=clone_pair_assets([row],out)[0]
    assert row==frozen and result["split"]=="validation" and result["source_original"]=="original"
    for key in ("input","target","mask"):
        assert Path(result[key]).is_relative_to(out)
        assert digest(result[key])==frozen[key+"_sha256"]
    with pytest.raises(ValueError):
        clone_pair_assets([{**row,"mask_sha256":"../bad"}],out)
