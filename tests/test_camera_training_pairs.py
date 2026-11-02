import random

import numpy as np
import pytest
from PIL import Image, ImageFont

from watermark_studio.training_pairs import render_pair, render_profile_pair


def test_versioned_camera_overlays_replay_exact_masks_and_do_not_change_legacy(monkeypatch):
    fallback=ImageFont.load_default()
    monkeypatch.setattr("watermark_studio.training_pairs.ImageFont.truetype",lambda *a,**k:fallback)
    clean=Image.fromarray(np.random.default_rng(5).integers(0,256,(448,576,3),dtype=np.uint8))
    before=clean.tobytes()
    for variant in range(4):
        for seed in (7,19,2026):
            image,mask,meta,layer=render_profile_pair(clean,variant,random.Random(seed),"unused",
                profile="camera-v2",return_overlay=True)
            assert clean.tobytes()==before
            assert image.tobytes()==Image.alpha_composite(clean.convert("RGBA"),layer).convert("RGB").tobytes()
            changed=np.any(np.asarray(image)!=np.asarray(clean),axis=2)
            assert not np.any(changed&(np.asarray(mask)==0))
            assert meta["profile"]=="camera-v2"
            if variant==3:assert not changed.any()
            if variant==0:assert meta["header_height"]>0 and meta["rows"]>=4
        old=render_pair(clean,variant,random.Random(7),"unused")[0]
        dispatched=render_profile_pair(clean,variant,random.Random(7),"unused")[0]
        assert old.tobytes()==dispatched.tobytes()
    with pytest.raises(ValueError):
        render_profile_pair(clean,0,random.Random(7),"unused",profile="unknown")


def test_rerender_keeps_exclusion_manifest_and_immutable_targets_inside_derived_root(tmp_path,monkeypatch):
    import hashlib
    import json
    import runpy
    import sys
    from pathlib import Path

    from watermark_studio.training_features import digest, validate_pairs

    fallback=ImageFont.load_default()
    monkeypatch.setattr("watermark_studio.training_pairs.ImageFont.truetype",lambda *a,**k:fallback)
    font=tmp_path/"font";font.write_bytes(b"fixed-font-artifact")
    parent=tmp_path/"parent";parent.mkdir();rows=[]
    for index,split in enumerate(["train","validation"]):
        photo=Image.new("RGB",(160,128),(90+index,120,60))
        target=parent/(str(index)+".png");photo.save(target)
        mask=parent/(str(index)+"mask.png");Image.new("L",photo.size,0).save(mask)
        rows.append({"id":str(index)+"-3","split":split,"source_original":str(target),
            "source_original_sha256":digest(target),"crop":[0,0,160,128],
            "input":str(target),"input_sha256":digest(target),"target":str(target),"target_sha256":digest(target),
            "mask":str(mask),"mask_sha256":digest(mask),"render":{"kind":"identity"},"seed":7})
    forbidden=json.dumps(["f"*64]).encode()
    (parent/"forbidden-sha256.json").write_bytes(forbidden)
    manifest=parent/"manifest.json"
    manifest.write_text(json.dumps({"version":1,"state":"completed","source_overlap":0,
        "previous_exposure_overlap":0,"train_originals":1,"validation_originals":1,
        "font_sha256":digest(font),"pairs":rows}))
    before=manifest.read_bytes();derived=tmp_path/"derived"
    script=Path(__file__).resolve().parents[1]/"scripts/rerender-lora-pilot.py"
    monkeypatch.setattr(sys,"argv",[str(script),"--manifest",str(manifest),"--target",str(derived),
        "--font",str(font),"--render-profile","camera-v2"])
    runpy.run_path(str(script),run_name="__main__")
    data,_=validate_pairs(derived/"manifest.json")
    assert manifest.read_bytes()==before
    assert (derived/"forbidden-sha256.json").read_bytes()==forbidden
    assert data["parent_manifest_sha256"]==hashlib.sha256(before).hexdigest()
    assert len(data["pairs"])==8
    for row in data["pairs"]:
        assert Path(row["target"]).is_relative_to(derived)
        assert row["source_original_sha256"] in {r["source_original_sha256"] for r in rows if r["split"]==row["split"]}
        assert digest(row["target"])==row["target_sha256"]
