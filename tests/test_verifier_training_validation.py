import hashlib
import json
import runpy
from pathlib import Path

import pytest
from PIL import Image


def helper():
    return runpy.run_path(str(Path(__file__).parents[1]/"scripts/train-verifier-presence-lora.py"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    photos=[]
    for index,color in enumerate(("red","blue")):
        file=tmp_path/f"{index}.png";Image.new("RGB",(64,64),color).save(file)
        photos.append(file)
    forbidden=tmp_path/"forbidden.json";forbidden.write_text(json.dumps(["f"*64]))
    rows=[]
    for index in range(100):
        group=int(index>=80);file=photos[group]
        rows.append({"id":str(index),"source_original_sha256":str(group)*64,
                     "split":"train" if group==0 else "validation",
                     "before":str(file),"before_sha256":digest(file),
                     "after":str(file),"after_sha256":digest(file),
                     "overlay_present":False,"uncertain":False,"area":"none"})
    data={"version":1,"state":"completed","task":"output_only_digital_overlay_presence",
          "forbidden_manifest":str(forbidden),"forbidden_manifest_sha256":digest(forbidden),"rows":rows}
    path=tmp_path/"manifest.json";path.write_text(json.dumps(data))
    return path,data,photos


def test_original_parent_alias_leak_is_rejected_before_gpu_allocation(tmp_path):
    path,data,_=fixture(tmp_path)
    data["rows"][-1]["source_original_sha256"]="0"*64
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="original photo"):
        helper()["validate_manifest"](path)


def test_identical_clean_pixels_cannot_cross_train_validation_even_with_different_parent_names(tmp_path):
    path,data,photos=fixture(tmp_path)
    for row in data["rows"]:
        for field in ("before","after"):
            row[field]=str(photos[0]);row[field+"_sha256"]=digest(photos[0])
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="Identical clean"):
        helper()["validate_manifest"](path)


def test_corrupted_pixels_or_source_exclusion_snapshot_fail_before_training(tmp_path):
    path,_,photos=fixture(tmp_path)
    photos[0].write_bytes(b"changed")
    with pytest.raises(ValueError,match="Frozen image"):
        helper()["validate_manifest"](path)


def test_exact_label_types_are_required(tmp_path):
    path,data,_=fixture(tmp_path)
    data["rows"][0]["overlay_present"]="false"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="boolean"):
        helper()["validate_manifest"](path)
