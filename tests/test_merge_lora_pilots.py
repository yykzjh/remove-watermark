import json
import runpy
from pathlib import Path

import pytest

from watermark_studio.training_features import digest, validate_pairs

MERGE=runpy.run_path(str(Path(__file__).parents[1]/"scripts/merge-lora-pilots.py"))["merge_pilots"]


def fixture(tmp_path):
    pool=tmp_path/"pool";pool.mkdir();parts=[];manifests=[]
    (pool/"forbidden-sha256.json").write_text(json.dumps(["f"*64]))
    for shard in range(2):
        source_hashes=[str(shard*2+1)*64,str(shard*2+2)*64]
        selector=pool/f"part-{shard}.json"
        selector.write_text(json.dumps({"version":1,"sources":[{"sha256":s} for s in source_hashes]}))
        parts.append(str(selector))
        folder=tmp_path/f"shard{shard}";folder.mkdir()
        # The later shard correctly records previous exposure from the first,
        # while both were reserved jointly before any screening.
        (folder/"forbidden-sha256.json").write_text(json.dumps(["f"*64]+(["1"*64,"2"*64] if shard else [])))
        rows=[]
        for index,split in enumerate(["train","validation"]):
            row={"id":f"{shard}-{index}","split":split,"source_original_sha256":source_hashes[index]}
            for key in ["input","target","mask"]:
                f=folder/f"{key}-{index}.png";f.write_bytes(f"{shard}-{index}-{key}".encode())
                row[key]=str(f);row[key+"_sha256"]=digest(f)
            rows.append(row)
        manifest=folder/"manifest.json"
        manifest.write_text(json.dumps({"version":1,"state":"completed","resolution":768,
            "require_native_pixels":True,"candidate_selector_sha256":digest(selector),"pairs":rows}))
        manifests.append(manifest)
    (pool/"report.json").write_text(json.dumps({"state":"completed","forbidden_sha256":1,"parts":parts}))
    return pool,manifests


def test_joint_merge_retains_source_disjointness_and_pre_screening_forbidden_snapshot(tmp_path):
    pool,manifests=fixture(tmp_path)
    result=MERGE(pool,tmp_path/"merged",manifests)
    assert result["train_originals"]==2 and result["validation_originals"]==2
    data,_=validate_pairs(result["manifest"])
    assert len(data["pairs"])==4
    assert json.loads((tmp_path/"merged/forbidden-sha256.json").read_text())==["f"*64]
    for row in data["pairs"]:
        assert Path(row["input"]).is_relative_to(tmp_path/"merged")
        assert row["split"]==("train" if row["id"].endswith("0") else "validation")


def test_duplicate_shard_and_pool_source_collision_rejected(tmp_path):
    pool,manifests=fixture(tmp_path)
    with pytest.raises(ValueError,match="Independent"):
        MERGE(pool,tmp_path/"bad",[manifests[0],manifests[0]])
    selector=pool/"part-1.json"
    selector.write_text(json.dumps({"version":1,"sources":[{"sha256":"1"*64}]}))
    with pytest.raises(ValueError,match="Disjoint"):
        MERGE(pool,tmp_path/"bad2",manifests)
