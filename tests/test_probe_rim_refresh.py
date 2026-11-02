import hashlib
import json

import pytest
from PIL import Image, ImageDraw

from watermark_studio.config import Region
from watermark_studio.probe_conditioning import fresh_rim_reports, refresh_verified_rims
from watermark_studio.source_panel_rims import propose_blue_card_rims


def fixture(tmp_path):
    root=tmp_path/"inputs";root.mkdir()
    source=Image.new("RGB",(800,600),(80,110,60))
    draw=ImageDraw.Draw(source);draw.rectangle((20,400,350,459),fill=(30,110,220))
    draw.rectangle((20,460,350,589),fill=(220,220,220))
    path=root/"one.png";source.save(path)
    mask=Region(kind="panel",box=(0,.59,.49,1))
    old=propose_blue_card_rims(source,[mask],[{"mask":list(mask.box),"core":[0,.62,.46,.99]}])[0]
    yes={"digital_card":True,"rim_clear":True,"uncertain":False,"area":"none"}
    old|={"approved":True,"editor":yes,"independent":yes,"weights_relation":"different_weights",
        "ocr":{"status":"clear"},"source_rim_sha256":hashlib.sha256(source.crop(tuple(old["restore_pixels"])).tobytes()).hexdigest()}
    fresh=old|{"source_rgb_sha256":hashlib.sha256(source.tobytes()).hexdigest()}
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    data={"mode":"top","sources_unchanged":True,"false_approvals":0,"revision":"abc1234",
        "weight_identity":{"relation":"different_weights","editor_sha256":"a"*64,"verifier_sha256":"b"*64},
        "rows":[{"source":str(path),"source_file_sha256":sha,"proof":fresh}]}
    report=tmp_path/"report.json";report.write_text(json.dumps(data))
    return source,root,old,report,sha,data


def test_fresh_proof_reuses_only_identical_frozen_geometry_and_keeps_legacy_immutable(tmp_path):
    source,root,old,report,sha,_data=fixture(tmp_path)
    cached=fresh_rim_reports([report],root,{"one.png":sha})
    updated,traces=refresh_verified_rims(source,[old],cached["one.png"])
    assert "source_rgb_sha256" not in old
    assert updated[0]["source_rgb_sha256"]==hashlib.sha256(source.tobytes()).hexdigest()
    assert len(traces)==1 and traces[0]["report_sha256"]==hashlib.sha256(report.read_bytes()).hexdigest()


def test_changed_source_or_geometry_cannot_gain_a_fresh_body_permission(tmp_path):
    source,root,old,report,sha,_data=fixture(tmp_path)
    cached=fresh_rim_reports([report],root,{"one.png":sha})["one.png"]
    changed=source.copy();changed.putpixel((700,100),(1,2,3))
    with pytest.raises(ValueError,match="whole original"):
        refresh_verified_rims(changed,[old],cached)
    cached[0]["proof"]["guard_region"]["box"][1]+=.01
    with pytest.raises(ValueError,match="geometry"):
        refresh_verified_rims(source,[old],cached)


def test_known_negative_approvals_or_source_file_changes_block_receipt_reuse(tmp_path):
    _source,root,_old,report,sha,data=fixture(tmp_path)
    with pytest.raises(ValueError,match="file bytes"):
        fresh_rim_reports([report],root,{"one.png":"c"*64})
    data["false_approvals"]=1;report.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="zero negative"):
        fresh_rim_reports([report],root,{"one.png":sha})
