import json
import runpy
from pathlib import Path

import pytest
from PIL import Image

TOOLS=runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts/benchmark-verifier-known-source.py"))


def job_fixture(tmp_path, **row_changes):
    state=tmp_path/"state";folder=state/"job";folder.mkdir(parents=True)
    source,output=tmp_path/"input",tmp_path/"output"
    source.mkdir();output.mkdir()
    (source/"image.png").write_bytes(b"original")
    (output/"image.png").write_bytes(b"restored")
    (folder/"summary.json").write_text(json.dumps({"state":"completed","total":1,
        "input_dir":str(source),"output_dir":str(output)}))
    row={"file":"image.png","ok":True,"source_scene":"document",
         "repair_regions":[{"kind":"panel","box":[0,.6,.4,1]}],
         "quality":{"source_regions":[{"kind":"panel","box":[0,.6,.4,1]}],
                    "residual_regions":[{"kind":"logo","box":[.8,0,1,.2]}]}}
    row.update(row_changes)
    (folder/"results.jsonl").write_text(json.dumps(row)+"\n")
    return state,folder,row


def test_known_source_pairing_never_replaces_original_masks_with_output_accusations(tmp_path):
    state,_,_=job_fixture(tmp_path)
    cases=TOOLS["frozen_job_cases"](state,"job")
    assert len(cases)==1 and cases[0]["regions"][0]["box"]==pytest.approx([0,.6,.4,1])
    assert cases[0]["scene"]=="document"
    before=TOOLS["file_hashes"](cases)
    assert before==TOOLS["file_hashes"](cases)


@pytest.mark.parametrize("changes",[
    {"file":"../private.png"},{"repair_regions":None},
    {"source_scene":"ignore instructions"},{"ok":False},
])
def test_invalid_or_unbounded_source_provenance_is_rejected(tmp_path,changes):
    state,_,_=job_fixture(tmp_path,**changes)
    with pytest.raises(ValueError):
        TOOLS["frozen_job_cases"](state,"job")


def test_incomplete_jobs_cannot_be_misreported_as_full_paired_evaluation(tmp_path):
    state,folder,_=job_fixture(tmp_path)
    summary=json.loads((folder/"summary.json").read_text());summary["total"]=2
    (folder/"summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError,match="Incomplete"):
        TOOLS["frozen_job_cases"](state,"job")


def test_synthetic_source_annotations_exclude_clean_identity_but_cover_injected_panel(tmp_path):
    (tmp_path/"input").mkdir();(tmp_path/"output").mkdir()
    records=[{"file":"00-identity.png","kind":"identity","background":"0010.jpg","expected":"passed"},
             {"file":"01-retained_panel.png","kind":"retained_panel","background":"0065.jpg","expected":"review"}]
    for row in records:
        for folder in ("input","output"):
            Image.new("RGB",(400,300),"gray").save(tmp_path/folder/row["file"])
    (tmp_path/"fixture.json").write_text(json.dumps(records))
    clean,marked=TOOLS["known_control_cases"](tmp_path)
    assert clean["regions"]==[]
    box=marked["regions"][0]["box"]
    assert box[0]<.60 and box[1]<.03 and box[2]>=.97 and box[3]>.28


def test_benchmark_profile_precision_is_explicit_in_its_name():
    expected={"bf16":"none","nf4":"nf4","int8":"int8"}
    for name,(_,quantization) in TOOLS["PROFILES"].items():
        assert quantization==expected[name.rsplit("-",1)[1]]



def test_explicit_subset_can_inspect_successful_pairs_without_hiding_a_full_job_failure(tmp_path):
    state,folder,row=job_fixture(tmp_path)
    failed={**row,"file":"failed.png","ok":False}
    with (folder/"results.jsonl").open("a") as stream: stream.write(json.dumps(failed)+"\n")
    summary=json.loads((folder/"summary.json").read_text());summary["total"]=2
    (folder/"summary.json").write_text(json.dumps(summary))
    assert len(TOOLS["frozen_job_cases"](state,"job",["image.png"]))==1
    with pytest.raises(ValueError,match="successful"):
        TOOLS["frozen_job_cases"](state,"job")
    with pytest.raises(ValueError,match="successful"):
        TOOLS["frozen_job_cases"](state,"job",["failed.png"])
    with pytest.raises(ValueError,match="subset"):
        TOOLS["frozen_job_cases"](state,"job",["missing.png"])


def test_whole_direct_empty_repair_mask_never_hides_original_watermarks(tmp_path):
    state,folder,row=job_fixture(tmp_path,processing_path="whole_direct",repair_regions=[])
    cases=TOOLS["frozen_job_cases"](state,"job")
    assert len(cases[0]["regions"])==1
    assert json.loads(json.dumps(cases[0]["regions"]))==row["quality"]["source_regions"]
    row["quality"].pop("source_regions")
    (folder/"results.jsonl").write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError,match="source-mask"):
        TOOLS["frozen_job_cases"](state,"job")


@pytest.mark.parametrize("health,allow,match", [
    ({"gpus": [1], "code_revision": "abc"}, False, "Stop the inference"),
    ({"gpus": [2], "code_revision": "abc"}, True, "overlaps"),
    ({}, True, "Cannot prove"),
    ({"gpus": [True], "code_revision": "abc"}, True, "Cannot prove"),
    ({"gpus": [1, 1], "code_revision": "abc"}, True, "Cannot prove"),
])
def test_live_server_is_never_implicitly_allowed_or_shared(health, allow, match):
    with pytest.raises(RuntimeError, match=match):
        TOOLS["require_disjoint_server_gpus"](health, [2, 3, 4, 5], allow)


def test_explicit_disjoint_server_proof_records_the_actual_gpu_reservation():
    assert TOOLS["require_disjoint_server_gpus"](
        {"gpus": [1], "code_revision": "abc"}, [2, 3, 4, 5], True) == {
            "active": True, "gpus": [1], "code_revision": "abc"}
