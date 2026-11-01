import json

import pytest

from watermark_studio.probe_conditioning import frozen_source_captions, observed_scene_prompt


def fixture(tmp_path):
    job="a"*32;folder=tmp_path/job;folder.mkdir()
    summary={"state":"completed","total":1,"sources_sha256":{"one.jpg":"b"*64}}
    row={"file":"one.jpg","ok":True,"controlnet_probe":{"source_caption":"A rutted dirt road borders a channel."}}
    (folder/"summary.json").write_text(json.dumps(summary))
    (folder/"results.jsonl").write_text(json.dumps(row)+"\n")
    return job,folder,summary,row


def test_original_caption_is_conditioning_only_and_preserves_removal_prompt(tmp_path):
    job,_folder,_summary,_row=fixture(tmp_path)
    observation=frozen_source_captions(tmp_path,job,{"one.jpg":"b"*64})["one.jpg"]
    assert observation["quality_permission"] is False
    assert observation["source_file_sha256"]=="b"*64
    prompt=observed_scene_prompt("Remove digital overlays. Preserve real text.",observation)
    assert prompt.startswith("Remove digital overlays. Preserve real text.")
    assert "rutted dirt road" in prompt and "not a quality verdict" in prompt


def test_caption_cannot_be_used_for_different_original_bytes(tmp_path):
    job,_folder,_summary,_row=fixture(tmp_path)
    with pytest.raises(ValueError,match="same original"):
        frozen_source_captions(tmp_path,job,{"one.jpg":"c"*64})
    with pytest.raises(ValueError,match="Explicit"):
        frozen_source_captions(tmp_path,"../elsewhere",{})


@pytest.mark.parametrize("failure",["duplicate","missing","failed"])
def test_unusable_caption_is_never_replaced_by_generic_guess(tmp_path,failure):
    job,folder,summary,row=fixture(tmp_path)
    if failure=="duplicate":
        summary["total"]=2
        (folder/"results.jsonl").write_text((json.dumps(row)+"\n")*2)
    else:
        row["ok"]=failure!="failed"
        row["controlnet_probe"]={}
        (folder/"results.jsonl").write_text(json.dumps(row)+"\n")
    (folder/"summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        frozen_source_captions(tmp_path,job,{"one.jpg":"b"*64})
