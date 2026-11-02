import json

import pytest

from watermark_studio.training_features import digest, validate_pairs


def test_frozen_training_pairs_cannot_leak_or_change_bytes(tmp_path):
    original=tmp_path/"input.png";original.write_bytes(b"known exact input")
    clean=tmp_path/"target.png";clean.write_bytes(b"known exact clean")
    mask=tmp_path/"mask.png";mask.write_bytes(b"known mask")
    rows=[]
    for split,sha in (("train","a"*64),("validation","b"*64)):
        row={"id":split,"split":split,"source_original_sha256":sha}
        for key,path in (("input",original),("target",clean),("mask",mask)):
            row[key]=str(path);row[key+"_sha256"]=digest(path)
        rows.append(row)
    forbidden=tmp_path/"forbidden-sha256.json";forbidden.write_text(json.dumps(["c"*64]))
    manifest=tmp_path/"manifest.json"
    def save():manifest.write_text(json.dumps({"version":1,"state":"completed","pairs":rows}))
    save()
    assert len(validate_pairs(manifest)[1])==3
    rows[1]["source_original_sha256"]="a"*64;save()
    with pytest.raises(ValueError,match="separation"):validate_pairs(manifest)
    rows[1]["source_original_sha256"]="b"*64;save()
    forbidden.write_text(json.dumps(["b"*64]))
    with pytest.raises(ValueError,match="separation"):validate_pairs(manifest)
    forbidden.write_text("[]");original.write_bytes(b"changed")
    with pytest.raises(ValueError,match="bytes"):validate_pairs(manifest)


def test_editing_layout_keeps_both_full_frame_image_grids():
    torch=pytest.importorskip("torch")
    from watermark_studio.training_features import editing_kwargs
    values={"clean_latents":torch.zeros(1,16,64),"source_latents":torch.zeros(1,16,64),
        "prompt_embeds":torch.zeros(1,11,32),"prompt_attention_mask":torch.ones(1,11,dtype=torch.bool),
        "image_pad_mask":torch.zeros(1,15,dtype=torch.bool)}
    result=editing_kwargs(values,64,64)
    assert result["img_shapes"]==[[(1,4,4),(1,4,4)]]
    assert result["encoder_hidden_states_mask"] is None
    values["image_pad_mask"]=torch.zeros(1,14,dtype=torch.bool)
    with pytest.raises(ValueError,match="layout"):editing_kwargs(values,64,64)


def test_editing_batch_rejects_cross_image_prefix_layout_mismatch():
    torch=pytest.importorskip("torch")
    from watermark_studio.training_features import editing_kwargs
    values={"clean_latents":torch.zeros(2,16,64),"source_latents":torch.zeros(2,16,64),
        "prompt_embeds":torch.zeros(2,11,32),"prompt_attention_mask":torch.ones(2,11,dtype=torch.bool),
        "image_pad_mask":torch.zeros(2,15,dtype=torch.bool)}
    assert len(editing_kwargs(values,64,64)["img_shapes"])==2
    values["image_pad_mask"][1,0]=True
    with pytest.raises(ValueError,match="layout"):editing_kwargs(values,64,64)
