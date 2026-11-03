import hashlib
import json

import pytest
from PIL import Image

from watermark_studio.quality_presence import (
    branch_weights,
    canonical_answer,
    output_chat,
    parent_splits,
    presence_chats,
)


def test_parent_aliases_and_multiple_crops_cannot_cross_train_validation():
    sha=lambda i: hashlib.sha256(str(i).encode()).hexdigest()
    parents=[{"source_original_sha256":sha(i),"kind":"physical" if i<20 else "text_free"}
             for i in range(40)]
    parents += [{"source_original_sha256":sha(0),"kind":"text_free"},
                {"source_original_sha256":sha(1),"kind":"physical"}]
    split=parent_splits(parents,77)
    assert split==parent_splits(list(reversed(parents)),77)
    assert len(split)==40
    assert sum(v=="validation" for v in split.values())==8


def test_digital_presence_protocol_never_mutates_original_naturalness_instructions_or_pixels():
    image=Image.new("RGB",(400,300),"blue")
    chat=[{"role":"user","content":[{"type":"image","image":image},
                                  {"type":"text","text":"base naturalness protocol"}]}]
    prepared=presence_chats([chat])[0]
    assert prepared[-1]["content"][0]["image"] is image
    assert "base naturalness protocol"==chat[-1]["content"][-1]["text"]
    assert "实物" in prepared[-1]["content"][-1]["text"]


def test_presence_training_layout_only_supplies_result_pixels():
    before,after=Image.new("RGB",(400,300),"red"),Image.new("RGB",(400,300),"blue")
    chat=output_chat(before,after,[],detail_patches=2)
    images=[row["image"] for row in chat[-1]["content"] if row["type"]=="image"]
    assert images and all(image.getpixel((0,0))==(0,0,255) for image in images)


def test_only_grammar_branches_receive_semantic_loss_weight_and_eos_is_mandatory():
    tree={10:{20:{30:{99:{}},31:{99:{}}}}}
    assert branch_weights(tree,[10,20,31,99])==[.2,.2,3.,.2]
    with pytest.raises(ValueError,match="terminal EOS"):
        branch_weights(tree,[10,20,31])
    with pytest.raises(ValueError,match="diverged"):
        branch_weights(tree,[10,20,32,99])


def test_presence_labels_require_real_booleans_and_no_location_for_an_unambiguous_clean_photo():
    assert json.loads(canonical_answer(False,False,"none"))=={
        "overlay_present":False,"uncertain":False,"area":"none"}
    with pytest.raises(ValueError):
        canonical_answer(1,False,"none")
    with pytest.raises(ValueError):
        canonical_answer(False,False,"bottom")


def test_clean_and_marked_pairs_use_the_exact_same_label_independent_view_geometry():
    import numpy as np
    clean=Image.new("RGB",(800,600),"gray")
    marked=clean.copy()
    # A large visual difference must not change the explicitly frozen layout.
    data=np.asarray(marked).copy();data[200:300,100:500]=(0,0,255)
    marked=Image.fromarray(data)
    layout=[{"box":[.1,.2,.7,.8],"max_edge":256},{"box":[.3,.3,.5,.5],"max_edge":128}]
    first=output_chat(marked,clean,[],detail_layout=layout)
    second=output_chat(marked,marked,[],detail_layout=layout)
    texts=lambda chat:[r["text"] for r in chat[-1]["content"] if r["type"]=="text"]
    assert texts(first)==texts(second)
    images=lambda chat:[r["image"].size for r in chat[-1]["content"] if r["type"]=="image"]
    assert images(first)==images(second)


def test_invisible_synthetic_positive_is_rejected_without_defining_a_native_quality_threshold():
    import numpy as np

    from watermark_studio.quality_presence import measurable_overlay
    clean=Image.new("RGB",(300,200),"white")
    assert not measurable_overlay(clean,clean.copy())["usable"]
    pixels=np.asarray(clean).copy();pixels[10:20,20:40]=230
    report=measurable_overlay(clean,Image.fromarray(pixels))
    assert report["usable"] and report["pixels_delta_at_least_3"]==200
    assert "Synthetic" in report["scope"]


def test_presence_specialist_cannot_clear_base_naturalness_or_uncertainty_vetoes():
    from watermark_studio.quality_presence import merge_presence_checks
    base={"watermark_removed":False,"natural":False,"uncertain":True,"area":"bottom_left"}
    result=merge_presence_checks(base,{"overlay_present":False,"uncertain":False,"area":"none"})
    assert result["watermark_removed"] is True
    assert result["natural"] is False and result["uncertain"] is True
    assert result["area"]=="bottom_left"
    assert result["base_watermark_removed"] is False
    assert base["watermark_removed"] is False


def test_rejection_by_source_or_numeric_checks_is_not_reported_as_real_watermark_detection():
    from watermark_studio.quality_presence import raw_presence_metrics
    rows=[{"file":"faint.png","expected_overlay_present":True,"quality":{
        "status":"review","guard_codes":["source_physical_ambiguity"],
        "output_checks":{"presence_predicate":{"overlay_present":False,"uncertain":False,"area":"none"}}}}]
    result=raw_presence_metrics(rows)
    assert result["overlay_false_negatives"]==["faint.png"]
    assert result["overlay_unsafe_clear"]==["faint.png"]
