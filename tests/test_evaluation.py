import json

import pytest

from watermark_studio.evaluation import exclusion_sources


def write_sources(root, name, paths):
    folder=root/name
    folder.mkdir()
    (folder/"sources.json").write_text(json.dumps(paths))
    return folder


def test_explicit_history_never_drops_original_development_samples(tmp_path):
    base=write_sources(tmp_path,"original64",["/images/old64.jpg"])
    write_sources(tmp_path,"first-pass-holdout-neutral-v1",["/images/old128.jpg"])
    write_sources(tmp_path,"first-pass-prior1024",["/images/other.jpg"])
    seen,automatic=exclusion_sources(tmp_path,base,["first-pass-prior1024"])
    assert seen=={"/images/old64.jpg","/images/old128.jpg","/images/other.jpg"}
    assert automatic==["original64","first-pass-holdout-neutral-v1"]


def test_first_evaluation_without_older_128_is_still_supported(tmp_path):
    base=write_sources(tmp_path,"original64",["/images/old64.jpg"])
    assert exclusion_sources(tmp_path,base,[])[0]=={"/images/old64.jpg"}
    with pytest.raises(ValueError,match="Invalid excluded"):
        exclusion_sources(tmp_path,base,["../outside"])
