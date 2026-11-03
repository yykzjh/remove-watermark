"""The original share and cloud copy must match paths, sizes and available hashes."""
import runpy
from pathlib import Path

import pytest

compare = runpy.run_path(str(Path(__file__).parents[1]/"scripts/verify-baidu-share-copy.py"))["compare_entries"]


def test_normalized_share_namespaces_preserve_deep_structure_and_hashes():
    shared = [{"path": "/sharelink123/photos/town/village/road/a.jpg", "size": 4, "md5": "same"}]
    own = [{"path": "/photos/town/village/road/a.jpg", "size": 4, "md5": "same"}]
    report = compare(shared, own, "/sharelink123/photos", "/photos")
    assert report["files_match"] and report["md5_comparisons"] == 1
    own[0]["md5"] = "different"
    assert compare(shared, own, "/sharelink123/photos", "/photos")["md5_mismatches"] == ["town/village/road/a.jpg"]


def test_missing_extra_and_wrong_size_are_not_accepted():
    share = [{"path": "/s/a", "size": 3}, {"path": "/s/missing", "size": 2}]
    own = [{"path": "/c/a", "size": 4}, {"path": "/c/extra", "size": 5}]
    report = compare(share, own, "/s", "/c")
    assert not report["files_match"]
    assert report["missing_from_cloud"] == ["missing"] and report["extra_in_cloud"] == ["extra"]
    assert report["size_mismatches"] == ["a"]


def test_duplicate_paths_are_rejected():
    rows = [{"path": "/s/a", "size": 1}]*2
    with pytest.raises(ValueError, match="Duplicate"):
        compare(rows, [], "/s", "/c")

normalize = runpy.run_path(str(Path(__file__).parents[1]/"scripts/verify-baidu-share-copy.py"))["normalize_share_path"]


def test_zero_owner_namespace_preserves_same_share_root_and_all_levels():
    root = "/sharelink123-456/photos"
    assert normalize("/sharelink0-456/photos/town/village/a.jpg", root) == root+"/town/village/a.jpg"
    assert normalize(root+"/empty", root) == root+"/empty"


@pytest.mark.parametrize("path", [
    "/sharelink123-999/photos/a.jpg", "/sharelink999-456/photos/a.jpg",
    "/sharelink0-456/different/a.jpg", "/sharelink0-456/photos/../outside.jpg",
])
def test_other_share_or_path_escape_is_rejected(path):
    with pytest.raises(ValueError):
        normalize(path, "/sharelink123-456/photos")
