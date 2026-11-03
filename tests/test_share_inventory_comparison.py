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
