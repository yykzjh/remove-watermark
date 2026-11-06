import hashlib
import json
import zipfile

import pytest

from watermark_studio.production_archives import build_archives


def prepare(tmp_path):
    output, state = tmp_path/"output", tmp_path/"controller"
    (output/"town/village").mkdir(parents=True)
    (output/"empty/deep").mkdir(parents=True)
    state.mkdir()
    (output/"town/village/photo.jpg").write_bytes(b"accepted image bytes")
    (output/"town/village/other.jpg").write_bytes(b"stale foreign image must not be packed")
    (output/"notes.txt").write_bytes(b"original document")
    (output/"empty.jpg").write_bytes(b"")
    row = {"file": "town/village/photo.jpg", "quality_status": "review",
           "quality_attempts": 3, "size": 20,
           "sha256": hashlib.sha256((output/"town/village/photo.jpg").read_bytes()).hexdigest()}
    (state/"one-results.json").write_text(json.dumps({"rows": [row]}))
    (state/"one-committed.json").write_text("{}")
    ownership = {"manifest_sha256": "input-sha",
                 "units": {"one": {"node": "cdu_dev1", "prefix": "town/village",
                                  "files": ["town/village/photo.jpg"]},
                           "two": {"node": "cdu_dev2", "prefix": "town/village",
                                   "files": ["town/village/other.jpg"]}},
                 "directories": [".", "town", "town/village", "empty", "empty/deep"],
                 "auxiliary": [{"path": p, "md5": hashlib.md5((output/p).read_bytes()).hexdigest()}
                               for p in ["notes.txt", "empty.jpg"]]}
    return output, state, ownership


def test_parts_preserve_names_empty_dirs_auxiliary_and_exclude_stale_ownership(tmp_path):
    output, state, ownership = prepare(tmp_path)
    archive_dir = tmp_path/"archives"
    index = build_archives(output, state, ownership, {"files": []}, "cdu_dev1", archive_dir,
                           target_bytes=21)
    assert index["files"] == 3 and index["image_files"] == 1
    names = set()
    for part in index["artifacts"]:
        with zipfile.ZipFile(part["local_path"]) as z:
            assert z.testzip() is None
            if part["kind"] == "images":
                names.update(z.namelist())
                assert "town/village/other.jpg" not in z.namelist()
                for info in z.infolist():
                    if not info.is_dir():
                        assert z.read(info.filename) == (output/info.filename).read_bytes()
    assert names == {"town/", "town/village/", "empty/", "empty/deep/",
                     "town/village/photo.jpg", "notes.txt", "empty.jpg"}
    before = [(p["name"], p["signature"]) for p in index["artifacts"]]
    cached = build_archives(output, state, ownership, {"files": []}, "cdu_dev1", archive_dir, 21)
    assert [(p["name"], p["signature"]) for p in cached["artifacts"]] == before


def test_changed_image_bytes_cannot_be_published(tmp_path):
    output, state, ownership = prepare(tmp_path)
    (output/"town/village/photo.jpg").write_bytes(b"unexpected changed bytes")
    with pytest.raises(ValueError, match="accepted evidence"):
        build_archives(output, state, ownership, {"files": []}, "cdu_dev1", tmp_path/"archives")


def test_uncommitted_outputs_cannot_be_archived(tmp_path):
    output, state, ownership = prepare(tmp_path)
    (state/"one-committed.json").unlink()
    with pytest.raises(ValueError, match="committed"):
        build_archives(output, state, ownership, {"files": []}, "cdu_dev1", tmp_path/"archives")
