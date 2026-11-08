import hashlib
import json
import zipfile

import pytest

from watermark_studio.final_zip import build_zip


def bundle(tmp_path, *, bad_path=False):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "a.txt").write_bytes(b"keep")
    (left / "empty.jpg").write_bytes(b"")
    (right / "b.jpg").write_bytes(b"old")
    # A stale peer copy must never override the recorded authoritative owner.
    (right / "a.txt").write_bytes(b"stale")
    expected = {
        "a.txt": {"size": 4, "sha256": hashlib.sha256(b"keep").hexdigest(), "source_root": 0},
        "b.jpg": {"size": 7, "sha256": hashlib.sha256(b"revised").hexdigest(), "source_root": 1},
        "empty.jpg": {"size": 0, "sha256": hashlib.sha256(b"").hexdigest(), "source_root": 0},
    }
    def entry(path, size=0, block="", directory=False, crc=""):
        return {"Path": path, "Size": str(size), "Block": str(block),
                "Modified": "2026-11-08 02:46:14", "CRC": crc,
                "Attributes": "D drwxr-xr-x" if directory else "A -rw-r--r--"}
    import zlib
    plan = {"root": "photos", "files": expected, "overrides": ["b.jpg"], "entries": [
        entry("photos", directory=True), entry("photos/空目录", directory=True),
        entry("photos/empty.jpg"),
        entry("photos/a.txt", 4, 0, crc=f"{zlib.crc32(b'keep'):08X}"),
        entry("photos/b.jpg", 7, 1, crc=f"{zlib.crc32(b'revised'):08X}"),
    ]}
    if bad_path:
        plan["entries"][-1]["Path"] = "photos/../escape.jpg"
    target = tmp_path / "delivery.zip"
    with zipfile.ZipFile(target, "w") as z:
        z.writestr("delivery-plan.json", json.dumps(plan, ensure_ascii=False))
        z.writestr("payload/b.jpg", b"revised")
    return target, [left, right], plan


def test_rebuild_preserves_exact_tree_overrides_empty_files_and_bytes(tmp_path):
    patch, roots, plan = bundle(tmp_path)
    destination = tmp_path / "result.zip"
    report = build_zip(patch, roots, destination, tmp_path / "state")
    assert report["completed"] == 3
    with zipfile.ZipFile(destination) as z:
        assert set(z.namelist()) == {"photos/", "photos/空目录/", "photos/a.txt",
                                    "photos/b.jpg", "photos/empty.jpg"}
        assert z.testzip() is None
        assert z.read("photos/a.txt") == b"keep"
        assert z.read("photos/b.jpg") == b"revised"
        assert z.read("photos/empty.jpg") == b""
    # The archive identity remains fixed across separate rebuilds and source locations.
    plan["expected_archive_sha256"] = report["sha256"]
    with zipfile.ZipFile(patch, "w") as z:
        z.writestr("delivery-plan.json", json.dumps(plan, ensure_ascii=False))
        z.writestr("payload/b.jpg", b"revised")
    again = tmp_path / "again.zip"
    next_report = build_zip(patch, roots, again, tmp_path / "next-state")
    assert next_report["identical_to_local_zip"]
    assert again.read_bytes() == destination.read_bytes()


def test_rebuild_rejects_wrong_source_hash_without_publishing(tmp_path):
    patch, roots, _ = bundle(tmp_path)
    (roots[0] / "a.txt").write_bytes(b"evil")
    output = tmp_path / "bad.zip"
    with pytest.raises(ValueError, match="Source bytes"):
        build_zip(patch, roots, output, tmp_path / "state")
    assert not output.exists()


def test_rebuild_rejects_path_traversal_before_writing(tmp_path):
    patch, roots, _ = bundle(tmp_path, bad_path=True)
    with pytest.raises(ValueError, match="Unsafe archive entry"):
        build_zip(patch, roots, tmp_path / "bad.zip", tmp_path / "state")
    assert not (tmp_path / "bad.zip.part").exists()
