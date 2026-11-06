import hashlib
import io
import json
import os
from pathlib import Path

import pytest

from watermark_studio.dataset_transfer import (
    beneath,
    digest,
    frozen_inventory,
    partition_manifest,
    read_token,
    relative_path,
    signature,
    transfer_file,
)


def test_partition_preserves_all_files_and_empty_directories():
    files = [{"path": f"镇{town}/村{village}/深层/照片 {i}.jpg", "size": 100, "md5": "x"}
             for town in range(2) for village in range(4) for i in range(village + 1)]
    files += [{"path": "镇0/村0/记录.mp4", "size": 10, "md5": "x"},
              {"path": "原有空文件.jpg", "size": 0, "md5": "x"}]
    dirs = {"."}
    for row in files:
        dirs.update(p.as_posix() for p in Path(row["path"]).parents)
    dirs.update({"空镇", "空镇/空村", "镇1/村3/深层/空目录"})
    manifest = {"files": files, "directories": sorted(dirs)}
    result = partition_manifest(manifest, {"server1": 4, "server2": 4, "server3": 3})
    flattened = [f["path"] for s in result["shards"].values() for f in s["files"]]
    assert len(flattened) == len(set(flattened)) == len(files)
    assert set(flattened) == {f["path"] for f in files}
    assert set().union(*(set(s["directories"]) for s in result["shards"].values())) == dirs
    assert sum(s["image_count"] for s in result["shards"].values()) == len(files) - 2
    for town in range(2):
        for village in range(4):
            owners = {host for host, s in result["shards"].items()
                      if any(f["path"].startswith(f"镇{town}/村{village}/") for f in s["files"])}
            assert len(owners) == 1


@pytest.mark.parametrize("value", ["/absolute", "../escape", "town/../../escape", "."])
def test_unsafe_paths_are_rejected(value):
    with pytest.raises(ValueError):
        relative_path(value)


def test_link_cannot_escape_dataset(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "escape").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        beneath(root, "escape/outside")


@pytest.mark.parametrize("payload", [b"", b"complete original bytes"])
def test_file_transfer_checks_bytes_and_keeps_temp_files_outside_input(tmp_path, payload):
    class Opener:
        def open(self, request, timeout):
            assert request.get_header("Authorization") == "Bearer " + "x" * 64
            response = io.BytesIO(payload)
            response.headers = {"Content-Length": str(len(payload))}
            return response

    target = tmp_path / "input"
    target.mkdir()
    row = {"path": "镇/村/照片 (2).jpg", "size": len(payload),
           "md5": hashlib.md5(payload).hexdigest()}
    assert transfer_file(target, row, Opener(), "https://source", "x" * 64,
                         tmp_path / "state/partial") == "copied"
    assert (target / row["path"]).read_bytes() == payload
    assert transfer_file(target, row, Opener(), "https://source", "x" * 64,
                         tmp_path / "state/partial") == "reused"
    assert [p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()] == [row["path"]]


def test_corrupt_transfer_does_not_replace_existing_file(tmp_path, monkeypatch):
    class Opener:
        def open(self, *_args, **_kwargs):
            response = io.BytesIO(b"wrong")
            response.headers = {"Content-Length": "5"}
            return response

    monkeypatch.setattr("watermark_studio.dataset_transfer.time.sleep", lambda _: None)
    target = tmp_path / "input"
    target.mkdir()
    (target / "a.jpg").write_bytes(b"prior")
    row = {"path": "a.jpg", "size": 5, "md5": hashlib.md5(b"right").hexdigest()}
    with pytest.raises(ValueError, match="checksum"):
        transfer_file(target, row, Opener(), "https://source", "x" * 64, tmp_path / "partial")
    assert (target / "a.jpg").read_bytes() == b"prior"


def test_frozen_source_must_match_its_verified_signature(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    image = root / "照片.jpg"
    image.write_bytes(b"source")
    state = tmp_path / "state"
    (state / "file-content-validation").mkdir(parents=True)
    (state / "source.json").write_text(json.dumps({"destination": str(root), "cloud_root": "/photos"}))
    (state / "manifest.json").write_text(json.dumps([{"path": "/photos/照片.jpg", "size": 6}]))
    (state / "directories.json").write_text(json.dumps(["/photos"]))
    row = {"file": "照片.jpg", "actual": digest(image), "signature": signature(image)}
    (state / "file-content-validation/results.jsonl").write_text(json.dumps(row) + "\n")
    manifest, _ = frozen_inventory(root, state)
    assert manifest["files"][0]["md5"] == digest(image)
    previous = image.stat()
    image.write_bytes(b"edited")
    os.utime(image, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000_000))
    with pytest.raises(ValueError, match="changed"):
        frozen_inventory(root, state)


def test_transfer_token_is_required(tmp_path):
    token = tmp_path / "token"
    token.write_text("")
    with pytest.raises(ValueError):
        read_token(token)
