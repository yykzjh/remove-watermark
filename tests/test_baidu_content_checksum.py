"""Checksum vectors use synthetic public messages, never user image fingerprints."""
import runpy
from pathlib import Path

import pytest

m = runpy.run_path(str(Path(__file__).parents[1]/"scripts/verify-baidu-file-content.py"))


@pytest.mark.parametrize("encoded,plain", [
    ("8e23f7635tb64136eddb0719602bc477", "d41d8cd98f00b204e9800998ecf8427e"),
    ("bd686f11dkea8dc5113480f530da507e", "5d41402abc4b2a76b9719d911017c592"),
    ("3df10ad71paa9d7729c23a155f3df292", "900150983cd24fb0d6963f7d28e17f72"),
])
def test_reversible_metadata_vectors(encoded, plain):
    assert m["decode_metadata_md5"](encoded) == plain
    assert m["decode_metadata_md5"](encoded.upper()) == plain
    assert m["decode_metadata_md5"](plain) == plain


@pytest.mark.parametrize("value", [None, "", "invalid", "z"*32, "0"*9+"w"+"0"*22])
def test_unknown_metadata_is_never_accepted(value):
    with pytest.raises(ValueError):
        m["decode_metadata_md5"](value)


def test_changed_payload_with_same_size_is_rejected(tmp_path):
    f = tmp_path/"payload"
    f.write_bytes(b"hello")
    reference = "bd686f11dkea8dc5113480f530da507e"
    assert m["verify_payload"](f, reference)["status"] == "verified"
    f.write_bytes(b"HELLO")
    result = m["verify_payload"](f, reference)
    assert result["status"] == "mismatch" and result["stable"]


def test_interrupted_cache_tail_is_removed_before_appending(tmp_path):
    path = tmp_path/"cache.jsonl"
    path.write_bytes(b'{"file":"first"}\n{"fil')
    assert set(m["restore_records"](path, {"first", "second"})) == {"first"}
    with path.open("ab") as f:
        f.write(b'{"file":"second"}\n')
    assert set(m["restore_records"](path, {"first", "second"})) == {"first", "second"}


def test_complete_last_record_without_newline_can_be_resumed(tmp_path):
    path = tmp_path/"cache.jsonl"
    path.write_bytes(b'{"file":"first"}')
    m["restore_records"](path, {"first", "second"})
    with path.open("ab") as f:
        f.write(b'{"file":"second"}\n')
    assert set(m["restore_records"](path, {"first", "second"})) == {"first", "second"}


def test_cache_outside_manifest_is_rejected(tmp_path):
    path = tmp_path/"cache.jsonl"
    path.write_bytes(b'{"file":"unrelated"}\n')
    with pytest.raises(ValueError, match="unexpected file"):
        m["restore_records"](path, {"first"})
