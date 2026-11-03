"""ZIP CRC verification detects damage that filename and size checks cannot."""
import runpy
import zipfile
from pathlib import Path

module = runpy.run_path(str(Path(__file__).parents[1]/"scripts/verify-baidu-zip-downloads.py"))


def test_valid_zip_checks_every_member_without_extracting(tmp_path):
    archive = tmp_path/"data.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("town/village/a.jpg", b"photograph payload")
        z.writestr("../not-extracted.txt", b"metadata")
    result = module["verify_archive"](archive)
    assert result["ok"] and result["stable"] and result["entries"] == 2
    assert not (tmp_path.parent/"not-extracted.txt").exists()
    assert not (tmp_path/"town").exists()


def test_same_size_payload_bit_flip_is_rejected(tmp_path):
    archive = tmp_path/"data.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("image.jpg", b"some image bytes"*10)
    raw = bytearray(archive.read_bytes())
    offset = 30+len("image.jpg")
    raw[offset] ^= 1
    archive.write_bytes(raw)
    result = module["verify_archive"](archive)
    assert not result["ok"] and result["stable"]


def test_truncated_zip_is_rejected(tmp_path):
    archive = tmp_path/"data.zip"
    archive.write_bytes(b"PK")
    assert not module["verify_archive"](archive)["ok"]
