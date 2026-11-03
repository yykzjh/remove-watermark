"""Only a whole object of the frozen size from the provider can supply a checksum."""
import base64
import runpy
from pathlib import Path

import pytest

m = runpy.run_path(str(Path(__file__).parents[1]/"scripts/verify-baidu-storage-checksums.py"))
HELLO = "5d41402abc4b2a76b9719d911017c592"


def test_whole_object_hex_and_base64_headers():
    assert m["storage_reference"]({"Content-Length": "5", "Content-MD5": HELLO.upper()}, 5) == HELLO
    encoded = base64.b64encode(bytes.fromhex(HELLO)).decode()
    assert m["storage_reference"]({"Content-Length": "5", "Content-MD5": encoded}, 5) == HELLO


@pytest.mark.parametrize("headers", [
    {"Content-Length": "4", "Content-MD5": HELLO},
    {"Content-Length": "5", "ETag": HELLO},
    {"Content-Length": "5", "Content-MD5": "bad"},
])
def test_partial_or_missing_checksum_never_approves(headers):
    with pytest.raises(ValueError):
        m["storage_reference"](headers, 5)


@pytest.mark.parametrize("url", [
    "https://baidu.com.evil.test/file", "https://evilbaidu.com/file",
    "http://evil.test/file", "https://user:secret@d.pcs.baidu.com/file",
])
def test_foreign_or_embedded_credentials_are_rejected(url):
    with pytest.raises(ValueError):
        m["secure_storage_url"](url)


def test_only_https_provider_storage_is_used():
    assert m["secure_storage_url"]("http://d.pcs.baidu.com/file?sign=opaque") == "https://d.pcs.baidu.com/file?sign=opaque"
