import hashlib
import io
import runpy
from pathlib import Path

import pytest

module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/download-quality-model.py"))


def test_truncated_asset_is_retried_and_verified(monkeypatch):
    responses = iter([b"bad", b"valid"])
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(next(responses)))
    monkeypatch.setattr("time.sleep", lambda _: None)
    expected = hashlib.sha256(b"valid").hexdigest()
    assert module["download_small"]("https://official/file", 5, expected, attempts=2) == b"valid"


def test_wrong_hash_is_never_accepted(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(b"wrong"))
    monkeypatch.setattr("time.sleep", lambda _: None)
    with pytest.raises(ValueError, match="SHA256"):
        module["download_small"]("https://official/file", 5,
                                 hashlib.sha256(b"valid").hexdigest(), attempts=2)
