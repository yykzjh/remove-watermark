import hashlib
from pathlib import Path

import pytest
from PIL import Image

from watermark_studio.early_review import create_packet, unpack_packet
from watermark_studio.production_archives import file_sha256


def packet(tmp_path):
    output = tmp_path / "output"
    (output / "town/village").mkdir(parents=True)
    rows, sources = [], {}
    for name, status in [("town/village/review.png", "review"), ("town/village/passed.png", "passed")]:
        p = output / name
        Image.new("RGB", (32, 48), "green").save(p)
        rows.append({"file": name, "quality_status": status, "size": p.stat().st_size,
                     "sha256": file_sha256(p), "original_size": [32, 48], "regions": []})
        sources[name] = {"path": name, "size": p.stat().st_size,
                         "md5": hashlib.md5(p.read_bytes()).hexdigest()}
    return output, rows, sources


def test_only_committed_rejected_candidates_enter_early_packets(tmp_path):
    output, rows, sources = packet(tmp_path)
    index = create_packet(output, rows, sources, "source-sha", "cdu_dev1", "unit-one", tmp_path / "packet")
    assert index["files"] == 1
    receipt = {"node": "cdu_dev1", "unit": "unit-one", "manifest_sha256": "source-sha"}
    artifact = Path(index["artifacts"][0]["local_path"])
    meta = unpack_packet(artifact, tmp_path / "extracted", receipt)
    assert [r["file"] for r in meta["files"]] == ["town/village/review.png"]
    assert (tmp_path / "extracted/images/town/village/review.png").read_bytes() == (
        output / "town/village/review.png").read_bytes()
    assert not (tmp_path / "extracted/images/town/village/passed.png").exists()
    assert create_packet(output, rows, sources, "source-sha", "cdu_dev1", "unit-one",
                         tmp_path / "packet") == index
    with pytest.raises(ValueError, match="snapshot"):
        unpack_packet(artifact, tmp_path / "extracted", receipt | {"unit": "foreign"})


def test_changed_committed_bytes_are_never_exported(tmp_path):
    output, rows, sources = packet(tmp_path)
    (output / "town/village/review.png").write_bytes(b"unexpected changes")
    with pytest.raises(ValueError, match="evidence"):
        create_packet(output, rows, sources, "sha", "cdu_dev1", "unit-one", tmp_path / "packet")


def test_cached_packet_identity_cannot_be_rebound(tmp_path):
    output, rows, sources = packet(tmp_path)
    create_packet(output, rows, sources, "first-sha", "cdu_dev1", "unit-one", tmp_path / "packet")
    with pytest.raises(ValueError, match="snapshot"):
        create_packet(output, rows, sources, "new-sha", "cdu_dev1", "unit-one", tmp_path / "packet")
