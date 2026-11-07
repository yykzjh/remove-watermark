import hashlib
import json
import threading
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


def test_one_pool_refills_across_packets_and_checks_scope(tmp_path, monkeypatch):
    from watermark_studio.codex_pool import photo_map
    from watermark_studio.early_review import prepared_deliveries

    state = tmp_path / "early"
    state.mkdir()
    source = tmp_path / "source.png"
    Image.new("RGB", (32, 48), "green").save(source)
    sha = file_sha256(source)
    groups = {"one": ["slow.png"], "two": ["a.png", "b.png", "c.png", "single.png", "passed.png"]}
    indexes = {}
    for token in groups:
        path = tmp_path / (token + ".json")
        path.write_text(json.dumps({"artifacts": [{"name": token + ".zip"}]}))
        indexes[token] = path
    def retrieve(deliveries, *_):
        return [indexes[deliveries[0]["index"]["sha256"]]]
    def unpack(packet, destination, _):
        names = groups[packet.stem]
        rows = [{"file": name, "sha256": sha} for name in names]
        for name in names:
            p = destination / "images" / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(source.read_bytes())
        return {"files": rows, "sources": [{"path": name} for name in names]}
    def proofs(meta, *_):
        return {hashlib.sha256(row["file"].encode()).hexdigest():
                {"sha256": sha, "status": "passed" if row["file"] == "passed.png" else "review",
                 "attempts": 1 if row["file"] == "single.png" else 3}
                for row in meta["files"]}
    module = "watermark_studio.early_review."
    monkeypatch.setattr(module + "retrieve", retrieve)
    monkeypatch.setattr(module + "unpack_packet", unpack)
    monkeypatch.setattr(module + "packet_dispositions", proofs)
    monkeypatch.setattr(module + "original_for", lambda *_: source)
    batches = {"cdu_dev1": {"deliveries": [{"index": {"sha256": "one"}}]},
               "cdu_dev2": {"deliveries": [{"index": {"sha256": "two"}}]}}
    config = {"baidu_config": str(tmp_path), "remote_states": {}, "codex_min_server_attempts": 3}
    prepared = prepared_deliveries(config, state, batches, {}, None, state / "stop", set())
    fast_done = []
    lock, released = threading.Lock(), threading.Event()
    def work(payload):
        name = payload[0]["file"]
        if name == "slow.png":
            assert released.wait(5), "Later packets did not fill the other three photo slots"
        else:
            with lock:
                fast_done.append(name)
                if len(fast_done) == 3:
                    released.set()
        return name
    completed = dict(photo_map(prepared, work, 4))
    assert set(completed) == {"slow.png", "a.png", "b.png", "c.png"}
    assert set(fast_done) == {"a.png", "b.png", "c.png"}


@pytest.mark.parametrize("handoff", [False, True])
def test_stream_cursor_only_advances_after_complete_snapshot(tmp_path, monkeypatch, handoff):
    from watermark_studio.early_review import local

    state = tmp_path / "early-codex"
    control = state / "finish-current-and-handoff"
    module = "watermark_studio.early_review."
    monkeypatch.setattr(module + "BaiduDelivery", lambda *_: None)
    calls = []
    def receipts(*_):
        calls.append(1)
        return {"cursor": 123, "deliveries": [1] if len(calls) == 1 else []}
    monkeypatch.setattr(module + "receipts", receipts)
    def prepared(*_):
        if len(calls) == 1:
            for name in ["first.png", "second.png"]:
                yield name, ({"file": name, "sha256": "a" * 64}, tmp_path / name,
                             tmp_path / name, False,
                             {"status": "review", "attempts": 3})
    monkeypatch.setattr(module + "prepared_deliveries", prepared)
    def review(*_):
        if handoff:
            control.touch()
        return {"verdict": {"acceptable": True}}
    monkeypatch.setattr(module + "review_photo", review)
    monkeypatch.setattr(module + "time.sleep", lambda _: control.touch())
    local({"state": str(tmp_path), "baidu_config": str(tmp_path), "codex_executable": "unused",
           "remote_states": {"cdu_dev1": "unused"}, "codex_concurrency": 4,
           "codex_min_server_attempts": 3})
    results = json.loads((state / "results.json").read_text())
    assert results
    cursor = state / "cursors.json"
    if handoff:
        assert not cursor.exists()
    else:
        assert len(results) == 2
        assert json.loads(cursor.read_text()) == {"cdu_dev1": 123}
    assert json.loads((state / "status.json").read_text())["phase"] == "handoff_ready"
