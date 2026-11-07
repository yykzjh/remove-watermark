from pathlib import Path

import pytest

from watermark_studio import production_local


def local_config(tmp_path):
    return {"state": str(tmp_path / "state"), "remote_states": {"cdu_dev1": "one", "cdu_dev2": "two"},
            "baidu_config": str(tmp_path / "baidu"), "artifacts": str(tmp_path / "artifacts"),
            "destination": str(tmp_path / "output"), "original_root": str(tmp_path / "originals"),
            "codex_executable": "unused"}


def receipt(node, identity="stable"):
    return {"state": "published", "node": node, "index": {"sha256": identity}}


def install_final_stages(monkeypatch, events):
    def assemble(indexes, *args):
        events.append(("assemble", [str(p) for p in indexes]))
        assert len(indexes) == 2
        return {"files": 2}
    monkeypatch.setattr(production_local, "assemble", assemble)
    monkeypatch.setattr(production_local, "wait_for_handoff", lambda *a: None)
    monkeypatch.setattr(production_local, "run_audit", lambda *a, **kw: {"sample": 1000})
    monkeypatch.setattr(production_local, "run_repairs", lambda *a, **kw: {"completed": 2})
    monkeypatch.setattr(production_local, "finalize", lambda *a: {"state": "completed"})
    monkeypatch.setattr(production_local.time, "sleep", lambda *a: None)


@pytest.mark.parametrize("together", [False, True])
def test_first_published_node_downloads_before_waiting_for_the_other(tmp_path, monkeypatch, together):
    events, counts = [], {}
    c = local_config(tmp_path)
    def snapshot(node, root):
        counts[node] = counts.get(node, 0) + 1
        if node == "cdu_dev1" and counts[node] == 1 and not together:
            events.append(("not_ready", node))
            return {}
        events.append(("published", node))
        return {"workflow/delivery.json": receipt(node)}
    downloads = []
    def retrieve(deliveries, config, cloud, artifacts):
        assert len(deliveries) == 1
        node = deliveries[0]["node"]
        events.append(("download", node))
        downloads.append(node)
        return [Path(artifacts) / (node + ".json")]
    monkeypatch.setattr(production_local, "remote_snapshot", snapshot)
    monkeypatch.setattr(production_local, "retrieve", retrieve)
    install_final_stages(monkeypatch, events)
    assert production_local.run(c)["state"] == "completed"
    assert downloads == (["cdu_dev1", "cdu_dev2"] if together else ["cdu_dev2", "cdu_dev1"])
    if not together:
        assert events.index(("download", "cdu_dev2")) < events.index(("published", "cdu_dev1"))
    assembled = next(value for name, value in events if name == "assemble")
    assert assembled == [str(Path(c["artifacts"]) / (node + ".json")) for node in ["cdu_dev1", "cdu_dev2"]]
    assert events.index(("download", "cdu_dev1")) < next(i for i, e in enumerate(events) if e[0] == "assemble")
    assert events.index(("download", "cdu_dev2")) < next(i for i, e in enumerate(events) if e[0] == "assemble")


def test_downloaded_node_receipt_cannot_change_while_waiting_for_other(tmp_path, monkeypatch):
    counts = {}
    def snapshot(node, root):
        counts[node] = counts.get(node, 0) + 1
        if node == "cdu_dev1":
            return {}
        return {"workflow/delivery.json": receipt(node, str(counts[node]))}
    downloads = []
    monkeypatch.setattr(production_local, "remote_snapshot", snapshot)
    monkeypatch.setattr(production_local, "retrieve",
                        lambda deliveries, *a: downloads.append(deliveries[0]["node"]) or [tmp_path / "two.json"])
    monkeypatch.setattr(production_local.time, "sleep", lambda *a: None)
    monkeypatch.setattr(production_local, "assemble", lambda *a: pytest.fail("Do not assemble inconsistent versions"))
    with pytest.raises(ValueError, match="delivery receipt changed"):
        production_local.run(local_config(tmp_path))
    assert downloads == ["cdu_dev2"]


def test_one_verified_node_is_insufficient_if_second_download_fails(tmp_path, monkeypatch):
    counts = {}
    def snapshot(node, root):
        counts[node] = counts.get(node, 0) + 1
        if node == "cdu_dev1" and counts[node] == 1:
            return {}
        return {"workflow/delivery.json": receipt(node)}
    downloads = []
    def retrieve(deliveries, *args):
        node = deliveries[0]["node"]
        downloads.append(node)
        if node == "cdu_dev1":
            raise ConnectionError("First node is not fully verified")
        return [tmp_path / "two.json"]
    monkeypatch.setattr(production_local, "remote_snapshot", snapshot)
    monkeypatch.setattr(production_local, "retrieve", retrieve)
    monkeypatch.setattr(production_local.time, "sleep", lambda *a: None)
    monkeypatch.setattr(production_local, "assemble", lambda *a: pytest.fail("Both nodes must be fully verified"))
    with pytest.raises(ConnectionError, match="not fully verified"):
        production_local.run(local_config(tmp_path))
    assert downloads == ["cdu_dev2", "cdu_dev1"]
