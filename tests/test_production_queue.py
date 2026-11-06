import concurrent.futures as cf

import pytest

from watermark_studio.production_queue import WorkBroker, directory_units


def manifest():
    return {"files": [{"path": f"town{t}/village{v}/deep/image{i}.jpg", "size": 100, "md5": "x"}
                      for t in range(2) for v in range(3) for i in range(110)]
            + [{"path": "empty.jpg", "size": 0, "md5": "x"},
               {"path": "document.pdf", "size": 50, "md5": "x"}],
            "directories": ["."]}


def completed_rows(unit, quality="passed"):
    return [{"file": r["path"], "quality_status": quality, "sha256": "a" * 64, "size": 100}
            for r in unit["files"]]


def test_recursive_directory_units_cover_images_exactly_once():
    units = directory_units(manifest())
    paths = [r["path"] for u in units for r in u["files"]]
    assert len(units) == 6
    assert len(paths) == len(set(paths)) == 660
    assert all(u["mode"] == "subtree" for u in units)
    assert "empty.jpg" not in paths and "document.pdf" not in paths


def test_parallel_claims_are_exclusive_and_progress_persists(tmp_path):
    broker = WorkBroker(manifest(), tmp_path, ["one", "two"])
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(broker.claim, ["one", "two"] * 8))
    selected = [c for c in claims if "unit" in c]
    ids = [c["unit"]["id"] for c in selected]
    assert len(ids) == len(set(ids)) == 6
    restarted = WorkBroker(manifest(), tmp_path, ["one", "two"])
    assert restarted.claim("one")["wait"]


def test_expired_lease_cannot_commit_after_reassignment(tmp_path):
    now = [0]
    broker = WorkBroker(manifest(), tmp_path, ["one", "two"], clock=lambda: now[0])
    first = broker.claim("one")
    now[0] = 1801
    second = broker.claim("two")
    assert first["unit"]["id"] == second["unit"]["id"]
    with pytest.raises(ValueError, match="Stale"):
        broker.complete("one", first["unit"]["id"], first["lease"], completed_rows(first["unit"]))
    broker.complete("two", second["unit"]["id"], second["lease"], completed_rows(second["unit"], "review"))
    assert broker.status()["images_completed"] == 110
    assert broker.status()["passed"] == 0


def test_completion_requires_every_output_and_preserves_failed_quality(tmp_path):
    broker = WorkBroker(manifest(), tmp_path, ["one", "two"])
    claim = broker.claim("one")
    with pytest.raises(ValueError, match="exact input"):
        broker.complete("one", claim["unit"]["id"], claim["lease"], completed_rows(claim["unit"])[:-1])
    rows = completed_rows(claim["unit"], "review")
    broker.complete("one", claim["unit"]["id"], claim["lease"], rows)
    assert broker.status()["passed"] == 0
    with pytest.raises(ValueError, match="not finished"):
        broker.ownership()


def test_unauthorized_third_machine_is_rejected(tmp_path):
    broker = WorkBroker(manifest(), tmp_path, ["cdu_dev1", "cdu_dev2"])
    with pytest.raises(ValueError, match="authorized"):
        broker.claim("cdu_dev3")
