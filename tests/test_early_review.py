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


def sample_packet(tmp_path):
    from watermark_studio.codex_sampling import make_sampling_plan
    output, rows, sources = packet(tmp_path)
    extra = "town/village/third.png"
    Image.new("RGB", (32, 48), "gray").save(output / extra)
    sources[extra] = {"path": extra, "size": (output / extra).stat().st_size,
                      "md5": hashlib.md5((output / extra).read_bytes()).hexdigest()}
    rows.append({"file": extra, "sha256": file_sha256(output / extra),
                 "size": sources[extra]["size"], "regions": []})
    plan = make_sampling_plan([r["file"] for r in rows], 2)
    for row in rows:
        row["quality_status"] = "review" if row["file"] == plan["primary"][1] else "passed"
    index = create_packet(output, rows, sources, "source", "cdu_dev1", "unit",
                          tmp_path / "packet", "sample-1000", plan)
    return output, plan, index


def test_passed_packet_contains_only_frozen_passed_candidates_and_no_large_ssh_index(tmp_path):
    output, plan, index = sample_packet(tmp_path)
    assert index["files"] == 1 and "sampling_plan" not in index
    assert len(json.dumps(index)) < 4096
    receipt = {"node": "cdu_dev1", "unit": "unit", "manifest_sha256": "source"}
    meta = unpack_packet(Path(index["artifacts"][0]["local_path"]), tmp_path / "unpacked", receipt)
    assert meta["sampling_plan"] == plan and meta["audit_group"] == "sample-1000"
    assert [r["file"] for r in meta["files"]] == [plan["primary"][0]]
    assert (tmp_path / "unpacked/images" / plan["primary"][0]).read_bytes() == (
        output / plan["primary"][0]).read_bytes()


@pytest.mark.parametrize("proof_status", ["passed", "review"])
@pytest.mark.parametrize("cached", [False, True])
def test_passed_inspection_requires_committed_passed_proof_but_not_three_edit_attempts(tmp_path, monkeypatch, proof_status, cached):
    import shutil

    from watermark_studio.early_review import prepared_deliveries
    output, plan, index = sample_packet(tmp_path)
    state = tmp_path / "early"
    state.mkdir()
    index_path = tmp_path / "packet/index.json"
    receipt = {"node": "cdu_dev1", "unit": "unit", "manifest_sha256": "source",
               "index": {"sha256": file_sha256(index_path)}}
    def retrieve(_, config, cloud, destination):
        destination.mkdir(parents=True, exist_ok=True)
        for artifact in index["artifacts"]:
            shutil.copyfile(artifact["local_path"], destination / artifact["name"])
        return [index_path]
    monkeypatch.setattr("watermark_studio.early_review.retrieve", retrieve)
    def proofs(meta, *_):
        return {hashlib.sha256(r["file"].encode()).hexdigest():
                {"sha256": r["sha256"], "status": proof_status, "attempts": 1}
                for r in meta["files"]}
    monkeypatch.setattr("watermark_studio.early_review.packet_dispositions", proofs)
    originals = []
    def original(row, *_):
        originals.append(row["path"])
        return output / row["path"]
    monkeypatch.setattr("watermark_studio.early_review.original_for", original)
    c = {"baidu_config": str(tmp_path), "remote_states": {}, "codex_min_server_attempts": 3,
         "codex_sample_population": 3, "codex_sample_size": 2}
    batches = {"cdu_dev1:sample-1000": {"deliveries": [receipt]}}
    previous = {plan["primary"][0]: {"baseline_sha256": file_sha256(output / plan["primary"][0]),
                                    "status": "needs_review"}} if cached else {}
    rejected_path = state / "results.json"
    rejected_path.write_text(json.dumps({"preserved-server-reject.png": {"status": "passed"}}))
    before = rejected_path.read_bytes()
    prepared = prepared_deliveries(c, state, batches, previous, None, state / "stop",
                                   {(plan["primary"][0], file_sha256(output / plan["primary"][0]))},
                                   "sample-1000")
    if proof_status != "passed":
        with pytest.raises(ValueError, match="committed passed"):
            list(prepared)
        assert not originals
    else:
        items = list(prepared)
        assert originals == ([] if cached else [plan["primary"][0]])
        if not cached:
            assert items[0][1][0]["_audit_group"] == "sample-1000"
            assert items[0][1][3] is False
        else:
            assert items == []
            assert json.loads((state / "passed-results.json").read_text()) == previous
        assert json.loads((state / "stream-sampling-plan.json").read_text()) == plan
    assert rejected_path.read_bytes() == before


def test_failed_samples_are_review_only_and_both_groups_share_four_slots(tmp_path, monkeypatch):
    from watermark_studio.early_review import local
    state = tmp_path / "early-codex"
    control = state / "finish-current-and-handoff"
    module = "watermark_studio.early_review."
    monkeypatch.setattr(module + "BaiduDelivery", lambda *_: None)
    calls = {}
    def receipts(node, root, cursor, group="initially-rejected"):
        calls[group] = calls.get(group, 0) + 1
        return {"cursor": 123 if group == "initially-rejected" else 456,
                "deliveries": [1] if calls[group] == 1 else []}
    monkeypatch.setattr(module + "receipts", receipts)
    def prepared(c, state, batches, results, client, control, human, group="initially-rejected"):
        if any(b["deliveries"] for b in batches.values()):
            for n in range(2):
                name = f"{group}-{n}.png"
                yield name, ({"file": name, "sha256": "a" * 64, "_audit_group": group},
                             tmp_path / name, tmp_path / name, False,
                             {"status": "passed" if group == "sample-1000" else "review",
                              "attempts": 1 if group == "sample-1000" else 3})
    monkeypatch.setattr(module + "prepared_deliveries", prepared)
    lock, release = threading.Lock(), threading.Event()
    active, peak, groups = [0], [0], []
    def review(executable, original, candidate, folder, regions):
        group = folder.parent.name
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            groups.append(group)
            if active[0] == 4:
                release.set()
        assert release.wait(5), "The two groups must run together in one pool"
        value = {"acceptable": group != "sample-1000", "watermark_removed": True,
                 "natural_fill": group != "sample-1000", "content_preserved": True,
                 "uncertain": False, "evidence": ["A visible seam remains."], "defects": []}
        if group == "sample-1000":
            value["defects"] = [{"kind": "unnatural_fill", "box": [.1, .1, .2, .2],
                                 "evidence": "A visible seam remains.", "repair_instruction": "Blend the seam."}]
        with lock:
            active[0] -= 1
        return {"identity": {"output_sha256": "a" * 64}, "verdict": value}
    monkeypatch.setattr(module + "review_photo", review)
    monkeypatch.setattr(module + "repair_one", lambda *a: pytest.fail("Never locally edit server-passed samples"))
    monkeypatch.setattr(module + "time.sleep", lambda _: control.touch())
    local({"state": str(tmp_path), "baidu_config": str(tmp_path), "codex_executable": "unused",
           "remote_states": {"cdu_dev1": "unused"}, "codex_concurrency": 4,
           "codex_min_server_attempts": 3, "stream_passed_sample": True,
           "codex_sample_population": 10})
    assert peak[0] == 4 and set(groups) == {"initially-rejected", "sample-1000"}
    passed = json.loads((state / "passed-results.json").read_text())
    rejected = json.loads((state / "results.json").read_text())
    assert len(passed) == len(rejected) == 2
    assert all(r["status"] == "needs_review" and r["audit_only"] for r in passed.values())
    assert all(r["server_attempts"] == 1 and not r["review"]["verdict"]["acceptable"] for r in passed.values())
    assert json.loads((state / "cursors.json").read_text()) == {
        "cdu_dev1": 123, "cdu_dev1:sample-1000": 456}

    assert json.loads((state / "status.json").read_text())["passed_sample_reviewed"] == 2


def test_export_cannot_rebind_its_frozen_candidates_even_without_new_units(tmp_path):
    from watermark_studio.early_review import export
    workflow = tmp_path / "state/workflow"
    workflow.mkdir(parents=True)
    (workflow / "status.json").write_text(json.dumps({"phase": "published"}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": [{"path": "a.jpg", "size": 1},
                                             {"path": "b.jpg", "size": 1}]}))
    c = {"production": {"state_dir": str(tmp_path / "controller")},
         "workflow_state": str(workflow)}
    export(c, manifest, "sample-1000", 1, 7, 2)
    path = workflow.parent / "passed-export/sampling-plan.json"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="already frozen"):
        export(c, manifest, "sample-1000", 1, 8, 2)
    assert path.read_bytes() == before


def test_prefetch_uses_separate_bounded_metadata_reads(tmp_path, monkeypatch):
    from watermark_studio.early_review import receipt_batch
    positions = []
    def read(node, root, cursor, group="initially-rejected"):
        positions.append(cursor)
        return {"cursor": cursor + 2, "deliveries": [cursor, cursor + 1]}
    monkeypatch.setattr("watermark_studio.early_review.receipts", read)
    result = receipt_batch("cdu_dev1", "unused", 0, maximum=8)
    assert positions == [0, 2, 4, 6]
    assert result == {"cursor": 8, "deliveries": list(range(8))}
