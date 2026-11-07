import hashlib
import json
import shutil
from pathlib import Path

import pytest

from watermark_studio import production_cloud, production_local, production_workflow


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sealed_index(tmp_path):
    source = tmp_path / "remote"
    source.mkdir()
    artifacts = []
    for n in range(3):
        p = source / f"cdu_dev1-part{n}.zip"
        p.write_bytes(bytes([n]) * 100)
        artifacts.append({"name": p.name, "size": p.stat().st_size, "sha256": sha(p),
                          "kind": "images", "local_path": str(p)})
    index = {"schema": 1, "node": "cdu_dev1", "manifest_sha256": "frozen",
             "files": 3, "image_files": 3, "auxiliary_files": 0, "artifacts": artifacts}
    path = tmp_path / "index.json"
    path.write_text(json.dumps(index))
    return path, index, source


def test_preview_uploads_only_same_tiny_index_as_final_publication(tmp_path, monkeypatch):
    path, index, _source = sealed_index(tmp_path)
    uploads = []
    class Client:
        def __init__(self, config, state, *a, **kw):
            Path(state).mkdir(parents=True, exist_ok=True)
        def command(self, *args):
            return 0
        def listing(self, *args):
            return {}
        def upload(self, path, folder, expected):
            p = Path(path)
            assert sha(p) == expected
            uploads.append((p.name, expected))
    monkeypatch.setattr(production_cloud, "BaiduDelivery", Client)
    preview = production_cloud.publish_preview(path, "/results", tmp_path, tmp_path / "preview")
    assert preview["state"] == "publishing" and len(uploads) == 1
    assert uploads[0][0].endswith(".json") and len(json.dumps(preview)) < 4096
    assert "artifacts" not in preview
    final = production_cloud.publish(path, "/results", tmp_path, tmp_path / "final")
    assert production_cloud.delivery_identity(preview) == production_cloud.delivery_identity(final)
    assert final["state"] == "published" and uploads[-1][1] == uploads[0][1]
    assert uploads[-1][0] != uploads[0][0]  # No concurrent writes to the metadata object.
    assert [n for n, _ in uploads[1:-1]] == [a["name"] for a in index["artifacts"]]


def receiver_fixture(tmp_path, monkeypatch):
    _path, index, source = sealed_index(tmp_path)
    public = source / "cdu_dev1-index.json"
    public.write_text(json.dumps(production_cloud.public_index(index)))
    receipt = {"state": "publishing", "node": "cdu_dev1", "cloud_root": "/results",
               "manifest_sha256": "frozen", "index": {"name": public.name,
               "size": public.stat().st_size, "sha256": sha(public)}}
    calls = []
    class Client:
        def __init__(self, *a, **kw):
            pass
        def download(self, root, artifact, destination):
            src = source / artifact["name"]
            assert src.stat().st_size == artifact["size"] and sha(src) == artifact["sha256"]
            calls.append(artifact["name"])
            out = Path(destination) / artifact["name"]
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, out)
            return out
    monkeypatch.setattr(production_cloud, "BaiduDelivery", Client)
    return receipt, calls, index


def test_only_available_prefix_is_fetched_once_and_final_state_is_required(tmp_path, monkeypatch):
    receipt, calls, index = receiver_fixture(tmp_path, monkeypatch)
    args = (tmp_path, tmp_path / "state", tmp_path / "cache")
    progress = production_cloud.retrieve_available(receipt, 0, *args)
    assert progress["verified"] == 0 and calls == [receipt["index"]["name"]]
    progress = production_cloud.retrieve_available(receipt, 2, *args, previous=progress)
    assert progress["verified"] == 2
    before = list(calls)
    # A producer restart can temporarily report a lower prefix; never lose verified work.
    progress = production_cloud.retrieve_available(receipt, 1, *args, previous=progress)
    assert progress["verified"] == 2 and calls == before
    progress = production_cloud.retrieve_available(receipt, 3, *args, previous=progress)
    assert progress["verified"] == 3 and receipt["state"] == "publishing"
    final = dict(receipt, state="published")
    progress = production_cloud.retrieve_available(final, None, *args, previous=progress)
    assert progress["verified"] == 3
    for artifact in index["artifacts"]:
        assert calls.count(artifact["name"]) == 1
    assert calls.count(receipt["index"]["name"]) == 2


@pytest.mark.parametrize("available", [-1, 4, True, 1.5, None])
def test_partial_index_cannot_request_unknown_or_unbounded_prefix(tmp_path, monkeypatch, available):
    receipt, calls, _ = receiver_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="prefix|bounded"):
        production_cloud.retrieve_available(receipt, available, tmp_path, tmp_path / "state", tmp_path / "cache")
    assert calls == [receipt["index"]["name"]]


def test_streaming_header_cannot_change_when_final_receipt_arrives(tmp_path, monkeypatch):
    receipt, calls, _ = receiver_fixture(tmp_path, monkeypatch)
    args = (tmp_path, tmp_path / "state", tmp_path / "cache")
    progress = production_cloud.retrieve_available(receipt, 1, *args)
    before = list(calls)
    changed = dict(receipt, state="published", manifest_sha256="another-input")
    with pytest.raises(ValueError, match="receipt changed"):
        production_cloud.retrieve_available(changed, None, *args, previous=progress)
    assert calls == before


def test_preview_cli_failure_does_not_replace_live_upload_status(tmp_path, monkeypatch):
    state = tmp_path / "workflow"
    state.mkdir()
    original = {"phase": "uploading", "upload_parallel": 16}
    (state / "status.json").write_text(json.dumps(original))
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"workflow_state": str(state)}))
    def failed(*a):
        raise ConnectionError("Preview metadata temporarily unavailable")
    monkeypatch.setattr(production_workflow, "preview_results", failed)
    monkeypatch.setattr("sys.argv", ["workflow", "publish-preview", "--config", str(cfg)])
    with pytest.raises(ConnectionError):
        production_workflow.main()
    assert json.loads((state / "status.json").read_text()) == original
    assert json.loads((state / "preview-status.json").read_text())["phase"] == "error"


def header(node, state="publishing", identity="stable"):
    return {"state": state, "node": node, "cloud_root": "/results",
            "manifest_sha256": "frozen", "index": {"name": node + "-index.json",
            "size": 100, "sha256": identity}}


def test_coordinator_prefetches_parts_but_assembles_only_after_both_final_receipts(tmp_path, monkeypatch):
    counts, events, final_seen = {}, [], set()
    def snapshot(node, root):
        counts[node] = counts.get(node, 0) + 1
        n = counts[node]
        ready = n >= (3 if node == "cdu_dev1" else 2)
        result = {"workflow/preview-delivery.json": header(node),
                  "workflow/cloud/status.json": {"complete": min(2, n - 1 if node == "cdu_dev1" else n)}}
        if ready:
            final_seen.add(node)
            events.append(("final", node))
            result["workflow/delivery.json"] = header(node, "published")
        return result
    def receive(delivery, available, config, cloud, artifacts, previous=None):
        node = delivery["node"]
        events.append(("prefix", node, available))
        identity = production_cloud.delivery_identity(delivery)
        if previous:
            assert previous["identity"] == identity
        return {"identity": identity, "index_path": str(Path(artifacts) / (node + ".json")),
                "artifacts": [{}, {}], "verified": 2 if available is None else available}
    def assemble(indexes, *args):
        assert final_seen == {"cdu_dev1", "cdu_dev2"}
        assert [Path(x).stem for x in indexes] == ["cdu_dev1", "cdu_dev2"]
        events.append(("assemble",))
        return {"files": 4}
    monkeypatch.setattr(production_local, "remote_snapshot", snapshot)
    monkeypatch.setattr(production_local, "retrieve_available", receive)
    monkeypatch.setattr(production_local, "retrieve", lambda *a: pytest.fail("Use the verified streamed prefix"))
    monkeypatch.setattr(production_local, "assemble", assemble)
    monkeypatch.setattr(production_local, "wait_for_handoff", lambda *a: None)
    monkeypatch.setattr(production_local, "run_audit", lambda *a, **kw: {})
    monkeypatch.setattr(production_local, "run_repairs", lambda *a, **kw: {})
    monkeypatch.setattr(production_local, "finalize", lambda *a: {"state": "completed"})
    monkeypatch.setattr(production_local.time, "sleep", lambda *a: None)
    c = {"state": str(tmp_path / "state"), "remote_states": {"cdu_dev1": "one", "cdu_dev2": "two"},
         "baidu_config": str(tmp_path / "private"), "artifacts": str(tmp_path / "artifacts"),
         "destination": str(tmp_path / "output"), "original_root": str(tmp_path / "originals"),
         "codex_executable": "unused"}
    assert production_local.run(c)["state"] == "completed"
    assert events.index(("prefix", "cdu_dev2", 1)) < events.index(("final", "cdu_dev2"))
    assert events.index(("prefix", "cdu_dev1", 1)) < events.index(("final", "cdu_dev1"))
    assert events[-1] == ("assemble",)


def test_coordinator_rejects_final_index_different_from_prefetch(tmp_path, monkeypatch):
    counts = {}
    def snapshot(node, root):
        counts[node] = counts.get(node, 0) + 1
        d = {"workflow/preview-delivery.json": header(node),
             "workflow/cloud/status.json": {"complete": 0}}
        if node == "cdu_dev1" and counts[node] > 1:
            d["workflow/delivery.json"] = header(node, "published", "changed")
        return d
    def receive(d, *a, **kw):
        return {"identity": production_cloud.delivery_identity(d), "index_path": "unused",
                "verified": 0, "artifacts": [{}]}
    monkeypatch.setattr(production_local, "remote_snapshot", snapshot)
    monkeypatch.setattr(production_local, "retrieve_available", receive)
    monkeypatch.setattr(production_local, "assemble", lambda *a: pytest.fail("Never mix publication versions"))
    monkeypatch.setattr(production_local.time, "sleep", lambda *a: None)
    c = {"state": str(tmp_path / "state"), "remote_states": {"cdu_dev1": "one", "cdu_dev2": "two"},
         "baidu_config": str(tmp_path), "artifacts": str(tmp_path / "artifacts")}
    with pytest.raises(ValueError, match="Final delivery differs"):
        production_local.run(c)


def test_public_ssh_snapshot_keeps_only_bounded_preview_receipt_metadata(tmp_path, monkeypatch):
    import shlex
    import subprocess
    import sys

    real_run = subprocess.run

    workflow = tmp_path / "workflow"
    workflow.mkdir()
    preview = header("cdu_dev1")
    preview["debug_payload"] = ["never-cross-ssh"] * 10000
    preview["index"]["debug_payload"] = ["never-cross-ssh"] * 10000
    (workflow / "preview-delivery.json").write_text(json.dumps(preview))
    (workflow / "cloud").mkdir()
    (workflow / "cloud/status.json").write_text(json.dumps(
        {"state": "uploading", "complete": 3, "total": 36, "ignored": ["large"] * 10000}))
    def execute(cmd, **kw):
        assert cmd[0] == "ssh"
        words = shlex.split(cmd[-1])
        assert words[:2] == ["python3", "-c"]
        result = real_run([sys.executable, "-c", words[2]], capture_output=True, text=True, check=True)
        assert len(result.stdout) < 4096
        return subprocess.CompletedProcess(cmd, 0, stdout=result.stdout, stderr="")
    monkeypatch.setattr(production_local.subprocess, "run", execute)
    snapshot = production_local.remote_snapshot("cdu_dev1", str(tmp_path))
    assert snapshot["workflow/preview-delivery.json"] == header("cdu_dev1")
    assert snapshot["workflow/cloud/status.json"]["complete"] == 3
    assert "ignored" not in snapshot["workflow/cloud/status.json"]


def test_final_metadata_alias_still_requires_identical_pinned_bytes(tmp_path, monkeypatch):
    receipt, _calls, _ = receiver_fixture(tmp_path, monkeypatch)
    args = (tmp_path, tmp_path / "state", tmp_path / "cache")
    progress = production_cloud.retrieve_available(receipt, 1, *args)
    old = tmp_path / "remote" / receipt["index"]["name"]
    final_name = "cdu_dev1-final-index.json"
    shutil.copyfile(old, old.parent / final_name)
    final = dict(receipt, state="published", index=dict(receipt["index"], name=final_name))
    assert production_cloud.delivery_identity(final) == production_cloud.delivery_identity(receipt)
    progress = production_cloud.retrieve_available(final, None, *args, previous=progress)
    assert progress["verified"] == 3 and Path(progress["index_path"]).name == final_name
