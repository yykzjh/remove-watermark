import hashlib
import json

import pytest

from watermark_studio import production_workflow
from watermark_studio.production_cloud import BaiduDelivery


@pytest.mark.parametrize("parallel", [0, 65, True, 1.5])
def test_invalid_upload_parallelism_cannot_start_network_work(tmp_path, parallel):
    with pytest.raises(ValueError, match="Upload parallelism"):
        BaiduDelivery(tmp_path / "missing-config", tmp_path / "state", upload_parallel=parallel)


@pytest.mark.parametrize("parallel", [None, 16])
def test_upload_passes_selected_threads_and_still_verifies_cloud_size(tmp_path, monkeypatch, parallel):
    config = tmp_path / "config"
    config.mkdir()
    (config / "pcs_config.json").write_text("{}")
    (config / "baidu.cookie").write_text("private-test-cookie")
    kwargs = {} if parallel is None else {"upload_parallel": parallel}
    client = BaiduDelivery(config, tmp_path / "state", **kwargs)
    source = tmp_path / "part.zip"
    source.write_bytes(b"sealed result bytes")
    cloud, calls = {}, []
    monkeypatch.setattr(client, "listing", lambda _: dict(cloud))
    def command(args, label):
        calls.append(args)
        cloud[source.name] = {"isdir": False, "size": source.stat().st_size}
        return 0
    monkeypatch.setattr(client, "command", command)
    client.upload(source, "/results", hashlib.sha256(source.read_bytes()).hexdigest())
    assert calls[0][:6] == ["upload", "-p", str(4 if parallel is None else parallel), "-l", "1", "--retry"]
    # A restart skips the same already-present sealed artifact without another upload.
    client.upload(source, "/results", hashlib.sha256(source.read_bytes()).hexdigest())
    assert len(calls) == 1


def test_upload_only_resume_never_deploys_or_claims_gpu_work(tmp_path, monkeypatch):
    state = tmp_path / "workflow"
    state.mkdir()
    ownership, original = {"units": {}, "manifest_sha256": "stable"}, {"files": [], "directories": ["."]}
    (state / "ownership.json").write_text(json.dumps(ownership))
    (state / "original-manifest.json").write_text(json.dumps(original))
    c = {"workflow_state": str(state), "production": {"output_root": str(tmp_path / "output"),
         "state_dir": str(tmp_path / "controller"), "node": "cdu_dev2"},
         "archive_dir": str(tmp_path / "archives"), "cloud_root": "/results",
         "baidu_config": str(tmp_path / "private-config"), "baidu_upload_parallel": 16}
    monkeypatch.setattr(production_workflow, "ProductionRunner",
                        lambda *a: pytest.fail("Publishing must never claim work or deploy a model"))
    monkeypatch.setattr(production_workflow, "gpu_inventory", lambda *a: pytest.fail("No GPU probes during upload-only resume"))
    verified = []
    def build(output, controller, saved, source, node, archives):
        assert saved == ownership and source == original and node == "cdu_dev2"
        verified.append(True)
        return {"artifacts": [{"name": "sealed.zip"}]}
    monkeypatch.setattr(production_workflow, "build_archives", build)
    receipt = {"state": "published", "node": "cdu_dev2", "manifest_sha256": "stable"}
    calls = []
    def publish(index, folder, config, cloud, upload_parallel):
        assert verified and index.name == "index.json" and cloud == state / "cloud"
        calls.append(upload_parallel)
        return receipt
    monkeypatch.setattr(production_workflow, "publish", publish)
    assert production_workflow.publish_results(c) == receipt
    assert calls == [16]
    assert json.loads((state / "delivery.json").read_text()) == receipt
    assert json.loads((state / "status.json").read_text())["phase"] == "published"


def test_upload_only_resume_refuses_missing_frozen_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(production_workflow, "build_archives", lambda *a: pytest.fail("Missing ownership"))
    monkeypatch.setattr(production_workflow, "publish", lambda *a: pytest.fail("Never upload without evidence"))
    with pytest.raises(FileNotFoundError):
        production_workflow.publish_results({"workflow_state": str(tmp_path)})
