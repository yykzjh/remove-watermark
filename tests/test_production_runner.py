import json
from pathlib import Path

from PIL import Image

from watermark_studio.dataset_transfer import digest
from watermark_studio.production import ProductionRunner, eligible_gpus, gpu_inventory
from watermark_studio.production_queue import WorkBroker


def test_headroom_allows_light_occupancy_and_excludes_own_models(monkeypatch):
    monkeypatch.setattr("watermark_studio.production.group_members", lambda _: [101])
    def query(command, **_):
        if "--query-compute-apps=gpu_uuid,pid,used_memory" in command:
            return "GPU-A, 101, 40000\nGPU-A, 202, 2000\nGPU-B, 203, 19000\n"
        return "GPU-A, 0, 46068, 42008\nGPU-B, 1, 46068, 19008\n"
    monkeypatch.setattr("subprocess.check_output", query)
    rows = gpu_inventory(99)
    assert rows[0]["available"] == 44060
    assert eligible_gpus(rows, 42000, [0, 1]) == [0]
    assert eligible_gpus(rows, 42000, [1]) == []


def test_directory_boundaries_expand_gpus_and_preserve_rejected_outputs(tmp_path, monkeypatch):
    source, output, state = tmp_path/"input", tmp_path/"output", tmp_path/"state"
    source.mkdir()
    for directory in ["A", "B", "empty/deeper"]:
        (source/directory).mkdir(parents=True)
    for relative in ["A/a.jpg", "B/b.jpg"]:
        Image.new("RGB", (16, 12), "green").save(source/relative)
    (source/"original-empty.jpg").write_bytes(b"")
    (source/"notes.txt").write_bytes(b"original auxiliary bytes")
    files = [{"path": p.relative_to(source).as_posix(), "size": p.stat().st_size, "md5": digest(p)}
             for p in source.rglob("*") if p.is_file()]
    dirs = ["."] + [p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_dir()]
    manifest = {"files": files, "directories": dirs}
    broker = WorkBroker(manifest, tmp_path/"broker", ["cdu_dev1"], target=1)
    state.mkdir()
    output.mkdir()
    runner = object.__new__(ProductionRunner)
    runner.c = {"options": {"quality_check": True, "quality_auto_retry": True,
                           "quality_retry_localized": True, "quality_max_attempts": 3},
                "source_is_complete": True, "preserve_auxiliary": True,
                "server_state_dir": str(tmp_path/"service")}
    runner.node, runner.source, runner.output, runner.state = "cdu_dev1", source, output, state
    runner.current, runner.service, runner.gpus, runner.last_heartbeat = None, None, [], 0
    deploys = []
    job = [None]

    def call_broker(path, payload=None):
        if path == "/manifest":
            return manifest
        if path == "/broker/status":
            return broker.status()
        if path == "/broker/claim":
            return broker.claim(payload["node"])
        if path == "/broker/heartbeat":
            return broker.heartbeat(payload["node"], payload["unit_id"], payload["lease"])
        if path == "/broker/complete":
            return broker.complete(payload["node"], payload["unit_id"], payload["lease"], payload["rows"])
        raise AssertionError(path)

    runner.broker = call_broker
    runner.resources = lambda: [0, 1] if broker.status()["directories_completed"] else [0]
    def deploy(gpus):
        deploys.append((broker.status()["directories_completed"], list(gpus)))
        runner.gpus = list(gpus)
        runner.service = {"pid": 42, "gpus": list(gpus)}
    runner.deploy = deploy
    monkeypatch.setattr("watermark_studio.production.group_members", lambda _: [42])
    monkeypatch.setattr("watermark_studio.production.stop_service", lambda _: None)

    def api(path, payload=None):
        if path == "/health":
            return {"stage": "ready", "job": job[0]}
        assert path == "/jobs"
        name = Path(payload["input_dir"]).name
        target = Path(payload["output_dir"])
        target.mkdir(parents=True, exist_ok=True)
        relative = "a.jpg" if name == "A" else "b.jpg"
        if name == "A":
            Image.new("RGB", (16, 12), "blue").save(target/relative)
            result = {"file": relative, "ok": True, "quality_attempts": 3,
                      "quality": {"status": "review", "reason": "visible rectangular shadow"}}
        else:
            result = {"file": relative, "ok": False, "error": "simulated worker failure"}
        folder = Path(runner.c["server_state_dir"])/name
        folder.mkdir(parents=True)
        (folder/"results.jsonl").write_text(json.dumps(result)+"\n")
        job[0] = {"id": name, "state": "completed", "total": 1, "completed": 1,
                  "input_dir": payload["input_dir"], "output_dir": payload["output_dir"],
                  "options": payload["options"]}
        return job[0]

    runner.api = api
    runner.run()
    assert deploys == [(0, [0]), (1, [0, 1])]
    assert broker.status()["state"] == "completed"
    assert broker.status()["passed"] == 0
    assert (output/"A/a.jpg").is_file()
    assert (output/"B/b.jpg").read_bytes() == (source/"B/b.jpg").read_bytes()
    assert (output/"notes.txt").read_bytes() == (source/"notes.txt").read_bytes()
    assert (output/"original-empty.jpg").read_bytes() == b""
    assert (output/"empty/deeper").is_dir()
    assert (state/"needs-review-relative-paths.txt").read_text().splitlines() == ["A/a.jpg", "B/b.jpg"]
    results = [json.loads(line) for line in (state/"needs-review.jsonl").read_text().splitlines()]
    assert results[0]["quality_attempts"] == 3
    assert results[1]["fallback_original"] is True
    assert {p.relative_to(output) for p in output.rglob("*")} == {p.relative_to(source) for p in source.rglob("*")}

def test_failed_service_job_retains_lease_and_is_resubmitted(tmp_path):
    import pytest
    runner = object.__new__(ProductionRunner)
    runner.c = {"options": {}, "server_state_dir": str(tmp_path / "service")}
    runner.output, runner.state = tmp_path / "output", tmp_path
    runner.gpus, runner.current = [0], {"lease": "still-owned"}
    runner.prepare = lambda _: tmp_path / "input/A"
    runner.status = lambda *args, **kwargs: None
    runner.heartbeat = lambda *args, **kwargs: None
    existing = {"id": "old", "state": "failed", "input_dir": str(tmp_path / "input/A"),
                "output_dir": str(tmp_path / "output/A"), "options": {"concurrency": 4}}
    submissions = []
    def api(path, payload=None):
        if path == "/jobs":
            submissions.append(payload)
            return {"id": "new"}
        return {"stage": "ready", "job": existing if not submissions else
                {"id": "new", "state": "failed", "completed": 0, "total": 1}}
    runner.api = api
    with pytest.raises(RuntimeError, match="retain lease"):
        runner.run_directory({"id": "one", "prefix": "A", "files": []})
    assert len(submissions) == 1
    assert runner.current["lease"] == "still-owned"
    assert not (tmp_path / "one-results.json").exists()


def test_deployment_capacity_race_cleans_only_own_group_and_resamples(monkeypatch):
    from watermark_studio.production import DeploymentCapacityError
    runner = object.__new__(ProductionRunner)
    runner.current, runner.service, runner.gpus = {"lease": "owned"}, {"pid": 11}, [0]
    snapshots = iter([[0, 4], [], [2]])
    runner.resources = lambda: next(snapshots)
    stopped, waits, deployments, statuses = [], [], [], []
    runner.status = lambda phase, **extra: statuses.append((phase, extra))
    runner.heartbeat = lambda progress: None
    monkeypatch.setattr("watermark_studio.production.stop_service",
                        lambda plan: stopped.append(plan["pid"]))
    monkeypatch.setattr("watermark_studio.production.time.sleep", waits.append)
    monkeypatch.setattr("watermark_studio.production.group_members", lambda _: [11])
    def deploy(gpus):
        deployments.append(list(gpus))
        runner.service, runner.gpus = {"pid": 22}, list(gpus)
        if len(deployments) == 1:
            raise DeploymentCapacityError("CUDA out of memory")
    runner.deploy = deploy
    runner.ensure_service()
    assert deployments == [[0, 4], [2]]
    assert stopped == [22]
    assert waits == [30, 30]
    assert statuses[0][1]["reason"] == "deployment_capacity_changed"
    assert runner.current == {"lease": "owned"}


def test_non_capacity_deployment_error_still_propagates():
    import pytest
    runner = object.__new__(ProductionRunner)
    runner.service, runner.gpus = None, []
    runner.resources = lambda: [0]
    def deploy(_):
        raise RuntimeError("Editor and verifier must use different weights")
    runner.deploy = deploy
    with pytest.raises(RuntimeError, match="different weights"):
        runner.ensure_service()


def test_resource_wait_releases_expired_lease_without_reviving_it(tmp_path):
    import urllib.error

    import pytest
    runner = object.__new__(ProductionRunner)
    runner.current = {"unit": {"id": "old"}, "lease": "expired", "state": "running"}
    runner.node, runner.state, runner.last_heartbeat = "cdu_dev2", tmp_path, 0
    def broker(*_):
        raise urllib.error.HTTPError("https://broker", 409, "expired", {}, None)
    runner.broker = broker
    runner.heartbeat({"phase": "waiting_for_gpu"})
    assert runner.current is None
    assert json.loads((tmp_path / "current-directory.json").read_text())["state"] == "lease_lost"
    runner.current = {"unit": {"id": "processing"}, "lease": "expired"}
    runner.last_heartbeat = 0
    with pytest.raises(urllib.error.HTTPError):
        runner.heartbeat({"completed": 3})


def test_idle_node_can_finish_when_peer_completes_all_work():
    runner = object.__new__(ProductionRunner)
    runner.current, runner.service, runner.gpus = None, None, []
    runner.resources = list
    runner.broker = lambda path: {"state": "completed"}
    runner.ensure_service()


def test_controller_recovery_reattaches_running_unit_before_expanding_gpus(tmp_path, monkeypatch):
    runner = object.__new__(ProductionRunner)
    runner.source, runner.output, runner.state = tmp_path / "input", tmp_path / "output", tmp_path / "state"
    runner.c = {"options": {"quality_check": True}}
    runner.gpus, runner.service = [0], {"pid": 42}
    monkeypatch.setattr("watermark_studio.production.group_members", lambda _: [42])
    expanded = []
    runner.resources = lambda: expanded.append(True) or [0, 1]
    runner.deploy = lambda selected: setattr(runner, "gpus", selected)
    for mode in ["direct", "subtree"]:
        unit = {"id": "owned", "prefix": "A", "mode": mode}
        runner.current = {"unit": unit}
        input_dir = runner.state / "direct-input/owned" if mode == "direct" else runner.source / "A"
        job = {"state": "running", "input_dir": str(input_dir), "output_dir": str(runner.output / "A"),
               "options": {"quality_check": True, "concurrency": 4}}
        runner.api = lambda _, value=job: {"stage": "ready", "job": value}
        runner.ensure_service()
        assert runner.gpus == [0] and not expanded
    job["state"] = "completed"
    runner.ensure_service()
    assert runner.gpus == [0, 1] and expanded == [True]


def test_controller_recovery_refuses_to_replace_an_unrecognized_active_job(tmp_path, monkeypatch):
    import pytest
    runner = object.__new__(ProductionRunner)
    runner.source, runner.output, runner.state = tmp_path / "input", tmp_path / "output", tmp_path
    runner.current = {"unit": {"id": "owned", "prefix": "A", "mode": "subtree"}}
    runner.c, runner.gpus, runner.service = {"options": {}}, [0], {"pid": 42}
    runner.api = lambda _: {"stage": "ready", "job": {"state": "running", "input_dir": "someone-else"}}
    runner.resources = lambda: pytest.fail("Do not redeploy a model serving an unrecognized job")
    monkeypatch.setattr("watermark_studio.production.group_members", lambda _: [42])
    with pytest.raises(RuntimeError, match="Unrecognized active"):
        runner.ensure_service()


def test_runtime_oom_cannot_be_committed_as_an_unprocessed_photo(tmp_path):
    import pytest

    from watermark_studio.production import DeploymentCapacityError
    runner = object.__new__(ProductionRunner)
    runner.c = {"options": {}, "server_state_dir": str(tmp_path / "service")}
    runner.output, runner.state = tmp_path / "output", tmp_path
    runner.gpus, runner.current = [0], {"lease": "owned"}
    runner.prepare = lambda _: tmp_path / "input/A"
    runner.status = lambda *args, **kwargs: None
    runner.heartbeat = lambda *args, **kwargs: None
    runner.validate_output = lambda *args: pytest.fail("Never normalize a CUDA failure into completed work")
    report = tmp_path / "service/new"
    report.mkdir(parents=True)
    (report / "results.jsonl").write_text(json.dumps(
        {"file": "a.jpg", "ok": False, "error": "RuntimeError: CUDA out of memory"}) + "\n")
    requested = []
    def api(path, payload=None):
        if path == "/jobs":
            requested.append(True)
            return {"id": "new"}
        return {"stage": "ready", "job": None if not requested else
                {"id": "new", "state": "completed", "completed": 1, "total": 1}}
    runner.api = api
    with pytest.raises(DeploymentCapacityError, match="retain lease"):
        runner.run_directory({"id": "unit", "prefix": "A", "files": [{"path": "A/a.jpg"}]})
    assert runner.current["lease"] == "owned"
    assert not (tmp_path / "unit-results.json").exists()
    history = [json.loads(line) for line in (tmp_path / "unit-capacity-replays.jsonl").read_text().splitlines()]
    assert history[0]["job"] == "new"
    assert history[0]["failed"][0]["file"] == "A/a.jpg"

    runner.source, runner.node = tmp_path / "input", "cdu_dev1"
    (runner.source / "A").mkdir(parents=True)
    (runner.output / "A").mkdir(parents=True)
    for photo in [runner.source / "A/a.jpg", runner.output / "A/a.jpg"]:
        Image.new("RGB", (16, 12), "green").save(photo)
    report = tmp_path / "service/recovered"
    report.mkdir()
    (report / "results.jsonl").write_text(json.dumps(
        {"file": "a.jpg", "ok": True, "quality_attempts": 1,
         "quality": {"status": "passed"}}) + "\n")
    runner.api = lambda _: {"stage": "ready", "job": {
        "id": "recovered", "state": "completed", "completed": 1, "total": 1,
        "input_dir": str(runner.source / "A"), "output_dir": str(runner.output / "A"),
        "options": {"concurrency": 4}}}
    runner.validate_output = lambda *args: ProductionRunner.validate_output(runner, *args)
    rows = runner.run_directory({"id": "unit", "prefix": "A", "files": [{"path": "A/a.jpg"}]})
    assert rows[0]["quality_status"] == "passed"
    assert rows[0]["quality_attempts"] == 1
    assert rows[0]["capacity_replays"] == [
        {"job": "new", "reason": "cuda_out_of_memory", "failed_this_file": True}]
