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
