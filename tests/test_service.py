import time

from fastapi.testclient import TestClient
from PIL import Image

from watermark_studio.config import DeploymentConfig
from watermark_studio.server import create_app


def wait_for(client, predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = client.get("/health").json()
        if predicate(state):
            return state
        time.sleep(0.1)
    raise AssertionError(state)


def test_complete_remote_directory_batch_and_limits(tmp_path):
    source, target = tmp_path / "input", tmp_path / "output"
    (source / "deep").mkdir(parents=True)
    for number in range(4):
        Image.new("RGB", (64, 48), (number * 30, 60, 90)).save(source / "deep" / f"{number}.png")
    (source / "irrelevant.txt").write_text("ignored")
    (source / "invalid.jpg").write_text("invalid")
    cfg = DeploymentConfig(
        backend="mock",
        gpus=[0, 1],
        allowed_roots=[str(tmp_path)],
        state_dir=str(tmp_path / "state"),
    )
    with TestClient(create_app(cfg)) as client:
        state = wait_for(client, lambda s: s["stage"] in {"ready", "failed"})
        assert state["stage"] == "ready", state
        assert client.get("/directories", params={"path": "/etc"}).status_code == 400
        payload = {
            "input_dir": str(source),
            "output_dir": str(target),
            "options": {"concurrency": 3},
        }
        assert client.post("/jobs", json=payload).status_code == 409
        payload["options"]["concurrency"] = 2
        response = client.post("/jobs", json=payload)
        assert response.status_code == 202, response.text
        state = wait_for(client, lambda s: s["job"] and s["job"]["state"] == "completed")
        assert state["job"]["succeeded"] == 4
        assert state["job"]["invalid"] == 1
        assert len(list(target.rglob("*.png"))) == 4
        assert not (target / "irrelevant.txt").exists()
        for original in source.rglob("*.png"):
            with Image.open(original) as a, Image.open(target / original.relative_to(source)) as b:
                assert a.size == b.size and a.tobytes() == b.tobytes()
        assert client.post("/jobs", json=payload).status_code == 202
        state = wait_for(client, lambda s: s["job"] and s["job"]["state"] == "completed")
        assert state["job"]["skipped"] == 4
        assert state["job"]["total"] == 0
        assert list((tmp_path / "state").glob("*/summary.json"))
