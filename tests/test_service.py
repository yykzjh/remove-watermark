import json
import time

import pytest
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
    (source / "乡" / "村" / "批次" / "deep" / "empty").mkdir(parents=True)
    for number in range(4):
        Image.new("RGBA", (64, 48), (number * 30, 60, 90, 170)).save(
            source / "乡" / "村" / "批次" / "deep" / f"{number}.png"
        )
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
            "options": {"concurrency": cfg.capacity + 1},
        }
        assert client.post("/jobs", json=payload).status_code == 409
        payload["options"]["concurrency"] = 4
        payload["options"]["quality_check"] = True
        response = client.post("/jobs", json=payload)
        assert response.status_code == 202, response.text
        state = wait_for(client, lambda s: s["job"] and s["job"]["state"] == "completed")
        assert state["job"]["succeeded"] == 4
        assert state["job"]["quality_counts"] == {"skipped": 4}
        assert "deployment" not in state["job"]
        summary = json.loads((tmp_path / "state" / state["job"]["id"] / "summary.json").read_text())
        assert summary["deployment"] == cfg.model_dump()
        assert state["job"]["invalid"] == 1
        assert len(list(target.rglob("*.png"))) == 4
        assert (target / "irrelevant.txt").read_bytes() == (source / "irrelevant.txt").read_bytes()
        assert (target / "invalid.jpg").read_bytes() == (source / "invalid.jpg").read_bytes()
        assert (target / "乡" / "村" / "批次" / "deep" / "empty").is_dir()
        assert state["job"]["mirror"]["copied_files"] == 2
        assert state["job"]["mirror"]["copied_invalid_images"] == 1
        for original in source.rglob("*.png"):
            with Image.open(original) as a, Image.open(target / original.relative_to(source)) as b:
                assert a.size == b.size and a.tobytes() == b.tobytes()
        saved = {p.name: p.read_bytes() for p in target.rglob("*.png")}
        assert client.post("/jobs", json=payload).status_code == 202
        state = wait_for(client, lambda s: s["job"] and s["job"]["state"] == "completed")
        assert state["job"]["skipped"] == 0
        assert state["job"]["total"] == 4
        assert state["job"]["existing_outputs"] == 4
        assert {p.name: p.read_bytes() for p in target.rglob("*.png")} == saved
        assert list((tmp_path / "state").glob("*/summary.json"))


@pytest.mark.parametrize("fail_copy", [False, True])
def test_files_only_tree_completion_and_copy_error_reporting(tmp_path, monkeypatch, fail_copy):
    from watermark_studio import images

    source, target = tmp_path/"input", tmp_path/"output"
    (source/"a"/"b"/"empty").mkdir(parents=True)
    (source/"a"/"b"/"notes.txt").write_bytes(b"notes")
    (source/"blob.bin").write_bytes(b"binary")
    if fail_copy:
        original = images.atomic_copy
        def copy(src, *args, **kwargs):
            if src.name == "blob.bin":
                raise OSError("injected disk full")
            return original(src, *args, **kwargs)
        monkeypatch.setattr(images, "atomic_copy", copy)
    cfg = DeploymentConfig(backend="mock", gpus=[0], allowed_roots=[str(tmp_path)],
                           state_dir=str(tmp_path/"state"))
    with TestClient(create_app(cfg)) as client:
        wait_for(client, lambda s: s["stage"] == "ready")
        response = client.post("/jobs", json={"input_dir":str(source), "output_dir":str(target),
                                              "options":{"concurrency":1}})
        assert response.status_code == 202
        state = wait_for(client, lambda s: s["job"] and s["job"]["state"] in {"completed","failed"})
        job = state["job"]
        assert job["total"] == 0 and job["state"] == ("failed" if fail_copy else "completed")
        assert job["mirror"]["copy_failures"] == int(fail_copy)
        assert job["mirror"]["copied_files"] == (1 if fail_copy else 2)
        assert (target/"a"/"b"/"empty").is_dir()
        assert (target/"a"/"b"/"notes.txt").read_bytes() == b"notes"
        errors = json.loads((tmp_path/"state"/job["id"]/"copy-errors.json").read_text())
        assert bool(errors) == fail_copy
        assert job["acceptance_complete"] is False
