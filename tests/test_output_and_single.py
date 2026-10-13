import base64
import io
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from watermark_studio.config import DeploymentConfig, ProcessingConfig
from watermark_studio.images import load_photo, save_output
from watermark_studio.server import create_app


def ready(client, finished=False):
    for _ in range(200):
        state = client.get("/health").json()
        if (state.get("job", {}).get("state") == "completed" if finished else state["stage"] == "ready"):
            return state
        time.sleep(.05)
    raise AssertionError(state)


@pytest.mark.parametrize("orientation", [1, 2, 3, 4, 5, 6, 7, 8])
def test_output_resize_normalizes_orientation_after_full_composition(tmp_path, orientation):
    source, target = tmp_path / "input.png", tmp_path / "output.png"
    exif = Image.Exif()
    exif[274] = orientation
    Image.new("RGBA", (120, 80), (30, 70, 100, 123)).save(source, exif=exif)
    rgb, meta, icc, alpha = load_photo(str(source))
    rgb.putalpha(alpha)
    save_output(rgb, source, target, meta, icc, ProcessingConfig(output_width=240, output_height=160))
    with Image.open(target) as output:
        assert output.size == (240, 160)
        assert output.getexif().get(274, 1) == 1
        assert output.getchannel("A").getextrema() == (123, 123)


def test_single_job_preview_limits_and_batch_scope(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    photo = folder / "photo.png"
    Image.effect_noise((2000, 1400), 100).convert("RGB").save(photo)
    cfg = DeploymentConfig(backend="mock", gpus=[0], allowed_roots=[str(tmp_path)],
                           state_dir=str(tmp_path / "state"))
    with TestClient(create_app(cfg)) as client:
        ready(client)
        assert client.get("/preview", params={"path": "/etc/passwd"}).status_code == 400
        info = client.get("/preview", params={"path": str(photo)}).json()
        encoded = base64.b64decode(info["jpeg"])
        assert len(encoded) <= 64 * 1024
        assert max(Image.open(io.BytesIO(encoded)).size) <= 1024
        assert info["size"] == [2000, 1400]
        payload = {"mode": "single", "input_file": str(photo), "input_dir": str(folder),
                   "preview_token": info["token"], "output_dir": str(tmp_path / "output"),
                   "options": {"concurrency": 1, "output_width": 320, "output_height": 240,
                               "preset": "custom", "regions": [{"box": [0, 0, .2, .2]}]}}
        payload["preview_token"] = "stale"
        assert client.post("/jobs", json=payload).status_code == 409
        payload["preview_token"] = info["token"]
        payload["output_dir"] = str(folder)
        assert client.post("/jobs", json=payload).status_code == 409
        payload["output_dir"] = str(tmp_path / "output")
        assert client.post("/jobs", json=payload).status_code == 202
        state = ready(client, True)
        assert state["job"]["succeeded"] == 1
        assert Image.open(tmp_path / "output/photo.png").size == (320, 240)
        payload = {"input_dir": str(folder), "output_dir": str(tmp_path / "batch"),
                   "options": {"concurrency": 1}}
        assert client.post("/jobs", json=payload).status_code == 202
        ready(client, True)
        assert client.app.state.runtime.job["options"]["scope"] == "full_image"
        assert client.app.state.runtime.job["options"]["regions"] == []


def test_bad_output_dimensions():
    with pytest.raises(ValueError):
        ProcessingConfig(output_width=100)
    with pytest.raises(ValueError):
        ProcessingConfig(output_width=16384, output_height=16384)

def test_parallel_scan_and_shape_sort_metadata(tmp_path):
    from watermark_studio.images import inference_shape, scan_images
    source = tmp_path / "source"
    source.mkdir()
    for name, orientation in [("a.jpg", 1), ("b.jpg", 6)]:
        exif = Image.Exif()
        exif[274] = orientation
        Image.new("RGB", (400, 300)).save(source / name, exif=exif)
    (source / "bad.png").write_bytes(b"invalid")
    (source / "ignore.txt").write_text("text")
    serial = scan_images(source, tmp_path / "out", False, workers=1)
    parallel = scan_images(source, tmp_path / "out", False, workers=4)
    assert serial == parallel
    assert inference_shape(source / "a.jpg", 1024) == (1184, 896)
    assert inference_shape(source / "b.jpg", 1024) == (896, 1184)


def test_summary_checkpoint_interval_keeps_final_progress(tmp_path):
    import json

    from watermark_studio.runtime import Runtime
    runtime = Runtime(DeploymentConfig(backend="mock", state_dir=str(tmp_path), report_interval=3))
    runtime.job = {"id": "checkpoint", "completed": 0, "succeeded": 0, "failed": 0,
                   "errors": [], "no_watermark": 0}
    runtime._persist()
    for i in range(2):
        runtime._record({"file": str(i), "ok": True})
    summary = tmp_path / "checkpoint/summary.json"
    assert json.loads(summary.read_text())["completed"] == 0
    assert runtime.job["completed"] == 2
    runtime._finish()
    assert json.loads(summary.read_text())["completed"] == 2
    assert len((tmp_path / "checkpoint/results.jsonl").read_text().splitlines()) == 2
