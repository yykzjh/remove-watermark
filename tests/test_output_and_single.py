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
