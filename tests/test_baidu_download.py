"""Check downloader inventory isolation and faithful, verified dataset layout."""
import io
import json
import runpy
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pytest
from PIL import Image

SCRIPT = Path(__file__).parents[1]/"scripts/download-baidu-images.py"


def arguments(monkeypatch, tmp_path, *extra):
    root, state = tmp_path/"images", tmp_path/"state"
    cookie = tmp_path/"private-cookie"
    cookie.write_text("BDUSS=unit-test-private")
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--cloud-root", "/photos", "--destination", str(root),
                                    "--state-dir", str(state), "--cookie-file", str(cookie), *extra])
    return root, state


@pytest.mark.parametrize("whole,selected", [(False, 1), (True, 2)])
def test_recursive_inventory_needs_no_client_and_does_not_log_credentials(monkeypatch, tmp_path, whole, selected):
    flags = ["--inventory-only"]+(["--include-non-images"] if whole else [])
    root, state = arguments(monkeypatch, tmp_path, *flags)
    listing = {
        "/photos": [{"path": "/photos/town", "isdir": 1}],
        "/photos/town": [{"path": "/photos/town/village", "isdir": 1}],
        "/photos/town/village": [{"path": "/photos/town/village/a.JPG", "isdir": 0, "size": 30},
                                {"path": "/photos/town/village/note.txt", "isdir": 0, "size": 3}],
    }
    def urlopen(request, timeout):
        assert request.get_header("Cookie") == "BDUSS=unit-test-private"
        directory = urllib.parse.parse_qs(urllib.parse.urlparse(request.full_url).query)["dir"][0]
        return io.BytesIO(json.dumps({"errno": 0, "list": listing[directory]}).encode())
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with pytest.raises(SystemExit) as stop:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert stop.value.code == 0
    summary = json.loads((state/"inventory-summary.json").read_text())
    assert summary["files"] == 2 and summary["images"] == 1 and summary["selected_files"] == selected
    assert len(json.loads((state/"directories.json").read_text())) == 3
    assert not (root/".download-state").exists()
    assert all("unit-test-private" not in p.read_text() for p in state.iterdir())


def test_completed_files_are_verified_and_empty_directories_preserved(monkeypatch, tmp_path):
    root, state = arguments(monkeypatch, tmp_path, "--include-non-images")
    village = root/"town/village"
    village.mkdir(parents=True)
    photo = village/"image.png"
    Image.new("RGB", (7, 5)).save(photo)
    note = village/"note.txt"
    note.write_text("original note")
    state.mkdir()
    (state/"manifest.json").write_text(json.dumps([
        {"path": "/photos/town/village/image.png", "size": photo.stat().st_size},
        {"path": "/photos/town/village/note.txt", "size": note.stat().st_size}]))
    (state/"directories.json").write_text(json.dumps(["/photos", "/photos/town/village/empty/deeper"]))
    runpy.run_path(str(SCRIPT), run_name="__main__")
    status = json.loads((state/"status.json").read_text())
    assert status["phase"] == "completed" and status["files"] == 2
    assert status["images"] == 1 and status["valid"] == 1
    assert (village/"empty/deeper").is_dir()
    assert note.read_text() == "original note"


def test_manifest_escape_is_rejected(monkeypatch, tmp_path):
    _, state = arguments(monkeypatch, tmp_path, "--inventory-only")
    state.mkdir()
    (state/"manifest.json").write_text(json.dumps([{"path": "/photos/../escape.png", "size": 2}]))
    with pytest.raises(ValueError, match="Unsafe manifest path"):
        runpy.run_path(str(SCRIPT), run_name="__main__")


def test_state_cannot_be_reused_for_another_source(monkeypatch, tmp_path):
    _, state = arguments(monkeypatch, tmp_path, "--inventory-only")
    state.mkdir()
    (state/"source.json").write_text(json.dumps({"cloud_root": "/another", "destination": "other"}))
    with pytest.raises(ValueError, match="another cloud root"):
        runpy.run_path(str(SCRIPT), run_name="__main__")


def test_matching_size_with_active_resume_marker_is_incomplete(tmp_path):
    complete = runpy.run_path(str(SCRIPT))["complete"]
    photo = tmp_path/"photo.jpg"
    photo.write_bytes(b"abcd")
    row = {"local": str(photo), "size": 4}
    assert complete(row)
    marker = Path(str(photo)+".BaiduPCS-Go-downloading")
    marker.write_bytes(b"unfinished ranges")
    assert not complete(row)
    marker.unlink()
    assert complete(row)
    row["size"] = 5
    assert not complete(row)
